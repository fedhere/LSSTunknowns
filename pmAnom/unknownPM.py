import sys,os, glob, time, astropy, warnings, pickle
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import pandas as pd
from scipy.stats import *
from astropy.stats import histogram
from astropy.io import fits
import sklearn.mixture.gaussian_mixture as GMM
from builtins import zip
### LSST dependencies 
from lsst.sims.maf.metrics import BaseMetric
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
import lsst.sims.maf.plots as plots
from lsst.sims.maf.utils.mafUtils import radec2pix
from lsst.sims.maf.utils import m52snr, astrom_precision
from opsimUtils import *

class LSPMmetric(BaseMetric): 
        def __init__(self, metricName='LSPMmetric',f='g',surveyduration=10,snr_lim=5,mag_lim=[17,25],percentiles=[2.5,97.5,50], 
                    U=np.arange(-100,100,25),V=np.arange(-100,100,25),W=np.arange(-100,100,25),unusual='uniform',treshold='1d',m5Col='fiveSigmaDepth',  
                    mjdCol='observationStartMJD',filterCol='filter', seeingCol='seeingFwhmGeom', nexp= 1,dataout=False, 
                    **kwargs):
            self.mjdCol = mjdCol 
            self.m5Col = m5Col 
            self.seeingCol = seeingCol 
            self.filterCol= filterCol 
            self.surveyduration =surveyduration 
            self.percentiles = percentiles
            self.snr_lim = snr_lim 
            self.mag_lim = mag_lim
            self.f = f 
            self.nexp = nexp 
            self.U=U 
            self.V=V 
            self.W=W 
            self.unusual=unusual
            self.treshold=treshold
            self.dataout = dataout 
             # to have as output all the simulated observed data set dataout=True, otherwise the relative error for  
             # each helpix is estimated 
            sim = pd.read_csv('hyperstar_uniform.csv', usecols=['MAG','MODE','d','PM','PM_out'])
            self.simobj = sim
            if self.dataout: 
                super(LSPMmetric, self).__init__(col=[self.mjdCol,self.filterCol, self.m5Col,self.seeingCol, 'night' ],metricDtype='object', units='', metricName=metricName, 
                                                      **kwargs) 
            else: 
                super(LSPMmetric, self).__init__(col=[self.mjdCol,self.filterCol,  self.m5Col,self.seeingCol, 'night'], 
                                                            units='Proper Motion relative error', metricName=metricName, 
                                                             **kwargs) 
          
            np.seterr(over='ignore',invalid='ignore')
         # typical velocity distribution from litterature (Binney et Tremain- Galactic Dynamics)
            
        def position_selection(self,R,z):
            # costants
            ab = 1 #kpc
            ah = 1.9 #kpc
            alphab = 1.8
            alphah = 1.63
            betah =2.17
            rb = 1.9 #Kpc
            qb = 0.6
            qh = 0.8
            z0 = 0.3 #kpc
            z1 = 1 #kpc
            Rd = 3.2 #kpc
            rho0b = 0.3*10**(-9) #Mskpc^-3
            rho0h = 0.27*10**(-9) #Mskpc^-3
            sigma = 300*10**(-6) #Mskpc^-2
            alpha1 = 0.5
            # parametes
            mb = np.sqrt(R**2+z**2/qb**2)
            mh = np.sqrt(R**2+z**2/qh**2)
            alpha0 =1-alpha1
            
            rhoh = rho0h*(mh/ah)**alphah*(1+mh/ah)**(alphah-betah)
            rhob = rho0b*(mb/ab)**(-alphab)*np.exp(-mb/rb)
            rhod = sigma*np.exp(R/Rd)*(alpha0/(2*z0)*np.exp(-np.absolute(z)/z0)+alpha1/(2*z1)*np.exp(-np.absolute(z)/z1))
            densities = np.array([rhoh,rhob,rhod])/np.sum(np.array([rhoh,rhob,rhod]))
            mode = np.array(['H','B','D'])
            idx = np.where(densities == np.amax(densities))
            return mode[idx]
        
        def DF(self,V_matrix,mode,R,z):                
            if mode == 'H':
                v = np.sqrt(V_matrix[0,:]**2+V_matrix[1,:]**2+V_matrix[2,:]**2)
                vesc = 575 #km/s
                vsun =187.5 #km/s
                N=1.003
                P=4*N/vsun/np.sqrt(np.pi)*(v/vesc)**2*np.exp(-v**2/vsun)
            if mode == 'B': 
                v = np.sqrt(V_matrix[0,:]**2+V_matrix[1,:]**2+V_matrix[2,:]**2)
                disp = 140 #km/s
                P = np.exp(-v**2/2/disp**2)/np.sqrt(np.pi)/disp
            if mode == 'D':
                k= 0.25
                q =0.45
                Rd = 3.2 #kpc
                sigmaz0= 19 #km/s
                sigmar0= 33.5 #km/s
                beta = 0.33
                L0 = 10 #km/s
                sigma = 300*10**(-6) #Ms/kpc**2
            
                # parameters
                v = V_matrix[2,:]
                Jz = v**2/2 + v**2/2*np.log(R**2+z**2/0.8**2) 
                Ar = k * Jz /np.exp(2*q*(120 - R)/Rd)/sigmar0
                Az =  k * Jz /np.exp(2*q*(120 - R)/Rd)/sigmaz0                
                P = v**2/Jz*sigma*(1+np.tanh(R*v/L0))*np.exp(-k*Jz/(sigmar0*np.exp(2*q*(120 - R)/Rd))**2)/(np.pi*k*sigmar0*np.exp(2*q*(120 - R)/Rd))
                
            return P
          
                           
        def run(self, dataSlice, slicePoint=None): 
            np.random.seed(2500) 
            obs = np.where((dataSlice['filter']==self.f) & (dataSlice[self.mjdCol]<min(dataSlice[self.mjdCol])+365*self.surveyduration)) 
            d = self.simobj['d']
            M = np.array(self.simobj['MAG'])
            mu = np.array(self.simobj['PM'])
            muf = np.array(self.simobj['PM_out'])
            mjd = dataSlice[self.mjdCol][obs]
            if len(dataSlice[self.m5Col][obs])>2: 
                
                # select objects above the limit magnitude threshold 
                snr = m52snr(M[:, np.newaxis],dataSlice[self.m5Col][obs])
                row, col =np.where(snr>self.snr_lim)
                precis = astrom_precision(dataSlice[self.seeingCol][obs], snr)
                sigmapm=sigma_slope_arr(dataSlice[self.mjdCol][obs], precis)*365.25*1e3

                #select the objects which displacement can be detected
                Times = list(mjd)
                DeltaTs = []
                while np.size(Times)>1:
                    for d in range(len(Times)-1):
                        DeltaTs.append(Times[d]-Times[d+1])
                    Times.remove(Times[0])
                DeltaTs.sort()
                DeltaTs = np.array(DeltaTs)
                if np.size(DeltaTs)>0:
                    dt_pm = 0.05*np.median(dataSlice[self.seeingCol])/muf[np.unique(row)]
                    selection = np.where((dt_pm>DeltaTs[0]) & (dt_pm<DeltaTs[-1]))

                    if np.size(selection)>0:
                        pa= np.random.uniform(0,2*np.pi,len(mu[selection]))
                        pm_alpha, pm_delta = mu[selection]*np.sin(pa), mu[selection]*np.cos(pa)
                        pm_un_alpha, pm_un_delta = muf[selection]*np.sin(pa), muf[selection]*np.cos(pa)                    
                        p_min,p_max,p_mean = self.percentiles[0],self.percentiles[1],self.percentiles[2]
                        if self.treshold =='1d':
                            mu_min,mu_max, mu_mean = np.percentile(np.array(mu)[selection],[p_min,p_max,p_mean])
                            OK = np.isfinite(np.array(muf)[selection])
                            pm = muf[selection][OK]
                            s = sigmapm[selection][OK]
                            muf_index = np.where((pm < mu_min)| (pm >mu_max))
                            ii = (pm[muf_index]+ s[muf_index]<mu_min)|(pm[muf_index]+ s[muf_index]>mu_max)
                            kk = (pm[muf_index]- s[muf_index]<mu_min)|(pm[muf_index]- s[muf_index]>mu_max)
                            u = ii & kk
                            unusual = np.size(muf_index)
                            if unusual == 0:
                                res = 0
                            else:
                                res = len(np.where(u==True))/unusual
                        elif self.treshold =='2d':
                            mu_min,mu_max, mu_mean = np.percentile(np.array([pm_alpha, pm_delta ]),[p_min,p_max,p_mean],axis=0)
                            center = [np.concatenate((mu_min,mu_max)).sum()/np.size(np.concatenate((mu_min,mu_max))), 
                                      pm_delta.sum()/np.size(pm_delta)]
                            radius = np.sqrt((np.concatenate((mu_min,mu_max))-center[0])**2+(np.concatenate((pm_delta,pm_delta)) -center[1])**2)
                            distance = np.sqrt((center[0]- pm_un_alpha)**2+(center[1]-pm_un_delta)**2)
                            outcontour= distance>radius.max()
                            u = np.where(outcontour==True)
                            res= np.size(u)/np.size(distance)
                        else:
                            raise Exception('no treshold mode has been selected')


                        fieldRA = np.mean(dataSlice['fieldRA']) 
                        fieldDec = np.mean(dataSlice['fieldDec'])
                        #dic = {'detected': res,
                        #        'pixID': radec2pix(nside=16, ra=np.radians(fieldRA), dec=np.radians(fieldDec))}  
                        dic= {'detected': res,
                              'pixID': radec2pix(nside=16, ra=np.radians(fieldRA), dec=np.radians(fieldDec)),
                              'PM': pd.DataFrame({'pm_alpha':pm_alpha, 'pm_delta':pm_delta}),
                              'PM_un': pd.DataFrame({'pm_alpha':pm_un_alpha, 'pm_delta':pm_un_delta})}
                                #'PM': np.array(mu)[selection], 
                                #'PM_OUT': pm}
                        if self.dataout:
                            return dic 
                        else: 
                            return res 