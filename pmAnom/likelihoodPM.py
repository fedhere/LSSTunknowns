import sys,os, glob, time, astropy, warnings, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import *
from astropy.io import fits
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
        def __init__(self, metricName='LSPMmetric',f='g',surveyduration=10,snr_lim=5, sigma_threshold=1,
                    gap_selection=False,m5Col='fiveSigmaDepth', percentiles=[2.5,97.5,50],
                    prob_type='uniform',U=np.arange(-100,100,25),V=np.arange(-100,100,25),W=np.arange(-100,100,25),
                    mjdCol='observationStartMJD',filterCol='filter', seeingCol='seeingFwhmGeom',dataout=False, 
                    **kwargs):
            #opsim 
            self.mjdCol = mjdCol 
            self.m5Col = m5Col 
            self.seeingCol = seeingCol 
            self.filterCol= filterCol 
            #selection criteria
            self.surveyduration =surveyduration 
            self.snr_lim = snr_lim 
            self.f = f  
            self.sigma_threshold = sigma_threshold
            self.percentiles = percentiles
            self.gap_selection = gap_selection
            #simulation parameters
            self.U, self.V, self.W= U,V,W #Galactic coordinates in km/s
            self.prob_type = prob_type #if not uniform, is the path to the file with the costumized velocity distribution function
            #output
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
            '''
            label a position as Halo, Bulge or Disk.
            '''
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
            p_prob = np.array([rhoh,rhob,rhod])/np.nansum(np.array([rhoh,rhob,rhod]),axis=0)
            component = np.array(['H','B','D'])
            
            idx = np.array(p_prob == np.nanmax(p_prob, axis=0))
            res = np.array([component[idx[:,i]][0] for i in range(np.shape(idx)[1])])
            return res  
        def DF(self,V_matrix,component,R,z): 
            '''
            retrive the probability distribution in the given region of the Galaxy.
            '''
            P = np.empty(shape=(np.size(component), np.size(V_matrix[0,:])))
            #Halo
            iH = np.where(component == 'H')
            if np.size(iH)>0:
                v = np.sqrt(V_matrix[0,:]**2+V_matrix[1,:]**2+V_matrix[2,:]**2)
                vesc = 575 #km/s
                vsun =187.5 #km/s
                N=1.003
                P[iH,:]=4*N/vsun/np.sqrt(np.pi)*(v/vesc)**2*np.exp(-v**2/vsun)
            #Bulge
            iB = np.where(component == 'B') 
            if np.size(iB)>0:
                v = np.sqrt(V_matrix[0,:]**2+V_matrix[1,:]**2+V_matrix[2,:]**2)
                disp = 140 #km/s
                P[iB,:] = np.exp(-v**2/2/disp**2)/np.sqrt(np.pi)/disp
            #Disk
            iD = np.where(component == 'D')
            if np.size(iD)>0:
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
                Jz = v**2/2 + v**2/2*np.log(R[iD,np.newaxis]**2+z[iD,np.newaxis]**2/0.8**2)
                Ar = k * Jz /np.exp(2*q*(120 - R[iD,np.newaxis])/Rd)/sigmar0
                Az =  k * Jz /np.exp(2*q*(120 - R[iD,np.newaxis])/Rd)/sigmaz0 
                P[iD,:]=v**2/Jz*sigma*(1+np.tanh(R[iD,np.newaxis]*v/L0))*np.exp(-k*Jz/(sigmar0*np.exp(2*q*(120 - R[iD,np.newaxis])/Rd))**2)/(np.pi*k*sigmar0*np.exp(2*q*(120 - R[iD,np.newaxis])/Rd))
            return P
        def sigma_slope(self,x, sigma_y):
            """
            Calculate the uncertainty in fitting a line, as
            given by the spread in x values and the uncertainties
            in the y values.

            Parameters
            ----------
            x : numpy.ndarray
                The x values of the data
            sigma_y : numpy.ndarray
                The uncertainty in the y values

            Returns
            -------
            float
                The uncertainty in the line fit
            """
            w = 1/sigma_y*1/sigma_y
            denom = np.sum(w)*np.einsum('i,ij->j',x**2,w.T)-np.einsum('i,ij->j',x,w.T)**2
            select= np.where(denom > 0)
            res=np.sqrt(np.sum(w,axis=1)[select]/denom[select] )*365.25*1e3
            return res                   
        def run(self, dataSlice, slicePoint=None): 
            np.random.seed(2500) 
            obs = np.where((dataSlice['filter']==self.f) & (dataSlice[self.mjdCol]<min(dataSlice[self.mjdCol])+365*self.surveyduration)) 
            d = np.array(self.simobj['d'])
            M = np.array(self.simobj['MAG'])
            fieldRA, fieldDec = np.radians(np.mean(dataSlice['fieldRA'])), np.radians(np.mean(dataSlice['fieldDec'])) 
            z, R =d*np.sin(fieldRA), d*np.cos(fieldRA)
            component = self.position_selection(R,z)
            mjd = dataSlice[self.mjdCol][obs]
            fwhm = dataSlice[self.seeingCol][obs]
            V_galactic=np.vstack((self.U,self.V,self.W))
            Pv=self.DF(V_galactic,component,R,z) 
            marg_P= np.nanmean(Pv/np.nansum(Pv,axis=0), axis=0)
            marg_P/=np.nansum(marg_P)
            vel_idx=np.random.choice(np.arange(0,len(V_galactic[0,:]),1)[np.isfinite(marg_P)],p=marg_P[np.isfinite(marg_P)], size = 3)
            vT_unusual = V_galactic[0,vel_idx][2]
            if self.prob_type=='uniform':
                p_vel_unusual= uniform(-100,100) 
                v_unusual = p_vel_unusual.rvs(size=(3,np.size(d)))
                vT = v_unusual[2,:]
            else:
                p_vel_un = pd.read_csv(self.prob_type)
                vel_idx = np.random.choice(p_vel_un['vel'],p=p_vel_un['fraction']/np.sum(p_vel_un['fraction']),size=3)
                vT_unusual = V_galactic[0,vel_idx][2]
            #vel_unusual = V_galactic[0,vel_idx]
            
            direction =np.random.choice((-1, 1))
            mu = direction*vT/4.75/d
            mu_unusual = direction*vT_unusual/4.75/d 
            
            if len(dataSlice[self.m5Col][obs])>2: 
                
                # select objects above the limit magnitude threshold 
                snr = m52snr(M[:, np.newaxis],dataSlice[self.m5Col][obs])
                row, col = np.where(snr>self.snr_lim)
                if self.gap_selection:
                    Times = np.sort(mjd)
                    dt = np.array(list(combinations(Times,2)))
                    DeltaTs = np.absolute(np.subtract(dt[:,0],dt[:,1]))            
                    DeltaTs = np.unique(DeltaTs)
                    if np.size(DeltaTs)>0:
                                 dt_pm = 0.05*np.amin(dataSlice[self.seeingCol][obs])/muf[np.unique(row)]
                                 selection = np.where((dt_pm>min(DeltaTs)) & (dt_pm<max(DeltaTs))& (pm >sigmapm))
                else:
                      selection = np.unique(row)
                precis = astrom_precision(dataSlice[self.seeingCol][obs], snr[row,:])
                sigmapm= self.sigma_slope(dataSlice[self.mjdCol][obs], precis)*365.25*1e3

                if np.size(selection)>0:
                    pa= np.random.uniform(0,2*np.pi,len(mu_unusual[selection]))
                    pm_alpha, pm_delta = mu[selection]*np.sin(pa), mu[selection]*np.cos(pa)
                    pm_un_alpha, pm_un_delta = mu_unusual[selection]*np.sin(pa), mu_unusual[selection]*np.cos(pa)                    
                    #p_min,p_max,p_mean = self.percentiles[0],self.percentiles[1],self.percentiles[2]                    
                    mu = mu[selection]*1e3
                    mu_unusual = mu_unusual[selection]*1e3
                    variance_k = np.array([np.std(mu[np.where(component[selection]==p)]) for p in ['H', 'D','B']])
                    variance_mu = np.std(mu)
                    sigmaL = np.sqrt(np.prod(variance_k, where=np.isfinite(variance_k))**2+variance_mu**2+np.nanmedian(sigmapm)**2)
                    unusual= np.where((mu_unusual<np.mean(mu_unusual)-self.sigma_threshold* sigmaL/2) | (mu_unusual>np.mean(mu_unusual)+self.sigma_threshold* sigmaL/2))             
                    res= np.size(unusual)/np.size(selection)
                    
                    
                    if self.dataout:
                        dic= {'detected': res,
                          'pixID': radec2pix(nside=16, ra=np.radians(fieldRA), dec=np.radians(fieldDec)),
                          'PM': pd.DataFrame({'pm_alpha':pm_alpha, 'pm_delta':pm_delta}),
                          'PM_un': pd.DataFrame({'pm_alpha':pm_un_alpha, 'pm_delta':pm_un_delta})}
                        return dic 
                    else: 
                        return res  
