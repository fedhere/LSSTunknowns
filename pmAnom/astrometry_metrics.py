import sys,os, glob, time, astropy, warnings, pickle
sys.path.append('/data/fragosta/work/lsst/sims_maf_contrib-master/')
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import pandas as pd
from scipy.stats import *
import scipy.optimize as so 
from astropy.stats import histogram
from astropy.io import fits
from sklearn.mixture import GaussianMixture
from builtins import zip
### LSST dependencies 
from lsst.sims.maf.metrics import BaseMetric
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.db as db
import lsst.sims.maf.plots as plots
from lsst.sims.maf.utils.mafUtils import radec2pix
from lsst.sims.maf.utils import m52snr, astrom_precision, sigma_slope
sys.path.append('/home/idies/LSST_OpSim/Scripts_NBs/')
from opsimUtils import *
from itertools import product 

__all__ = ['sigma_slope_arr','DF','position_selection','simulate_pm','getDataMetric',
           'LSPMmetric','reducedPM','TransienPM','RPMD_plot',
           'find_confidence_interval','PMContourPlot']


def DF(V_matrix,mode,R,z): 
            '''
            retrive the probability distribution in the given region of the Galaxy.
            '''
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
        
def position_selection(R,z):
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
            densities = np.array([rhoh,rhob,rhod])/np.sum(np.array([rhoh,rhob,rhod]))
            mode = np.array(['H','B','D'])
            idx = np.where(densities == np.amax(densities))
            return mode[idx]
        
def simulate_pm(nexp=1000,M_min=15,M_max=25, prob_type='uniform',U=np.arange(-100,100,25),V=np.arange(-100,100,25),W=np.arange(-100,100,25)):
    '''
    This module allows to simulate nexp stars given the velocity distribution matrix.
    nexp = the 
    prob_type = if "uniform" use a uniform distribution to pick velocities from the distribution, otherwise a filename of a customized array of (probability, velocity) has to be given.
    U,V,W = velocity components in the Galaxy reference. 
    
    We assign to each object an absolute magnitude following a population synthesis model (Percival et al. 2008);
    absMag_g-dist.csv, time-dist.csv are files where the description of the population synthesis model are.
    '''
    np.random.seed(5000)
    g_dist = pd.read_csv('absMag_g-dist.csv', usecols=['mag_g', 'frequency'], index_col=False)
    age_dist = pd.read_csv('time-dist.csv', usecols=['age', 'frequency'], index_col=False)
    fout = open('hypervelocity_cut.csv', 'w+')
    fout.write('MAG'+','+'MODE'+','+'d'+','+'PM'+','+'PM_out'+'\n')
    n=0
    while n < nexp:
        g_abs = np.random.choice(g_dist['mag_g'],p=g_dist['frequency']) # magAB 
        age =  np.random.choice(age_dist['age'],p=age_dist['frequency']) # magAB 
        R = np.random.uniform(0, 120) # kpc 
        z = np.random.uniform(-120*0.8, 120*0.8) # kpc 
        d = np.sqrt(R**2+z**2)*10**(3)
        M = g_abs +5*np.log10(d)-5 
        if (M>M_min) and (M< M_max):  
            n+=1
            mode = position_selection(R,z)
            V_matrix=np.vstack((U,V,W))
            velmat = np.empty_like(V_matrix[0,:])
            veloutmat = np.empty_like(V_matrix[0,:])
            Pv=DF(V_matrix,mode,R,z)  
            vel_idx=np.random.choice(np.arange(0,len(V_matrix[0,:]),1)[np.isfinite(Pv)],p=Pv[np.isfinite(Pv)]/np.sum(Pv[np.isfinite(Pv)]), size = 3)
            if prob_type=='uniform':
                vel_out= uniform(-100,100) 
                veloutmat = vel_out.rvs(size=3)
            else:
                vel_out = pd.read_csv(prob_type)
                veloutmat = np.random.choice(vel_out['vel'],p=vel_out['fraction']/np.sum(vel_out['fraction']),size=3)
                    
            velmat = V_matrix[0,vel_idx]
            vT = velmat[2]
            vfT = veloutmat[2]
            direction =np.random.choice((-1, 1))
            mu = direction*vT/4.75/d 
            muf = direction*vfT/4.75/d 
            fout.write('{}'.format(M)+','+'{}'.format(mode)+','+'{}'.format(d)+','+'{}'.format(mu)+','+'{}'.format(muf)+' \n')    
        else:
            continue
    return print('simulated pm of {} objects'.format(nexp))   
    
    
class getDataMetric(metrics.BaseMetric):
    """
    extract data from database by column names, 
    combined with UniSlicer()
    """
    
    def __init__(self, colname=['expMJD', 'airmass'], **kwargs):
        self.colname = colname
        super().__init__(col=colname, metricDtype='object', **kwargs)
        
        
    def run(self, dataSlice, slicePoint=None):
        
        # return dataSlice directly
        result = dataSlice
        
        return result
        
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
            fieldRA, fieldDec = np.mean(dataSlice['fieldRA']), np.mean(dataSlice['fieldDec']) 
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
                                 dt_pm = 0.05*np.amin(dataSlice[self.seeingCol])/muf[np.unique(row)]
                                 selection = np.where((dt_pm>min(DeltaTs)) & (dt_pm<max(DeltaTs)))
                else:
                      selection = np.unique(row)
                precis = astrom_precision(dataSlice[self.seeingCol][obs], snr[row,:])
                sigmapm= self.sigma_slope(dataSlice[self.mjdCol][obs], precis)*365.25*1e3

                if np.size(selection)>0:
                    pa= np.random.uniform(0,2*np.pi,len(mu_unusual[selection]))
                    pm_alpha, pm_delta = mu[selection]*np.sin(pa), mu[selection]*np.cos(pa)
                    pm_un_alpha, pm_un_delta = mu_unusual[selection]*np.sin(pa), mu_unusual[selection]*np.cos(pa)                    
                    p_min,p_max,p_mean = self.percentiles[0],self.percentiles[1],self.percentiles[2]                    
                    mu = mu[selection]
                    mu_unusual = mu_unusual[selection]
                    variance_k = np.array([np.std(mu[np.where(component[selection]==p)]) for p in ['H', 'D','B']])
                    variance_mu = np.std(mu)
                    sigmaL = np.sqrt(np.prod(variance_k, where=np.isfinite(variance_k))**2+variance_mu**2)
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
          
          


class reducedPM(BaseMetric):         
        def __init__(self, fits_filename = 'sa93.all.fits', snr_lim=5,mode=None, MagIterLim=[0,1,1], surveyduration=10,
                      metricName='reducedPM',m5Col='fiveSigmaDepth',gap_selection=False,real_data= True,
                      mjdCol='observationStartMJD',filterCol='filter', seeingCol='seeingFwhmGeom',dataout=True,**kwargs): 
            self.mjdCol = mjdCol 
            self.m5Col = m5Col 
            self.seeingCol = seeingCol 
            self.filterCol = filterCol 
            self.snr_lim = snr_lim 
            self.fits_filename = fits_filename  
            self.dataout = dataout 
            self.mode = mode 
            self.gap_selection=gap_selection
            self.MagIterLim = MagIterLim 
            self.surveyduration = surveyduration 
            self.real_data = real_data
             # to have as output all the simulated observed data set dataout=True, otherwise the relative error for  
             # each helpix is estimated 
            if self.dataout: 
                super(reducedPM, self).__init__(col=[self.mjdCol,self.filterCol, self.m5Col,self.seeingCol, 'night' ],metricDtype='object', units='', metricName=metricName, 
                                                      **kwargs) 
            else: 
                super(reducedPM, self).__init__(col=[self.mjdCol,self.filterCol,  self.m5Col,self.seeingCol, 'night' ], 
                                                            units='', metricDtype='float',metricName=metricName, 
                                                             **kwargs) 
            
            data_sag93 = fits.open(self.fits_filename)
            table = data_sag93[1].data    
            mu_sag= np.transpose(np.sqrt(table['MUX']**2+table['MUY']**2))
            M_sag = np.transpose(table['GMAG'])
            self.mu_sag = mu_sag
            self.mag_sag = M_sag
            self.gr = np.transpose(table['GMAG']-table['RMAG'])
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
            #pm = np.array(self.data['PM_OUT'])
            #mag = np.array(self.data['MAG'])
            obs = np.where(dataSlice[self.mjdCol]<min(dataSlice[self.mjdCol])+365*self.surveyduration)
            
            deltamag= np.arange(self.MagIterLim[0],self.MagIterLim[1],self.MagIterLim[2])
            out = {}
            for dm in deltamag: 
                
                    if self.mode == 'distance': 
                        pmnew= self.mu_sag /(10**(dm/5)) 
                        mag = self.mag_sag + dm
                    elif self.mode == 'density': 
                        pmnew= self.mu_sag  
                        mag = self.mag_sag + dm
                    else: 
                        print('##### ERROR: the metric is not implemented for this mode.')
                        
                    mjd = dataSlice[self.mjdCol][obs]
                    flt = dataSlice[self.filterCol][obs]
                    if ('g' in flt) and ('r' in flt):
                        
                        # select objects above the limit magnitude threshold 
                        snr = m52snr(mag[:, np.newaxis],dataSlice[self.m5Col][obs])
                        row, col =np.where(snr>self.snr_lim)
                        if self.gap_selection:
                            Times = np.sort(mjd)
                            dt = np.array(list(combinations(Times,2)))
                            DeltaTs = np.absolute(np.subtract(dt[:,0],dt[:,1]))            
                            DeltaTs = np.unique(DeltaTs)
                            if np.size(DeltaTs)>0:
                                         dt_pm = 0.05*np.amin(dataSlice[self.seeingCol])/pmnew[np.unique(row)]
                                         selection = np.where((dt_pm>min(DeltaTs)) & (dt_pm<max(DeltaTs)))
                        else:
                            selection = np.unique(row)
                        precis = astrom_precision(dataSlice[self.seeingCol][obs], snr[row,:])
                        sigmapm= self.sigma_slope(dataSlice[self.mjdCol][obs], precis)
                        Hg = mag[selection]+5*np.log10(pmnew[selection])-10
                        sigmaHg = np.sqrt((mag[selection]/m52snr(mag[selection],np.median(dataSlice[self.m5Col])))**(2)+ (4.715*sigmapm[selection]/np.ceil(pmnew[selection]))**2) 
                        sigmag = np.sqrt((mag[selection]/m52snr(mag[selection],np.median(dataSlice[self.m5Col])))**2+((mag[selection]-self.gr[selection])/m52snr((mag[selection]-self.gr[selection]),np.median(dataSlice[self.m5Col])))**2)
                        err_ellipse = np.pi*sigmaHg*sigmag
                        if self.dataout:
                            CI = np.array([np.nansum((([gr-gcol ])/sigmag)**2 + ((Hg-h)/sigmaHg)**2 <= 1)/np.size(pmnew[selection]) for (gcol,h) in zip(gr,Hg)])                      

                            out[dm] = {'CI':CI,'alpha':np.size(pmnew[selection])/np.size(pmnew) ,'err':err_ellipse,'Hg':Hg,'gr':gr, 'sigmaHg':sigmaHg,'sigmagr':sigmag}
                        else:
                            out[dm] = {'alpha':np.size(pmnew[selection])/np.size(pmnew) ,'err':err_ellipse}
                    else:
                        if self.dataout:
                            out[dm] = {'CI':0,'alpha':0,'err':0,'Hg':Hg,'gr':gr, 'sigmaHg':sigmaHg,'sigmagr':sigmag} 
                        else:
                            out[dm] = {'alpha':0 ,'err':0}
            if self.dataout: 
                return out  
            else:
                if ('g' in flt) and ('r' in flt):
                    res = out[dm]['alpha']/np.nanmean(out[dm]['err'][np.isfinite(out[dm]['err'])])
                    
                    return res  

                
                
                
                
class TransienPM(BaseMetric): 
     #    Generate a population of transient objects and see what is its proper motion  , 
    def __init__(self, metricName='TransienPM', f='g', snr_lim=5,m5Col='fiveSigmaDepth',  
                  mjdCol='observationStartMJD',filterCol='filter',seeingCol='seeingFwhmGeom', surveyduration=10, **kwargs): 
            self.mjdCol = mjdCol 
            self.seeingCol= seeingCol 
            self.m5Col = m5Col 
            self.filterCol = filterCol 
            self.snr_lim = snr_lim 
            self.f = f
            self.surveyduration = surveyduration  
            sim = pd.read_csv('/home/idies/workspace/Storage/fragosta/persistent/LSST_OpSim/Scripts_NBs/simulation_pm.csv', usecols=['MAG','MODE','d','PM','PM_out'])
            self.simobj = sim
            super(TransienPM, self).__init__(col=[self.mjdCol, self.m5Col,self.seeingCol, self.filterCol], 
                                                       units='Fraction Detected', 
                                                       metricName=metricName, **kwargs) 
      
         # typical velocity distribution from litterature (Binney et Tremain- Galactic Dynamics) 
      
    def lightCurve(self, t, t0, peak, duration, slope): 
     #      A simple top-hat light curve., 
     #         
     #        Parameters , 
     #        ---------- , 
     #        t : array , 
     #            Times to generate lightcurve points (mjd) , 
     #        t0 : float , 
     #            Initial time (mjd) , 
     #        m_r_0 : float , 
     #            initial r-band brightness (mags) , 
            #lightcurve = np.zeros((np.size(t), np.size(t0)), dtype=float) + 99.
            T_matrix=np.ones((np.size(t0),np.size(t)))*t
            T0s_matrix=np.ones((np.size(t),np.size(t0)))*t0
            M = np.ones((np.size(t),np.size(t0)))*peak
            S = np.ones((np.size(t),np.size(t0)))*slope
     # Select only times in the lightcurve duration window , 
            T_good=np.where((T_matrix>=T0s_matrix.T) & (T_matrix<=np.array(T0s_matrix+duration).T),T_matrix,0) 
            T0s_good=np.where((T_matrix.T>=T0s_matrix) & (T_matrix.T<=np.array(T0s_matrix+duration)),T0s_matrix,0)
            M_good=np.where((T_matrix.T>=T0s_matrix) & (T_matrix.T<=np.array(T0s_matrix+duration)),M,0)
            S_good=np.where((T_matrix.T>=T0s_matrix) & (T_matrix.T<=np.array(T0s_matrix+duration)),S,0)
            lightcurve = M_good.T + S_good.T*(T_good-T0s_good.T) 
            return lightcurve 
      
    def run(self,  dataSlice, slicePoint=None): 
            pm = np.array(self.simobj['PM_out'])
            mag = np.array(self.simobj['MAG'])
            obs = np.where(dataSlice[self.mjdCol]<min(dataSlice[self.mjdCol])+365*self.surveyduration)
            np.random.seed(5000)
            mjd = dataSlice[self.mjdCol][obs]
            flt = dataSlice[self.filterCol][obs]
            if (self.f in flt):
                snr = m52snr(mag[:, np.newaxis],dataSlice[self.m5Col][obs])
                row, col =np.where(snr>self.snr_lim)
                
                Times = np.sort(mjd)
                dt = np.array(list(combinations(Times,2)))
                DeltaTs = np.absolute(np.subtract(dt[:,0],dt[:,1]))            
                DeltaTs = np.unique(DeltaTs)
                if np.size(DeltaTs)>0:
                             dt_pm = 0.05*np.amin(dataSlice[self.seeingCol])/pm[np.unique(row)]
                             selection = np.where((dt_pm>min(DeltaTs)) & (dt_pm<max(DeltaTs)))

                objRate = 0.7 # how many go off per day
                nObj=np.size(pm[selection])
                m0s = mag[selection]
                t = dataSlice[self.mjdCol][obs] - dataSlice[self.mjdCol].min() 
                detected = 0 
         # Loop though each generated transient and decide if it was detected , 
         # This could be a more complicated piece of code, for example demanding  , 
         # A color measurement in a night. , 
                durations = dt_pm[selection]
                slopes = np.random.uniform(-3,3,np.size(selection))
                t0s = np.random.uniform(0,self.surveyduration,nObj)
                lcs = self.lightCurve(t, t0s, m0s,durations, slopes) 
                good = m52snr(lcs,dataSlice[self.m5Col][obs])> self.snr_lim
                detectedTest = good.sum(axis=0)
                detected = np.sum(detectedTest>2)
                
                 # Return the fraction of transients detected , 
                if float(nObj) == 0:
                    A = np.inf 
                else: 
                    A=float(nObj) 
                    res = float(detected)/A            
                    #print('detected fraction:{}'.format(res)) 
                    return res
                
                
                
def RPMD_plot(cind_dens,H_dens,c_dens,cind_dist,H_dist,c_dist,alpha_dens,err_ell_dens,alpha_dist,err_ell_dist,version='v1.4', xmin=-0.1,xmax=1.999, ymin=5,ymax=21,vmin=0,vmax=0.6,runs=[]):
    '''
    This function allows to plot the proper motion measurement in the Hg-color phase space
    '''
    import matplotlib.pyplot as plt
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)

    fig = plt.figure(figsize=(20,20))

    subfigs = fig.subfigures(4, 4, wspace=0.07,hspace=0)

    for outerind, (subfig,(mag, key)) in enumerate(zip(subfigs.flat,product([0,1,2,3],runs))):
        n=key.split('_')
        if version in n:
            n.remove(version)
            n.remove('10yrs')
        sep='_'
        name= sep.join(n)

        axs = subfig.subplots(2, 1,sharex=True)
        for innerind, (ax,mode) in enumerate(zip(axs.flat,['density','distance'])):
                    if (mode=='density') and (outerind==0):
                        zero_mask = np.where(cind_dens[key][mag]==0)
                        non_zero_mask = np.where(c_dens[key][mag]!=0)
                        gplot=np.dstack([H_dens[key][mag],c_dens[key][mag]])
                        gm = GaussianMixture(n_components=2, random_state=0).fit(gplot[0])
                        N=gm.weights_[0]*100
                        ax.scatter(y=H_dens[key][mag][non_zero_mask],x=c_dens[key][mag][non_zero_mask],
                                   c=cind_dens[key][mag][non_zero_mask],cmap='Spectral_r',s=40,vmin=vmin,vmax=0.6,alpha=0.4)
                        ax.set_xlim([xmin,xmax])
                        ax.set_ylim([ymin,ymax])
                        ax.set_yticks(np.arange(ymin, ymax+0.01, 4.5))
                        ax.annotate(r'$\alpha={} ,$'.format(np.round(alpha_dens[key][mag][0],2)),[0.05,8.9], size=20)
                        ax.annotate(r'$N={}$'.format(np.round(N,2)),[1.4,8.9], size=20)
                        ax.annotate(r'$\Delta m={} mag,$'.format(mag),[0.05,6.9], size=20)
                        ax.annotate(r'$CI = {} mag^2$'.format(np.round(np.median(err_ell_dens[key][mag])),2),[0.05,10.9], size=20)
                        ax.invert_yaxis()
                        ax.yaxis.set_label_coords(-0.15,0)
                        ax.set_ylabel('Hg',size=22)
                        ax.set_xlabel('')
                        ax.set_title(f'{name}',size=22)
                        ax.get_xaxis().set_visible(False)


                    if mode=='distance'and outerind==0:
                        zero_mask = np.where(cind_dist[key][mag]==0)
                        non_zero_mask = np.where(c_dist[key][mag]!=0)
                        gplot=np.dstack([H_dens[key][mag],c_dens[key][mag]])
                        gm = GaussianMixture(n_components=2, random_state=0).fit(gplot[0])
    
                        N=gm.weights_[0]*100
                        ax.scatter(y=H_dist[key][mag][non_zero_mask],x=c_dist[key][mag][non_zero_mask],
                                   c=cind_dist[key][mag][non_zero_mask],cmap='Spectral_r',s=40,vmin=vmin,vmax=0.6,alpha=0.4)
                        ax.set_xlim([xmin,xmax])
                        ax.set_ylim([ymin,ymax])
                        ax.set_yticks(np.arange(ymin, ymax+0.01, 4.5))
                        ax.annotate(r'$\alpha={} ,$'.format(np.round(alpha_dist[key][mag][0],2)),[0.04,8.9], size=20)
                        ax.annotate(r'$N={}$'.format(np.round(N,2)),[1.4,8.9], size=20)
                        ax.annotate(r'$d_{new}=$'+'{} kpc,'.format(np.round(28*10**(mag/5),1)),[0.04,6.9], size=20)
                        ax.annotate(r'$CI = {} mag^2$'.format(np.round(np.median(err_ell_dist[key][mag])),2),[0.04,10.9], size=20)
                        ax.invert_yaxis()
                        ax.set_ylabel('')
                        ax.set_xlabel('')
                        ax.get_xaxis().set_visible(False)
                    if (mode=='density') and (outerind>=4) and (outerind not in [0,4,8,12,13,14,15]):
                        zero_mask = np.where(cind_dens[key][mag]==0)
                        non_zero_mask = np.where(c_dens[key][mag]!=0)
                        gplot=np.dstack([H_dens[key][mag],c_dens[key][mag]])
                        gm = GaussianMixture(n_components=2, random_state=0).fit(gplot[0])
    
                        N=gm.weights_[0]*100
                        ax.scatter(y=H_dens[key][mag][non_zero_mask],x=c_dens[key][mag][non_zero_mask],
                                   c=cind_dens[key][mag][non_zero_mask],cmap='Spectral_r',s=40,vmin=vmin,vmax=0.6,alpha=0.4)
                        ax.set_xlim([xmin,xmax])
                        ax.set_ylim([ymin,ymax])
                        ax.set_yticks(np.arange(ymin, ymax+0.01, 4.5))
                        ax.annotate(r'$\alpha={} ,$'.format(np.round(alpha_dens[key][mag][0],2)),[0.04,8.9], size=20)
                        ax.annotate(r'$N={}$'.format(np.round(N,2)),[1.4,8.9], size=20)
                        ax.annotate(r'$\Delta m={} mag,$'.format(mag),[0.04,6.9], size=20)
                        ax.annotate(r'$CI = {} mag^2$'.format(np.round(np.median(err_ell_dens[key][mag])),2),[0.04,10.9], size=20)
                        ax.invert_yaxis()
                        ax.set_ylabel('')
                        ax.set_xlabel('')
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)

                    if (mode=='distance')and (outerind>=4) and (outerind not in [0,4,8,12,13,14,15]):
                        zero_mask = np.where(cind_dist[key][mag]==0)
                        non_zero_mask = np.where(c_dist[key][mag]!=0)
                        gplot=np.dstack([H_dens[key][mag],c_dens[key][mag]])
                        gm = GaussianMixture(n_components=2, random_state=0).fit(gplot[0])
    
                        N=gm.weights_[0]*100
                        ax.scatter(y=H_dist[key][mag][non_zero_mask],x=c_dist[key][mag][non_zero_mask],
                                   c=cind_dist[key][mag][non_zero_mask],cmap='Spectral_r',s=40,vmin=vmin,vmax=0.6,alpha=0.4)
                        ax.set_xlim([xmin,xmax])
                        ax.set_ylim([ymin,ymax])
                        ax.set_yticks(np.arange(ymin, ymax+0.01, 4.5))
                        ax.annotate(r'$\alpha={} ,$'.format(np.round(alpha_dist[key][mag][0],2)),[0.04,8.9], size=20)
                        ax.annotate(r'$N={}$'.format(np.round(N,2)),[1.4,8.9], size=20)
                        ax.annotate(r'$d_{new}=$'+'{} kpc,'.format(np.round(28*10**(mag/5),1)),[0.04,6.9], size=20)
                        ax.annotate(r'$CI = {} mag^2$'.format(np.round(np.median(err_ell_dist[key][mag])),2),[0.04,10.9], size=20)
                        ax.invert_yaxis()
                        ax.set_ylabel('')
                        ax.set_xlabel('')
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)

                    elif (mode=='density')and (outerind in [4,8]):
                        zero_mask = np.where(cind_dens[key][mag]==0)
                        non_zero_mask = np.where(c_dens[key][mag]!=0)
                        gplot=np.dstack([H_dens[key][mag],c_dens[key][mag]])
                        gm = GaussianMixture(n_components=2, random_state=0).fit(gplot[0])
    
                        N=gm.weights_[0]*100
                        ax.scatter(y=H_dens[key][mag][non_zero_mask],x=c_dens[key][mag][non_zero_mask],
                                   c=cind_dens[key][mag][non_zero_mask],cmap='Spectral_r',s=40,vmin=vmin,vmax=0.6,alpha=0.4)
                        ax.set_xlim([xmin,xmax])
                        ax.set_ylim([ymin,ymax])
                        ax.set_yticks(np.arange(ymin, ymax+0.01, 4.5))
                        ax.annotate(r'$\alpha={} ,$'.format(np.round(alpha_dens[key][mag][0],2)),[0.04,8.9], size=20)
                        ax.annotate(r'$N={}$'.format(np.round(N,2)),[1.4,8.9], size=20)
                        ax.annotate(r'$\Delta m={} mag,$'.format(mag),[0.04,6.9], size=20)
                        ax.annotate(r'$CI = {} mag^2$'.format(np.round(np.median(err_ell_dens[key][mag])),2),[0.04,10.9], size=20)
                        ax.invert_yaxis()
                        ax.yaxis.set_label_coords(-0.15,0)
                        ax.set_ylabel('Hg',size=22)       
                        ax.set_xlabel('')
                        ax.get_xaxis().set_visible(False)


                    if (mode=='distance') and (outerind in [4,8]):
                        zero_mask = np.where(cind_dist[key][mag]==0)
                        non_zero_mask = np.where(c_dist[key][mag]!=0)
                        gplot=np.dstack([H_dens[key][mag],c_dens[key][mag]])
                        gm = GaussianMixture(n_components=2, random_state=0).fit(gplot[0])
    
                        N=gm.weights_[0]*100
                        ax.scatter(y=H_dist[key][mag][non_zero_mask],x=c_dist[key][mag][non_zero_mask],
                                   c=cind_dist[key][mag][non_zero_mask],cmap='Spectral_r',s=40,vmin=vmin,vmax=0.6,alpha=0.4)
                        ax.set_xlim([xmin,xmax])
                        ax.set_ylim([ymin,ymax])
                        ax.set_yticks(np.arange(ymin, ymax+0.01, 4.5))
                        ax.annotate(r'$\alpha={} ,$'.format(np.round(alpha_dist[key][mag][0],2)),[0.04,8.9], size=20)
                        ax.annotate(r'$N={}$'.format(np.round(N,2)),[1.4,8.9], size=20)
                        ax.annotate(r'$d_{new}=$'+'{} kpc,'.format(np.round(28*10**(mag/5),1)),[0.04,6.9], size=20)
                        ax.annotate(r'$CI = {} mag^2$'.format(np.round(np.median(err_ell_dist[key][mag])),2),[0.04,10.9], size=20)
                        ax.invert_yaxis()
                        ax.set_xlabel('')
                        ax.get_xaxis().set_visible(False)


                    if (mode=='density') and (outerind  in [1,2,3]):
                        zero_mask = np.where(cind_dens[key][mag]==0)
                        non_zero_mask = np.where(c_dens[key][mag]!=0)
                        gplot=np.dstack([H_dens[key][mag],c_dens[key][mag]])
                        gm = GaussianMixture(n_components=2, random_state=0).fit(gplot[0])
    
                        N=gm.weights_[0]*100
                        ax.scatter(y=H_dens[key][mag][non_zero_mask],x=c_dens[key][mag][non_zero_mask],
                                   c=cind_dens[key][mag][non_zero_mask],cmap='Spectral_r',s=40,vmin=vmin,vmax=0.6,alpha=0.4)
                        ax.set_xlim([xmin,xmax])
                        ax.set_ylim([ymin,ymax])
                        ax.set_yticks(np.arange(ymin, ymax+0.01, 4.5))
                        ax.annotate(r'$\alpha={} ,$'.format(np.round(alpha_dens[key][mag][0],2)),[0.04,8.9], size=20)
                        ax.annotate(r'$N={}$'.format(np.round(N,2)),[1.4,8.9], size=20)
                        ax.annotate(r'$\Delta m={} mag,$'.format(mag),[0.04,6.9], size=20)
                        ax.annotate(r'$CI = {} mag^2$'.format(np.round(np.median(err_ell_dens[key][mag])),2),[0.04,10.9], size=20)
                        ax.invert_yaxis()
                        ax.set_ylabel('')
                        ax.set_xlabel('')
                        ax.set_title(f'{name}',size=22)
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)

                    if (mode=='distance') and (outerind  in [1,2,3]):
                        zero_mask = np.where(cind_dist[key][mag]==0)
                        non_zero_mask = np.where(c_dist[key][mag]!=0)
                        gplot=np.dstack([H_dens[key][mag],c_dens[key][mag]])
                        gm = GaussianMixture(n_components=2, random_state=0).fit(gplot[0])
    
                        N=gm.weights_[0]*100
                        ax.scatter(y=H_dist[key][mag][non_zero_mask],x=c_dist[key][mag][non_zero_mask],
                                   c=cind_dist[key][mag][non_zero_mask],cmap='Spectral_r',s=40,vmin=vmin,vmax=0.6,alpha=0.4)
                        ax.set_xlim([xmin,xmax])
                        ax.set_ylim([ymin,ymax])
                        ax.set_yticks(np.arange(ymin, ymax+0.01, 4.5))
                        ax.annotate(r'$\alpha={} ,$'.format(np.round(alpha_dist[key][mag][0],2)),[0.04,8.9], size=20)
                        ax.annotate(r'$N={}$'.format(np.round(N,2)),[1.4,8.9], size=20)
                        ax.annotate(r'$d_{new}=$'+'{} kpc,'.format(np.round(28*10**(mag/5),1)),[0.04,6.9], size=20)
                        ax.annotate(r'$CI = {} mag^2$'.format(np.round(np.median(err_ell_dist[key][mag])),2),[0.04,10.9], size=20)
                        ax.invert_yaxis()
                        ax.set_ylabel('')
                        ax.set_xlabel('')
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)


                    if (mode=='density') and (outerind==12):
                        zero_mask = np.where(cind_dens[key][mag]==0)
                        non_zero_mask = np.where(c_dens[key][mag]!=0)
                        gplot=np.dstack([H_dens[key][mag],c_dens[key][mag]])
                        gm = GaussianMixture(n_components=2, random_state=0).fit(gplot[0])
    
                        N=gm.weights_[0]*100
                        ax.scatter(y=H_dens[key][mag][non_zero_mask],x=c_dens[key][mag][non_zero_mask],
                                   c=cind_dens[key][mag][non_zero_mask],cmap='Spectral_r',s=40,vmin=vmin,vmax=0.6,alpha=0.4)
                        ax.set_xlim([xmin,xmax])
                        ax.set_ylim([ymin,ymax])
                        ax.set_yticks(np.arange(ymin, ymax+0.01, 4.5))
                        ax.annotate(r'$\alpha={} ,$'.format(np.round(alpha_dens[key][mag][0],2)),[0.04,8.9], size=20)
                        ax.annotate(r'$N={}$'.format(np.round(N,2)),[1.4,8.9], size=20)
                        ax.annotate(r'$\Delta m={} mag,$'.format(mag),[0.04,6.9], size=20)
                        ax.annotate(r'$CI = {} mag^2$'.format(np.round(np.median(err_ell_dens[key][mag])),2),[0.04,10.9], size=20)
                        ax.invert_yaxis()
                        ax.yaxis.set_label_coords(-0.15,0)
                        ax.set_ylabel('Hg',size=22)
                        ax.set_xlabel('')

                    if mode=='distance'and outerind==12:
                        zero_mask = np.where(cind_dist[key][mag]==0)
                        non_zero_mask = np.where(c_dist[key][mag]!=0)
                        gplot=np.dstack([H_dens[key][mag],c_dens[key][mag]])
                        gm = GaussianMixture(n_components=2, random_state=0).fit(gplot[0])
    
                        N=gm.weights_[0]*100
                        ax.scatter(y=H_dist[key][mag][non_zero_mask],x=c_dist[key][mag][non_zero_mask],
                                   c=cind_dist[key][mag][non_zero_mask],cmap='Spectral_r',s=40,vmin=vmin,vmax=0.6,alpha=0.4)
                        ax.set_xlim([xmin,xmax])
                        ax.set_ylim([ymin,ymax])
                        ax.set_yticks(np.arange(ymin, ymax+0.01, 4.5))
                        ax.annotate(r'$\alpha={} ,$'.format(np.round(alpha_dist[key][mag][0],2)),[0.05,8.9], size=20)
                        ax.annotate(r'$N={}$'.format(np.round(N,2)),[1.45,8.9], size=20)
                        ax.annotate(r'$d_{new}=$'+'{} kpc,'.format(np.round(28*10**(mag/5),1)),[0.05,6.9], size=20)
                        ax.annotate(r'$CI = {} mag^2$'.format(np.round(np.median(err_ell_dist[key][mag])),2),[0.05,10.9], size=20)
                        ax.invert_yaxis()
                        ax.set_ylabel('')
                        ax.set_xlabel('g-r',size=22)

                    if (mode=='density') and (outerind in [13,14,15]):
                        zero_mask = np.where(cind_dens[key][mag]==0)
                        non_zero_mask = np.where(c_dens[key][mag]!=0)
                        gplot=np.dstack([H_dens[key][mag],c_dens[key][mag]])
                        gm = GaussianMixture(n_components=2, random_state=0).fit(gplot[0])
    
                        N=gm.weights_[0]*100
                        ax.scatter(y=H_dens[key][mag][non_zero_mask],x=c_dens[key][mag][non_zero_mask],
                                   c=cind_dens[key][mag][non_zero_mask],cmap='Spectral_r',s=40,vmin=vmin,vmax=0.6,alpha=0.4)
                        ax.set_xlim([xmin,xmax])
                        ax.set_ylim([ymin,ymax])
                        ax.set_yticks(np.arange(ymin, ymax+0.01, 4.5))
                        ax.annotate(r'$\alpha={} ,$'.format(np.round(alpha_dens[key][mag][0],2)),[0.05,8.9], size=20)
                        ax.annotate(r'$N={}$'.format(np.round(N,2)),[1.45,8.9], size=20)
                        ax.annotate(r'$\Delta m={} mag,$'.format(mag),[0.05,6.9], size=20)
                        ax.annotate(r'$CI = {} mag^2$'.format(np.round(np.median(err_ell_dens[key][mag])),2),[0.05,10.9], size=20)
                        ax.invert_yaxis()
                        ax.set_ylabel('')
                        ax.set_xlabel('')
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)

                    if mode=='distance'and outerind in [13,14,15]:
                        zero_mask = np.where(cind_dist[key][mag]==0)
                        non_zero_mask = np.where(c_dist[key][mag]!=0)
                        gplot=np.dstack([H_dens[key][mag],c_dens[key][mag]])
                        gm = GaussianMixture(n_components=2, random_state=0).fit(gplot[0])
                        N=gm.weights_[0]*100
                        im = ax.scatter(y=H_dist[key][mag][non_zero_mask],x=c_dist[key][mag][non_zero_mask],
                                   c=cind_dist[key][mag][non_zero_mask],cmap='Spectral_r',s=40,vmin=vmin,vmax=0.6,alpha=0.4)
                        ax.set_xlim([xmin,xmax])
                        ax.set_ylim([ymin,ymax])
                        ax.set_yticks(np.arange(ymin, ymax, 4.5))
                        ax.annotate(r'$\alpha={} ,$'.format(np.round(alpha_dist[key][mag][0],2)),[0.05,8.9], size=20)
                        ax.annotate(r'$N={}$'.format(np.round(N,2)),[1.45,8], size=20)
                        ax.annotate(r'$d_{new}=$'+'{} kpc,'.format(np.round(28*10**(mag/5),1)),[0.05,6.9], size=20)
                        ax.annotate(r'$CI = {} mag^2$'.format(np.round(np.median(err_ell_dist[key][mag])),2),[0.05,10.9], size=20)
                        ax.invert_yaxis()
                        ax.set_ylabel('')
                        ax.set_xlabel('g-r',size=22)
                        ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)   
    cb_ax = fig.add_axes([1.02, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax, extend= 'max')
    #all_axes = fig.get_axes()
    #fig.tight_layout()
    #cax,kw = mpl.colorbar.make_axes([axs for axs in all_axes], aspect=40)
    #cbar= fig.colorbar(all_axes[0].get_children()[0], extend= 'max',cax=cax,**kw)
    cbar.ax.set_ylabel('Fraction of objects inside of the error ellipse', size=30)
    #fig.savefig("ConfunsionIndex_{}.png".format(tup[1]))
    return subfigs


def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level
def PMContourPlot(cut, contour_id=1):
    plt.rc('xtick',labelsize=15)
    plt.rc('ytick',labelsize=15)
    fig, ax= plt.subplots(1,1,figsize=(10,10))

    ax.set(adjustable='box', aspect=1)
    alpha= np.random.uniform(0,2*np.pi,np.size(cut['PM']))
    pm_alpha,pm_delta = cut['PM']*10**3*np.sin(alpha),cut['PM']*10**3*np.cos(alpha) # to convert pm to mas/yr
    pm_un_alpha, pm_un_delta = cut['PM_out']*10**3*np.sin(alpha),cut['PM_out']*10**3*np.cos(alpha)
    nbin = 37
    H, xedges, yedges = np.histogram2d(pm_alpha,pm_delta, bins=(nbin,nbin), normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbin))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbin,1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))

    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
    levels = [three_sigma,one_sigma]#,one_sigma]

    X, Y = np.meshgrid(0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1]))
    Z = pdf.T


    #### plot the contours

    contour = ax.contour(X, Y, Z, levels=levels,colors='black',linewidths=0,extent='both')
    p =contour.collections[0].get_paths()[contour_id].vertices
    p_ =contour.collections[1].get_paths()[0].vertices
    c3s = Polygon(p)

    points = np.column_stack((pm_un_alpha,pm_un_delta))
    id_p = np.zeros(np.shape(points)[0],dtype=bool)
    for k,pp in enumerate(points):
        pp = Point(pp)
        if not pp.within(c3s):
            id_p[k] = True
    ax.plot(p_[:,0],p_[:,1],'-',color='black',lw=2)
    ax.plot(p[:,0],p[:,1],'-',color='black',lw=2)
    ax.scatter(pm_un_alpha,pm_un_delta,marker='h',s=10, color='gray',alpha=0.4)
    ax.scatter(pm_un_alpha[id_p],pm_un_delta[id_p],marker='h',s=10, color='darkorange',alpha =0.7)
    ax.text(-.7,0.2,r'$3\sigma$',  weight='bold', fontsize=14,color='darkred')
    ax.text(-.07,0.03,r'$1\sigma$', weight='bold',fontsize=14,color='darkred')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_xlabel(r'$\mu_{\alpha}$ '+r'$[mas/yr]$',fontsize=16)
    ax.set_ylabel(r'$\mu_{\delta}$ '+r'$[mas/yr]$',fontsize=16)

    return plt.tight_layout()
    
