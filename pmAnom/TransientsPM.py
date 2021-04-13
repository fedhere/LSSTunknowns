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





class TransienPM(BaseMetric): 
    def sigma_slope_arr(x, sigmay):
            w = 1./np.array(sigmay)**2
            denom = np.sum(x)*np.sum(w*x**2,axis=1)-np.sum(w*x,axis=1)**2
            denom[np.where(denom <= 0)]=np.nan
            denom[np.where(denom > 0)] = np.sqrt(np.sum(w)*denom[np.where(denom > 0)]**-1 )
            return denom
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
            sim = pd.read_csv('/home/idies/workspace/Storage/fragosta/persistent/LSST_OpSim/Scripts_NBs/True Novelities/simulation_pm.csv', usecols=['MAG','MODE','d','PM','PM_out'])
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
             lightcurve = np.zeros(np.size(t), dtype=float) + 99. 
     # Select only times in the lightcurve duration window , 
             good = np.where( (t >= t0) & (t <= t0+duration) ) 
             lightcurve[good] = peak + slope*(t[good]-t0) 
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
                precis = astrom_precision(dataSlice[self.seeingCol][obs], snr)
                sigmapm=sigma_slope_arr(dataSlice[self.mjdCol][obs], precis)*365.25*1e3

                #select the objects which displacement can be detected
                Times = list(mjd)
                DeltaTs = []
                while np.size(Times)>1:
                    for d in range(len(Times)-1):
                        DeltaTs.append(Times[d]-Times[d+1])
                    Times.remove(Times[0])
                dt_pm = 0.05*np.median(dataSlice[self.seeingCol])/pm[np.unique(row)]
                DeltaTs.sort()
                DeltaTs = np.array(DeltaTs)
                if np.size(DeltaTs)>0:
                    selection = np.where((dt_pm>DeltaTs[0]) & (dt_pm<DeltaTs[-1]))

                    objRate = 0.7 # how many go off per day
                    nObj=np.size(pm[selection])
                    m0s = mag[selection]
                    t = dataSlice[self.mjdCol][obs] - dataSlice[self.mjdCol].min() 
                    detected = 0 
             # Loop though each generated transient and decide if it was detected , 
             # This could be a more complicated piece of code, for example demanding  , 
             # A color measurement in a night. , 

                    for i,t0 in enumerate(np.random.uniform(0,self.surveyduration,nObj)): 
                        duration =dt_pm[selection][i]
                        slope = np.random.uniform(-3,3) 
                        lc = self.lightCurve(t, t0, m0s[i],duration, slope) 
                        good = m52snr(lc,dataSlice[self.m5Col][obs])> self.snr_lim 
                        detectTest = dataSlice[self.m5Col][obs] - lc 
                        if detectTest.max() > 0 and len(good)>2: 
                             detected += 1 
                     # Return the fraction of transients detected , 
                    if float(nObj) == 0:
                        A = np.inf 
                    else: 
                        A=float(nObj) 
                        res = float(np.sum(detected))/A            
                        #print('detected fraction:{}'.format(res)) 
                        return res