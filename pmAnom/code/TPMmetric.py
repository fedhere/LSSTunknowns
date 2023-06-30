import numpy as np
import pandas as pd
import pickle
from scipy.stats import truncnorm
from itertools import combinations
import healpy as hp
### LSST dependencies
from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.utils.mafUtils import radec2pix
from rubin_sim.maf.utils import m52snr, astrom_precision, sigma_slope

def RADec2pix(nside, ra, dec, degree=True):
    """
    Calculate the nearest healpixel ID of an RA/Dec array, assuming nside.

    Parameters
    ----------
    nside : int
        The nside value of the healpix grid.
    ra : numpy.ndarray
        The RA values to be converted to healpix ids, in degree by default.
    dec : numpy.ndarray
        The Dec values to be converted to healpix ids, in degree by default.

    Returns
    -------
    numpy.ndarray
        The healpix ids.
    """
    if degree:
        ra = np.radians(ra) # change to radians
        dec = np.radians(dec)
    
    lat = np.pi/2. - dec
    hpid = hp.ang2pix(nside, lat, ra )
    return hpid

class TransienPM(BaseMetric): 
     #    Generate a population of transient objects and see what is its proper motion  , 
    def __init__(self, metricName='TransienPM', f='g', snr_lim=5,m5Col='fiveSigmaDepth', nside= 32, populationfile = 'gaia2pix.p', mjdCol='observationStartMJD',filterCol='filter',seeingCol='seeingFwhmGeom', surveyduration=10, **kwargs): 
        self.populationfile = populationfile
        with open(self.populationfile, 'rb') as data:
            self.population = pickle.load(data)
        self.mjdCol = mjdCol 
        self.seeingCol= seeingCol 
        self.m5Col = m5Col 
        self.filterCol = filterCol 
        self.snr_lim = snr_lim 
        self.f = f
        self.surveyduration = surveyduration  
        self.nside = nside
        super(TransienPM, self).__init__(col=[self.mjdCol, self.filterCol, self.m5Col, self.seeingCol, 'night'],units='fraction of transients recovered', metricName=metricName,**kwargs) 
        
        with open(self.populationfile, 'rb') as data:
            self.population = pickle.load(data)
            self.population_pix = RADec2pix(self.nside,self.population['ra'],self.population['dec'])
        
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
            
            obs = np.where((dataSlice['filter'] == self.f) & (
            dataSlice[self.mjdCol] < min(dataSlice[self.mjdCol]) + 365 * self.surveyduration))  
            if np.size(obs)>2:
                fieldRA, fieldDec = dataSlice['fieldRA'], dataSlice['fieldDec'] 
                footprint_pix = RADec2pix(self.nside,fieldRA, fieldDec)
                pid = np.where( self.population_pix == np.unique(footprint_pix)[0])

                mjd = dataSlice[self.mjdCol][obs]
                start_time = time.time()

                mags = np.array(self.population['mag'])[pid]        

                mu_ra, mu_dec = np.array(self.population['pm_ra_cosdec'])[pid], np.array(self.population['pm_dec'])[pid]
                mu = np.sqrt(mu_ra**2+mu_dec**2)

                mu_ra_un, mu_dec_un= np.array(self.population['pm_un_ra_cosdec'])[pid], np.array(self.population['pm_un_dec'])[pid]
                mu_unusual = np.sqrt(mu_ra_un**2+ mu_dec_un**2)
                time1 = time.time() 
                print('simulation of population takes {} min'.format((time1-start_time)/60))
                print('### upload population for los in field ({},{})'. format(np.round(fieldRA,2),np.round(fieldDec,2)))
                # select objects above the limit magnitude threshold whatever the magnitude of the star is
                snr = m52snr(np.array(mags)[:, np.newaxis], dataSlice[self.m5Col][obs])  
                #select the snr above the threshold
                row, col = np.where(snr > self.snr_lim) 
                #estimate the uncertainties on the position
                precis = astrom_precision(dataSlice[self.seeingCol][obs], snr[row, :])  
                #estimate the uncertainties on the proper motion
                sigmapm = sigma_slope(dataSlice[self.mjdCol][obs], precis) * 365.25 * 1e3   

                Times = np.sort(mjd)
                dt = np.array(list(combinations(Times,2)))
                if np.size(dt)>0:
                    DeltaTs = np.absolute(np.subtract(dt[:,0],dt[:,1]))            
                    DeltaTs = np.unique(DeltaTs)

                    dt_pm = 0.05*np.amin(dataSlice[self.seeingCol])/np.absolute(mu)
                    selection = np.where((dt_pm > min(DeltaTs)) & (dt_pm < max(DeltaTs)) & (np.absolute(mu) > sigmapm)) 
                    objRate = 0.7 # how many go off per day
                    nObj=np.size(mu[selection])
                    m0s = np.array(mags)[selection]
                    t = dataSlice[self.mjdCol][obs] - dataSlice[self.mjdCol].min() 
                    detected = 0 
             # Loop though each generated transient and decide if it was detected , 
             # This could be a more complicated piece of code, for example demanding  , 
             # A color measurement in a night. , 
                    durations = dt_pm[selection]
                    slopes = (np.random.rand(nObj) * 2.5 + 0.5 )*(np.random.binomial(1, 0.5, nObj) * 2 - 1)
                    t0s = np.random.uniform(0,np.amin(dataSlice[self.mjdCol])+365*self.surveyduration,nObj)
                    lcs = self.lightCurve(t, t0s, m0s,durations, slopes) 
                    good = m52snr(lcs,dataSlice[self.m5Col][obs])> self.snr_lim

                    detectedTest = good.sum(axis=1)
                    detected = np.sum(detectedTest>2)

                    #for i,t0 in enumerate(np.random.uniform(0,self.surveyduration,nObj)): 
                    #    duration =dt_pm[selection][i]
                    #    slope = np.random.uniform(-3,3) 
                    #    lc = self.lightCurve(t, t0, m0s[i],duration, slope) 
                    #    good = m52snr(lc,dataSlice[self.m5Col][obs])> self.snr_lim 
                    #    detectTest = dataSlice[self.m5Col][obs] - lc 
                    #    if detectTest.max() > 0 and len(good)>2: 
                    #         detected += 1 
                    # Return the fraction of transients detected , 
                    if float(nObj) == 0:
                        A = np.inf 
                    else: 
                        A=float(nObj) 
                        res = float(detected)/A            
                        #print('detected fraction:{}'.format(res)) 
                        return res
