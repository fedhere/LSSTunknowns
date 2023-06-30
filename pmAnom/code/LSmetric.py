import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools, pickle
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.coordinates import (CartesianRepresentation,
                                 CartesianDifferential, Galactic)
from astropy.io import ascii
import healpy as hp
from scipy.stats import truncnorm
from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.utils.mafUtils import radec2pix
from rubin_sim.maf.utils import m52snr, astrom_precision, sigma_slope
from galpy.potential import NFWPotential, MiyamotoNagaiPotential, PowerSphericalPotentialwCutoff
from galpy.df import isotropicNFWdf, kingdf, dehnendf
import time


# The LSPMmetric class is a custom metric designed to evaluate the #efficiency of the Large Synoptic Survey Telescope (LSST) in detecting high proper #motion stars. The __init__ method initializes the instance variables of the class #with the input values provided to the method.
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

class LSPMmetric(BaseMetric):
    def __init__(self,metricName='LSPMmetric', populationfile = 'gaia2pix.p', f='g', nside= 32, surveyduration=10, snr_lim=5., sigma_threshold=1, m5Col='fiveSigmaDepth', mjdCol='observationStartMJD', filterCol='filter', seeingCol='seeingFwhmGeom', dataout=False,dynplot=False,**kwargs):
    
        self.populationfile = populationfile
        self.nside = nside
        # opsim
        self.mjdCol = mjdCol  # Column name for modified julian date of the observation
        self.m5Col = m5Col    # Column name for the five sigma limit
        self.seeingCol = seeingCol  # Column name for the geometrical seeing
        self.filterCol = filterCol  # Column name for filters
        # selection criteria
        self.surveyduration = surveyduration # integer, number of years from the start of the survey
        self.snr_lim = snr_lim               # float, threshold for the signal to noise ratio (snr), all the signals with a snr>snr_lim are detected
        self.f = f                          # string, filter used for the observations
        self.sigma_threshold = sigma_threshold  # integer,
        # output
        self.dataout = dataout  # to have as output all the simulated observed data set dataout=True, otherwise the relative error for
        self.dynplot = dynplot
        if self.dataout:
            super(LSPMmetric, self).__init__(col=[self.mjdCol, self.filterCol, self.m5Col, self.seeingCol, 'night'],
                                             metricDtype='object', units='', metricName=metricName,
                                             **kwargs)
        else:
            super(LSPMmetric, self).__init__(col=[self.mjdCol, self.filterCol, self.m5Col, self.seeingCol, 'night'],
                                             units='Proper Motion relative error', metricName=metricName,
                                             **kwargs)
        
        #start = time.time()
        with open(self.populationfile, 'rb') as data:
            self.population = pickle.load(data)
            self.population_pix = RADec2pix(self.nside,self.population['ra'],self.population['dec'])
    

    
    def run(self, dataSlice, slicePoint=None):
        np.random.seed(2500)
        ''' simulation of the measured proper motion '''
        #select the observations in the reference filter within the survey duration
        
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
            time2  = time.time()
            print('measure sigmapm takes {} min'.format((time2-time1)/60))
            print('### measure sigmapm')
            Times = np.sort(mjd)
            dt = np.array(list(itertools.combinations(Times, 2)))
            #estimate all the possible time gaps given the dates of the observations
            DeltaTs = np.absolute(np.subtract(dt[:, 0], dt[:, 1])) 
            DeltaTs = np.unique(DeltaTs)
            # time gap of the motion given the proper motion
            dt_pm = 0.05 * np.amin(dataSlice[self.seeingCol][obs]) / np.absolute(mu) 
            dt_pm_unusual = 0.05 * np.amin(dataSlice[self.seeingCol][obs]) / np.absolute(mu_unusual)
            #select measurable proper motions
            selection_usual = np.where((dt_pm > min(DeltaTs)) & (dt_pm < max(DeltaTs)) 
                                       & (np.absolute(mu) > sigmapm)) 
            selection_unusual = np.where((dt_pm_unusual > min(DeltaTs)) & (dt_pm_unusual < max(DeltaTs)) & (np.absolute(mu_unusual) > sigmapm)) 
            time3  = time.time()
            print('selections take {} min'.format((time3-time2)/60))
            print('### selections of observable subpopulation')
            #select measurable proper motions
            if np.size(selection_usual)>0:# and 
                if np.size(selection_unusual) > 0:
                    pm_alpha, pm_delta = mu_ra[selection_unusual], mu_dec[selection_unusual]
                    pm_un_alpha, pm_un_delta = mu_ra_un[selection_unusual], mu_dec_un[selection_unusual]
                    mu = mu[selection_usual]
                    mu_unusual = mu_unusual[selection_unusual]
                    
                    variance_mu = np.std(mu)
                    
                    #select the proper motion measurement outside the n*sigma limit
                    unusual = np.where(np.abs(mu_unusual - np.mean(mu_unusual))> self.sigma_threshold * variance_mu ) 
                    
                    #estimate the fraction of unusual proper motion that we can identify as unusual
                    res = np.size(unusual) / np.size(selection_unusual) 
                    time4  = time.time()-time3
                    print('measure likelihood score takes {} min'.format(time4/60))
                    print('### estimation of likelihood score')
                    if self.dataout:
                        dic = {'detected': res,
                               'pixID': np.unique(footprint_pix)[0],
                               'PM': pd.DataFrame({'pm_alpha': pm_alpha, 'pm_delta': pm_delta}),
                               'PM_un': pd.DataFrame({'pm_alpha': pm_un_alpha, 'pm_delta': pm_un_delta})}
                        return dic
                    else:
                        res = res
                    
                else:
                    #print('no measurable pm unusual in this location')
                    res=0
            else:
                #print('no measurable pm usual in this location')
                res=1
        else:
            #print('no observations in filter {} in this location'.format(self.f))
            res=0
        print('### fraction unknown = {}'.format( res))
        
        return res
