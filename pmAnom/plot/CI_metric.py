import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, time, glob,warnings, glob
from scipy.stats import *
from itertools import combinations
from astropy.io import fits
import pickle
from util import *
import healpy as hp

#lsst modules
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.db as db
import lsst.sims.maf.plots as plots
import lsst.sims.maf.slicers as slicers
from lsst.sims.maf.metrics import BaseMetric
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.utils.mafUtils import radec2pix
from lsst.sims.maf.utils import m52snr, astrom_precision, sigma_slope
import lsst.sims.maf.maps as maps



class reducedPM(BaseMetric):         
        def __init__(self, fits_filename = 'sa93.all.fits', snr_lim=5,mode=None, MagIterLim=[0,1,1], surveyduration=10,
                      metricName='reducedPM',m5Col='fiveSigmaDepth', atm_err =0.01,
                      mjdCol='observationStartMJD',filterCol='filter', seeingCol='seeingFwhmGeom',dataout=True,**kwargs): 
            #opsim cols name
	    self.mjdCol = mjdCol  # default ='observationStartMJD' The column name for the exposure time. Default observationStartMJD.
            self.m5Col = m5Col    # default ='fiveSigmaDepth'  The default column name for m5 information in the input data. Default fiveSigmaDepth.
            self.seeingCol = seeingCol  # default ='seeingFwhmGeom'  The column name for the seeing information. Since the astrometry errors are based on the physical size of the PSF, this should be the FWHM of the physical psf.
            self.filterCol = filterCol  # default ='filter' The column name for the filter information. Default filter.

            # survey params
            self.surveyduration = surveyduration # default= 10 
            self.atm_err = atm_err  # default =0.01
            self.snr_lim = snr_lim  # default = 5  The expected centroiding error due to the atmosphere, in arcseconds

	    # metric params            
            self.fits_filename = fits_filename  # default= 'sa93.all.fits'  The name of the fits file that contain the data of the stars in the galaxy or the stream
            self.dataout = dataout              # default= True  Boolean value to retrieve the simulated data as output
            self.mode = mode                    # default= None This can be 'distance' or 'density' rather the simulation is moving the structure closer/farther or it is considering the structure in a less/more stellar dense environment
            self.MagIterLim = MagIterLim       # default= [0,1,1] The range of magnitude considered for the delta m
            

             # to have as output all the simulated observed data set dataout=True, otherwise the relative error for  
             # each helpix is estimated 
            if self.dataout: 
                super(reducedPM, self).__init__(col=[self.mjdCol,self.filterCol, self.m5Col,self.seeingCol, 'night' ],metricDtype='object', units='', metricName=metricName, 
                                                      **kwargs) 
            else: 
                super(reducedPM, self).__init__(col=[self.mjdCol,self.filterCol,  self.m5Col,self.seeingCol, 'night' ], 
                                                            units='', metricDtype='float',metricName=metricName, 
                                                             **kwargs) 
            
            
	    data_sag93 = fits.open(self.fits_filename)   #data from sagitariusA are extracted
            table = data_sag93[1].data    
            mu_sag= np.transpose(np.sqrt(table['MUX']**2+table['MUY']**2)) #the proper motion components are summed in quadrature to retrive the total proper motion of each star in the dataset
            M_sag = np.transpose(table['GMAG'])
            self.mu_sag = mu_sag  # The proper motion variable is set to the proper motion from the data
            self.mag_sag = M_sag  # The magnitude variable is set to the g magnitude from the data
            self.gr = np.transpose(table['GMAG']-table['RMAG'])  # The color variable is set to the g-r color from the data
                  
        def run(self, dataSlice, slicePoint=None): 
            obs = np.where(dataSlice[self.mjdCol]<min(dataSlice[self.mjdCol])+365*self.surveyduration)  #filter to select only the index related to the mid within the survey duration window
            
            deltamag= np.arange(self.MagIterLim[0],self.MagIterLim[1],self.MagIterLim[2]) # The delta magnitude range variable is initialized 
            out = {}   # The output variable is initialized
            for dm in deltamag: 
                    mjd = dataSlice[self.mjdCol][obs]    
                    flt = dataSlice[self.filterCol][obs]
                    if ('g' in flt) and ('r' in flt):                        
                        # select objects above the limit magnitude threshold 
                        snr = m52snr(mag[:, np.newaxis],dataSlice[self.m5Col][obs])  # signal to noise ratio matrix, rows are the signal to noise ratio of all the stars in the dataset for a given magnitude limit, columns are all the magnitude limits for the given star in the dataset
                        selection_mag =np.where(np.mean(snr,axis=0)>self.snr_lim) # marginalized matrix with respect the magnitudes
                        Times = np.sort(mjd)
                        dt = np.array(list(combinations(Times,2))) # The list of all the possible couple of mjd of the observations
                        DeltaTs = np.absolute(np.subtract(dt[:,0],dt[:,1]))    # The list of all the possible time gaps
                        DeltaTs = np.unique(DeltaTs)
                        displasement_error = astrom_precision(dataSlice[self.seeingCol][obs][selection_mag], np.mean(snr,axis=0)[selection_mag])  # uncertainties on the deltaX                         displasement_error = np.sqrt(displasement_error**2 + self.atm_err**2)
                        sigmapm = sigma_slope(dataSlice[self.mjdCol][obs][selection_mag], displasement_error)  # The uncertainty on proper motion measurements
                        sigmapm *= 365.25*1e3   # convert the unit to ''/yr
                        
                        #filter to select the proper motion measurements within the possible time gap of the stars above the snr threshold
                        if np.size(DeltaTs)>0:
                                     dt_pm = 0.05*np.amin(dataSlice[self.seeingCol][obs])/pmnew
                                     selection_mag_pm = np.where((dt_pm>min(DeltaTs)) & (dt_pm<max(DeltaTs)) & (np.absolute(pmnew) >sigmapm) & (np.mean(snr,axis=1)>self.snr_lim))
                        
                        Hg = mag[selection_mag_pm]+5*np.log10(pmnew[selection_mag_pm])-10   # reduced proper motion
                        sigmaHg = np.sqrt((mag[selection_mag_pm]/m52snr(mag[selection_mag_pm],np.median(dataSlice[self.m5Col])))**(2)+ (4.715*sigmapm/pmnew[selection_mag_pm])**2)    # uncertainties on reduced proper motion
                        sigmag = np.sqrt((mag[selection_mag_pm]/m52snr(mag[selection_mag_pm],np.median(dataSlice[self.m5Col])))**2+((mag[selection_mag_pm]-self.gr[selection_mag_pm])/m52snr((mag[selection_mag_pm]-self.gr[selection_mag_pm]),np.median(dataSlice[self.m5Col])))**2)    # uncertainties on proper motion
                        err_ellipse = np.pi*sigmaHg*sigmag   #error ellipse area of each star in the RPMD

                        if self.dataout:
                            CI = np.array([np.nansum((([self.gr[selection_mag_pm]-gcol ])/sigmag)**2 + ((Hg-h)/sigmaHg)**2 <= 1)/np.size(pmnew[selection_mag_pm]) for (gcol,h) in zip(self.gr[selection_mag_pm],Hg)])                      # Confusion Index = for each star, the number of points of the RPMD within its error ellipse

                            out[dm] = {'CI':CI,'alpha':np.size(pmnew[selection_mag_pm])/np.size(pmnew) ,'err':err_ellipse,'Hg':Hg,'gr':self.gr[selection_mag_pm], 'sigmaHg':sigmaHg,'sigmagr':sigmag}
                        else:
                            out[dm] = {'alpha':np.size(pmnew[selection_mag_pm])/np.size(pmnew) ,'err':err_ellipse}
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
