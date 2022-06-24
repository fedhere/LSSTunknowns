import sys,os, glob, time, astropy, warnings, pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
import sklearn.mixture.gaussian_mixture as GMM
from builtins import zip
from itertools import combinations
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
