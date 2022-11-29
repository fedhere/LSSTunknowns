import numpy as np
import pandas as pd
### LSST dependencies
from lsst.sims.maf.metrics import BaseMetric
from lsst.sims.maf.utils import m52snr, astrom_precision, sigma_slope
from opsimUtils import *


class reducedPM(BaseMetric):         
        def __init__(self, fits_filename = 'sa93.all.fits', snr_lim=5,mode=None, MagIterLim=[0,1,1], surveyduration=10,
                      metricName='reducedPM',m5Col='fiveSigmaDepth',gap_selection=False, atm_err =0.01,
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
            self.atm_err = atm_err
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
                  
        def run(self, dataSlice, slicePoint=None): 
            #pm = np.array(self.data['PM_OUT'])
            #mag = np.array(self.data['MAG'])
            obs = np.where(dataSlice[self.mjdCol]<min(dataSlice[self.mjdCol])+365*self.surveyduration)
            
            deltamag= np.arange(self.MagIterLim[0],self.MagIterLim[1],self.MagIterLim[2])
            out = {}
            for dm in deltamag: 
                    #print(f'mode={self.mode}')
                    if self.mode == 'distance': 
                        pmnew= self.mu_sag /(10**(dm/5)) 
                        mag = self.mag_sag +dm 
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
                        selection_mag =np.where(np.mean(snr,axis=0)>self.snr_lim)
                        Times = np.sort(mjd)
                        dt = np.array(list(combinations(Times,2)))
                        DeltaTs = np.absolute(np.subtract(dt[:,0],dt[:,1]))            
                        DeltaTs = np.unique(DeltaTs)
                        displasement_error = astrom_precision(dataSlice[self.seeingCol][obs][selection_mag], np.mean(snr,axis=0)[selection_mag])
                        displasement_error = np.sqrt(displasement_error**2 + self.atm_err**2)
                        sigmapm = sigma_slope(dataSlice[self.mjdCol][obs][selection_mag], displasement_error)
                        sigmapm *= 365.25*1e3
                        #print(f'sigmapm={sigmapm}')
                        #print(f'size pm ={np.size(pmnew)}')
                        #print(f'size selection_mag ={np.size(selection_mag)}')
                        if np.size(DeltaTs)>0:
                                     dt_pm = 0.05*np.amin(dataSlice[self.seeingCol][obs])/pmnew
                                     selection_mag_pm = np.where((dt_pm>min(DeltaTs)) & (dt_pm<max(DeltaTs)) & (np.absolute(pmnew) >sigmapm) & (np.mean(snr,axis=1)>self.snr_lim))
                        
                        #print(f'size selection_mag_pm ={np.size(selection_mag_pm)}')
                        Hg = mag[selection_mag_pm]+5*np.log10(pmnew[selection_mag_pm])-10
                        sigmaHg = np.sqrt((mag[selection_mag_pm]/m52snr(mag[selection_mag_pm],np.median(dataSlice[self.m5Col])))**(2)+ (4.715*sigmapm/pmnew[selection_mag_pm])**2) 
                        sigmag = np.sqrt((mag[selection_mag_pm]/m52snr(mag[selection_mag_pm],np.median(dataSlice[self.m5Col])))**2+((mag[selection_mag_pm]-self.gr[selection_mag_pm])/m52snr((mag[selection_mag_pm]-self.gr[selection_mag_pm]),np.median(dataSlice[self.m5Col])))**2)
                        err_ellipse = np.pi*sigmaHg*sigmag
                        if self.dataout:
                            CI = np.array([np.nansum((([self.gr[selection_mag_pm]-gcol ])/sigmag)**2 + ((Hg-h)/sigmaHg)**2 <= 1)/np.size(pmnew[selection_mag_pm]) for (gcol,h) in zip(self.gr[selection_mag_pm],Hg)])                      

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
