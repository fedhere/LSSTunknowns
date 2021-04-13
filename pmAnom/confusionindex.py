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



class reducedPM(BaseMetric): 
        def sigma_slope_arr(x, sigmay):
            w = 1./np.array(sigmay)**2
            denom = np.sum(x)*np.sum(w*x**2,axis=1)-np.sum(w*x,axis=1)**2
            denom[np.where(denom <= 0)]=np.nan
            denom[np.where(denom > 0)] = np.sqrt(np.sum(w)*denom[np.where(denom > 0)]**-1 )
            return denom
    
        def readfile(self, filename='', colsname=['']): 
            if 'csv' in filename: 
                #print('reading {}'.format(filename)) 
                data = pd.read_csv(filename, header=0, names = colsname ) 
            elif 'fits' in filename: 
                 #print('reading {}'.format(filename)) 
                hdul = fits.open(filename) 
                data = hdul[1].data 
            elif ['txt', 'dat'] in filename: 
                #print('reading {}'.format(filename)) 
                data = {k:[] for k in colsname} 
                f = open(filename) 
                righe = f.readlines() 
                for line in righe: 
                    line.split() 
                    for i, k in enumerate(colsname): 
                        data[k].append(float(line[i]))                 
            elif 'json'in filename: 
                print('not implemented to read .json extention') 
            return data 
        
        def __init__(self, filename = 'data.csv', snr_lim=5,mode=None, MagIterLim=[0,1,1], surveyduration=10,
                      metricName='reducedPM',m5Col='fiveSigmaDepth',real_data= True, out_type = 'confusion',
                      mjdCol='observationStartMJD',filterCol='filter', seeingCol='seeingFwhmGeom',dataout=True,**kwargs): 
            self.mjdCol = mjdCol 
            self.m5Col = m5Col 
            self.seeingCol = seeingCol 
            self.filterCol = filterCol 
            self.snr_lim = snr_lim 
            self.filename = filename  
            self.dataout = dataout 
            self.mode = mode 
            self.MagIterLim = MagIterLim 
            self.surveyduration = surveyduration 
            self.out_type = out_type
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
            if self.real_data:    
                colsname=['RA', 'DEC','g','g-r','Hg','PM_OUT','deltaX'] 
                self.data = self.readfile(self.filename, colsname)
                self.data['MAG']=self.data['g']
            else:
                colsname=['MAG','MODE','d','PM','PM_OUT']
                self.data = self.readfile(self.filename, colsname)
            data_sag93 = fits.open('sa93.all.fits')
            table = data_sag93[1].data    
            mu_sag= np.transpose([table['MUX'],table['MUY']])
            self.mu_sag=mu_sag
        def run(self, dataSlice, slicePoint=None): 
            pm = np.array(self.data['PM_OUT'])
            mag = np.array(self.data['MAG'])
            obs = np.where(dataSlice[self.mjdCol]<min(dataSlice[self.mjdCol])+365*self.surveyduration)
            
            deltamag= np.arange(self.MagIterLim[0],self.MagIterLim[1],self.MagIterLim[2])
            out = {}
            for dm in deltamag: 
                
                    if self.mode == 'distance': 
                        pmnew= pm/(10**(dm/5)) 
                    elif self.mode == 'density': 
                        pmnew= pm 
                    else: 
                        print('##### ERROR: the metric is not implemented for this mode.')
                        
                    mjd = dataSlice[self.mjdCol][obs]
                    flt = dataSlice[self.filterCol][obs]
                    if ('g' in flt) and ('r' in flt):
                        
                        # select objects above the limit magnitude threshold 
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
                        DeltaTs.sort()
                        DeltaTs = np.array(DeltaTs)
                        if np.size(DeltaTs)>0:
                            dt_pm = 0.05*np.median(dataSlice[self.seeingCol])/pmnew[np.unique(row)]
                            selection = np.where((dt_pm>DeltaTs[0]) & (dt_pm<DeltaTs[-1]))

                            if self.real_data:
                                Hg = np.array(self.data['Hg'])[selection]
                                gr = np.array(self.data['g-r'])[selection] 
                            else:
                                colsname=['RA', 'DEC','MAG','g-r','Hg','PM_OUT','deltaX'] 
                                color_dist = self.readfile('data.csv', colsname)
                                gr = np.random.choice(color_dist['g-r'],size= np.size(selection))
                                Hg = mag[selection]+5*np.log10(pm[selection])-10

                            g= mag+dm 
                            sigmaHg = np.sqrt((g[selection]/m52snr(g[selection],np.median(dataSlice[self.m5Col])))**(2)+ (4.715*sigmapm[selection]/pmnew[selection])**2) 
                            sigmag = np.sqrt((g[selection]/m52snr(g[selection],np.median(dataSlice[self.m5Col])))**2+((g[selection]-gr)/m52snr((g[selection]-gr),np.median(dataSlice[self.m5Col])))**2)
                            err_ellipse = np.pi*sigmaHg*sigmag
                            n = len(pmnew[selection])/len(pmnew)
                            pa= np.random.uniform(0,2*np.pi,len(pmnew[selection]))
                            n_components = np.arange(1, 10)
                            pm_alpha = pmnew[selection]*np.sin(pa)
                            pm_delta = pmnew[selection]*np.cos(pa)
                            mu = np.transpose(np.array([pm_alpha,pm_delta]))
                            Phi_c = np.exp(((mu[:,0]-0.42)**2+(mu[:,1]+2.66)**2)/2)
                            gamma=((0.42-np.median(self.mu_sag[:,0]))*(-2.66-np.median(self.mu_sag[:,0])))
                            Phi_f=np.exp(((mu[:,0]-np.median(self.mu_sag[:,0]))**2-(mu[:,0]-np.median(self.mu_sag[:,0]))*(mu[:,1]-np.median(self.mu_sag[:,1]))+(mu[:,1]-np.median(self.mu_sag[:,1]))**2)/2/(1-gamma))
                            P=Phi_c/Phi_f
                            idx_c =np.where(P>1)

                            try:
                                models = [GMM.GaussianMixture(n, covariance_type='full', random_state=0).fit(mu[idx_c]) for n in n_components]
                                BIC = [m.bic(mu[idx_c]) for m in models]
                                k_cluster=n_components[np.where(BIC==np.min(BIC))][0]    
                            except:
                                k_cluster = 0
                            out[dm] = {'CI':np.median(err_ellipse), 'k': k_cluster, 'PM':mu}  
                        else:
                            out[dm] = {'CI':0, 'k': 0, 'PM':pm} 
            if self.dataout: 
                return out  
            else:
                if ('g' in flt) and ('r' in flt):
                    if self.out_type =='cluster':
                        res = out[dm]['k']
                    elif self.out_type == 'confusion':
                        res = out[dm]['CI']*np.size(selection)/np.size(pm)
                    return res