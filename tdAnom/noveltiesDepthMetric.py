import numpy as np
import pandas as pd

import lsst.sims.maf.metrics as metrics

# the depth goal 
mag_stretch = {'u':24.0, 'g':25.1, 'r':24.8, 'i':24.1, 'z':23.4, 'y':22.2 }

class noveltiesDepthMetric(metrics.BaseMetric):
    """
    depth 
    
    returns
    fiveSigmaDepth - stretch_goal
    
    Parameters:
        colname: 
        fltpair: filter pair, eg ['r', 'i']
        snr_lim: list, signal to noise ratio (fiveSigmaDepth) threshold for fltpair, default [5, 5]
        filename: output a csv table for time gaps of each field
    
    """

    def __init__(self, colname=['observationStartMJD', 'filter', 'fiveSigmaDepth'], 
                 fltpair=['r', 'i'], mag_lim=[18, 18], 
                 dataout=False, **kwargs):
        self.colname = colname
        self.fltpair = fltpair
        self.mag_lim = mag_lim
        self.dataout = dataout
        
        self.Nrun = 0   # record the how many time run run()
       
        if self.dataout:
            super().__init__(col=self.colname, metricDtype='object', **kwargs)
        else:
            super().__init__(col=self.colname, metricDtype='float', **kwargs)

    def run(self, dataSlice, slicePoint=None):
        
        # return all possible time gaps for each fields
        
        f0 = self.fltpair[0]
        f1 = self.fltpair[1]
        
        #check input config
        #print(f0, f1, self.tmin, self.tmax, self.mag_lim)
            
        # sort dataSlice
        
        idx0 = ( dataSlice['filter'] == f0 ) & ( dataSlice['fiveSigmaDepth'] > self.mag_lim[0])
        idx1 = ( dataSlice['filter'] == f1 ) & ( dataSlice['fiveSigmaDepth'] > self.mag_lim[1])
        
        
        magSlice0 = dataSlice['fiveSigmaDepth'][idx0]
        magSlice1 = dataSlice['fiveSigmaDepth'][idx0]
        
        if (len(magSlice0)>0) and (len(magSlice1)>0):
        
            depth0 = np.median( magSlice0 )
            depth1 = np.median( magSlice1 )
        else:
            depth0 = np.nan
            depth1 = np.nan
        
        dic = {'f0': f0,
                'f1': f1,
                'depth0': depth0,
                'depth1': depth1,
                  }
        
        if self.dataout:
            # return dT
            result = dic
            return result
        else:
            # return mean of depth between two filters
            result = np.mean([depth0, depth1])
            return float(result)    

