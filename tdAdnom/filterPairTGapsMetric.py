import numpy as np
#from lsst.sims.maf.utils import radec2pix
#import lsst.sims.maf.metrics as metrics

# if has rubin_sim installed
from rubin_sim.maf.utils.mafUtils import radec2pix
import rubin_sim.maf.metrics as metrics

class filterPairTGapsMetric(metrics.BaseMetric):
    """
    Parameters:
        colname: list, ['observationStartMJD', 'filter', 'fiveSigmaDepth']
        fltpair: filter pair, eg ['r', 'i']
        mag_lim: list, fiveSigmaDepth threshold each filter, default [18, 18]
        dt_lim: list, [tmin, tmax], minimum and maximum of time gaps
        save_dT: boolean, save time gaps array as result if True
        allgaps: boolean, all possible pairs if True, else consider only nearest

    Returns:
        result: dictionary,
        reduce_tgaps: median

    """

    def __init__(self, colname=['observationStartMJD', 'filter', 'fiveSigmaDepth'],
                 fltpair=['r', 'i'], mag_lim=[18, 18], dt_lim=[0, 1.5/24],
                 save_dT=False, allgaps=True, nside=16, **kwargs):
        self.colname = colname
        self.fltpair = fltpair
        self.mag_lim = mag_lim
        self.dt_lim = dt_lim

        self.save_dT = save_dT
        self.allgaps = allgaps
        self.nside = nside

        super().__init__(col=self.colname, **kwargs)

    def run(self, dataSlice, slicePoint=None):

        f0 = self.fltpair[0]
        f1 = self.fltpair[1]

        # sort the dataSlice in order of time.
        dataSlice.sort(order='observationStartMJD')

        # select
        idx0 = ( dataSlice['filter'] == f0 ) & ( dataSlice['fiveSigmaDepth'] > self.mag_lim[0])
        idx1 = ( dataSlice['filter'] == f1 ) & ( dataSlice['fiveSigmaDepth'] > self.mag_lim[1])

        timeCol0 = dataSlice['observationStartMJD'][idx0]
        timeCol1 = dataSlice['observationStartMJD'][idx1]

        timeCol0 = timeCol0.reshape((len(timeCol0), 1))
        timeCol1 = timeCol1.reshape((len(timeCol1), 1))

        # calculate time gaps matrix
        diffmat = np.subtract(timeCol0, timeCol1.T)

        if self.allgaps:
            # collect all time gaps
            diffmat = np.abs( diffmat )
            if f0==f1:
                # get only triangle part
                dt_tri = np.tril(diffmat, -1)
                dT = dt_tri[dt_tri!=0]    # flatten lower triangle
            else:
                dT = diffmat.flatten()
        else:
            # collect only nearest
            if f0==f1:
                # get diagonal ones nearest nonzero, offset=1
                dT = np.diagonal(diffmat, offset=1)
            else:
                # get tgaps both left and right
                # keep only negative ones
                masked_ar = np.ma.masked_where(diffmat>=0, diffmat, )
                left_ar = np.max(masked_ar, axis=1)
                dT_left = -left_ar.data[~left_ar.mask]

                # keep only positive ones
                masked_ar = np.ma.masked_where(diffmat<=0, diffmat)
                right_ar = np.min(masked_ar, axis=1)
                dT_right = right_ar.data[~right_ar.mask]

                dT = np.concatenate([dT_left.flatten(), dT_right.flatten()])

        dT_lim = dT[(dT>self.dt_lim[0]) & (dT<self.dt_lim[1])]

        Nv = len(dT_lim)

        fieldRA = np.mean(dataSlice['fieldRA'])
        fieldDec = np.mean(dataSlice['fieldDec'])

        pixId = radec2pix(nside=self.nside, ra=np.radians(fieldRA), dec=np.radians(fieldDec))

        if self.save_dT:
            result = {
                'pixId': pixId,
                'Nv': Nv,
                'dT_lim': dT_lim,
                'median': np.median(dT_lim),
                'dataSlice': dataSlice
                  }
        else:
            result = {
                'pixId': pixId,
                'Nv': Nv,
                 'median': np.median(dT_lim)
                  }

        return result

    def reduce_tgaps_median(self, metric):
        """unit in hours"""
        return metric['median'] * 24

    def reduce_tgaps_Nv(self, metric):
        return metric['Nv']
