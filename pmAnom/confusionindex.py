import numpy as np
from astropy.io import fits
from builtins import zip
from itertools import combinations
### LSST dependencies
from lsst.sims.maf.metrics import BaseMetric
from lsst.sims.maf.utils import m52snr, astrom_precision, sigma_slope
from opsimUtils import *


class reducedPM(BaseMetric):
    def __init__(self, fits_filename='sa93.all.fits', snr_lim=5, mode=None, MagIterLim=[0, 1, 1], surveyduration=10,
                 metricName='reducedPM', m5Col='fiveSigmaDepth', atm_err=0.01,
                 mjdCol='observationStartMJD', filterCol='filter', seeingCol='seeingFwhmGeom', dataout=True, **kwargs):
        self.mjdCol = mjdCol    # Column name for modified julian date of the observation
        self.m5Col = m5Col      # Column name for the five sigma limit
        self.seeingCol = seeingCol  # Column name for the geometrical seeing
        self.filterCol = filterCol  # Column name for filters
        self.snr_lim = snr_lim  # float, threshold for the signal to noise ratio (snr), all the signals with a snr>snr_lim are detected
        self.fits_filename = fits_filename
        self.dataout = dataout  #to have as output all the simulated observed data set dataout=True, otherwise the relative error for
        self.mode = mode    # simulate the structure farther away (distance) or in a more dense environment (density)
        self.MagIterLim = MagIterLim #array, dm values to simulate the experiment in the given mode
        self.surveyduration = surveyduration    # integer, number of years from the start of the survey
        self.atm_err = atm_err      #the expected centroiding error due to the atmosphere, in arcseconds. Default 0.01.
        if self.dataout:
            super(reducedPM, self).__init__(col=[self.mjdCol, self.filterCol, self.m5Col, self.seeingCol, 'night'],
                                            metricDtype='object', units='', metricName=metricName,
                                            **kwargs)
        else:
            super(reducedPM, self).__init__(col=[self.mjdCol, self.filterCol, self.m5Col, self.seeingCol, 'night'],
                                            units='', metricDtype='float', metricName=metricName,
                                            **kwargs)

        data_sag93 = fits.open(self.fits_filename) #data of the structure
        table = data_sag93[1].data
        mu_sag = np.transpose(np.sqrt(table['MUX'] ** 2 + table['MUY'] ** 2))
        M_sag = np.transpose(table['GMAG'])
        self.mu_sag = mu_sag
        self.mag_sag = M_sag
        self.gr = np.transpose(table['GMAG'] - table['RMAG'])

    def run(self, dataSlice, slicePoint=None):
        obs = np.where(dataSlice[self.mjdCol] < min(dataSlice[self.mjdCol]) + 365 * self.surveyduration)  #select the observations within the survey duration
        sigma_slope = np.vectorize(sigma_slope)
        deltamag = np.arange(self.MagIterLim[0], self.MagIterLim[1], self.MagIterLim[2])
        out = {}
        for dm in deltamag:
            if self.mode == 'distance':
                pmnew = self.mu_sag / (10 ** (dm / 5)) #update the proper motion ad we measure it from farther away
                mag = self.mag_sag + dm
            elif self.mode == 'density':
                pmnew = self.mu_sag
                mag = self.mag_sag + dm  #update the magnitude ad we simulate a denser environment
            else:
                print('##### ERROR: the metric is not implemented for this mode.')

            mjd = dataSlice[self.mjdCol][obs]
            flt = dataSlice[self.filterCol][obs]
            if ('g' in flt) and ('r' in flt):   #check if we observe in the reference filters

                snr = m52snr(mag[:, np.newaxis], dataSlice[self.m5Col][obs])    # select objects above the limit magnitude threshold whatever the magnitude of the star is

                selection_mag = np.where(np.mean(snr, axis=0) > self.snr_lim)   #select the snr above the threshold
                Times = np.sort(mjd)
                dt = np.array(list(combinations(Times, 2)))
                DeltaTs = np.absolute(np.subtract(dt[:, 0], dt[:, 1])) #estimate all the possible time gaps given the dates of the observations
                DeltaTs = np.unique(DeltaTs)
                displasement_error = astrom_precision(dataSlice[self.seeingCol][obs][selection_mag],
                                                      np.mean(snr, axis=0)[selection_mag])  #estimate the uncertainties on the position
                displasement_error = np.sqrt(displasement_error ** 2 + self.atm_err ** 2)
                sigmapm = sigma_slope(dataSlice[self.mjdCol][obs][selection_mag], displasement_error)   #estimate the uncertainties on the proper motion
                sigmapm *= 365.25 * 1e3     #conversion to mas/yr
                if np.size(DeltaTs) > 0:
                    dt_pm = 0.05 * np.amin(dataSlice[self.seeingCol][obs]) / pmnew  # time gap of the motion given the proper motion
                    selection_mag_pm = np.where(
                        (dt_pm > min(DeltaTs)) & (dt_pm < max(DeltaTs)) & (np.absolute(pmnew) > sigmapm) & (
                        np.mean(snr, axis=1) > self.snr_lim))   #select measurable proper motions
                Hg = mag[selection_mag_pm] + 5 * np.log10(pmnew[selection_mag_pm]) - 10 #estimate the reduced proper motion
                sigmaHg = np.sqrt(
                    (mag[selection_mag_pm] / m52snr(mag[selection_mag_pm], np.median(dataSlice[self.m5Col]))) ** (2) + (
                    4.715 * sigmapm / pmnew[selection_mag_pm]) ** 2)     #uncertainties on the reduced proper motion
                sigmag = np.sqrt(
                    (mag[selection_mag_pm] / m52snr(mag[selection_mag_pm], np.median(dataSlice[self.m5Col]))) ** 2 + (
                    (mag[selection_mag_pm] - self.gr[selection_mag_pm]) / m52snr(
                        (mag[selection_mag_pm] - self.gr[selection_mag_pm]), np.median(dataSlice[self.m5Col]))) ** 2)   #uncertainties on the magnitude
                err_ellipse = np.pi * sigmaHg * sigmag  #area of the uncertainties ellipse in the reduced proper motion diagram (Hg vs g)
                if self.dataout:
                    stars_err_ellipse = np.array([np.nansum((([self.gr[selection_mag_pm] - gcol]) / sigmag) ** 2 + (
                    (Hg - h) / sigmaHg) ** 2 <= 1) / np.size(pmnew[selection_mag_pm]) for (gcol, h) in
                                   zip(self.gr[selection_mag_pm], Hg)])     # for each star, number of the other stars in the sample in its the uncertainty ellipse

                    out[dm] = {'fraction_err': stars_err_ellipse, 'alpha': np.size(pmnew[selection_mag_pm]) / np.size(pmnew), 'CI': err_ellipse,
                               'Hg': Hg, 'gr': self.gr[selection_mag_pm], 'sigmaHg': sigmaHg, 'sigmagr': sigmag}
                else:
                    res = np.size(pmnew[selection_mag_pm]) / np.size(pmnew)/np.median(err_ellipse)
            else:
                if self.dataout:
                    out[dm] = {'fraction_err': 0, 'alpha': 0, 'CI': 0, 'Hg': Hg, 'gr': gr, 'sigmaHg': sigmaHg, 'sigmagr': sigmag}
                else:
                    res = 0
        return res
