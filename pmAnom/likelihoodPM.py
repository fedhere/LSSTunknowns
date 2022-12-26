{\rtf1\ansi\ansicpg1252\cocoartf2706
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww29200\viewh15880\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
import pandas as pd\
from scipy.stats import *\
from astropy import units as u\
from astropy.coordinates import SkyCoord\
### LSST dependencies\
from lsst.sims.maf.metrics import BaseMetric\
from lsst.sims.maf.utils.mafUtils import radec2pix\
from lsst.sims.maf.utils import m52snr, astrom_precision, sigma_slope\
from opsimUtils import *\
\
\
class LSPMmetric(BaseMetric):\
    def __init__(self, metricName='LSPMmetric', f='g', surveyduration=10, snr_lim=5., sigma_threshold=1,\
                 m5Col='fiveSigmaDepth', prob_type='uniform', U=np.arange(-100, 100, 25), V=np.arange(-100, 100, 25),\
                 W=np.arange(-100, 100, 25),\
                 mjdCol='observationStartMJD', filterCol='filter', seeingCol='seeingFwhmGeom', dataout=False,\
                 **kwargs):\
        # opsim\
        self.mjdCol = mjdCol  # Column name for modified julian date of the observation\
        self.m5Col = m5Col    # Column name for the five sigma limit\
        self.seeingCol = seeingCol  # Column name for the geometrical seeing\
        self.filterCol = filterCol  # Column name for filters\
        # selection criteria\
        self.surveyduration = surveyduration # integer, number of years from the start of the survey\
        self.snr_lim = snr_lim               # float, threshold for the signal to noise ratio (snr), all the signals with a snr>snr_lim are detected\
        self.f = f                          # string, filter used for the observations\
        self.sigma_threshold = sigma_threshold  # integer,\
        # simulation parameters\
        self.U, self.V, self.W = U, V, W  # Galactic coordinate of the velocity in km/s\
        self.prob_type = prob_type  # if not uniform, is the path to the file with the costumized velocity distribution function\
        # output\
        self.dataout = dataout  # to have as output all the simulated observed data set dataout=True, otherwise the relative error for\
\
        sim = pd.read_csv('hyperstar_uniform.csv', usecols=['MAG', 'MODE', 'd', 'PM', 'PM_out']) #stellar population as described in Ragosta et al.\
        self.simobj = sim\
        if self.dataout:\
            super(LSPMmetric, self).__init__(col=[self.mjdCol, self.filterCol, self.m5Col, self.seeingCol, 'night'],\
                                             metricDtype='object', units='', metricName=metricName,\
                                             **kwargs)\
        else:\
            super(LSPMmetric, self).__init__(col=[self.mjdCol, self.filterCol, self.m5Col, self.seeingCol, 'night'],\
                                             units='Proper Motion relative error', metricName=metricName,\
                                             **kwargs)\
\
        np.seterr(over='ignore', invalid='ignore')\
        # typical velocity distribution from litterature (Binney et Tremain- Galactic Dynamics)\
\
    def position_selection(self, R, z):\
        '''\
        density profile in the Galaxy from Binney & Tremain, eq. 2.207a (bulge), 2.209(halo), 2.210 (disk)\
        '''\
        # costants\
        ab = 1  # kpc\
        ah = 1.9  # kpc\
        alphab = 1.8\
        alphah = 1.63\
        betah = 2.17\
        rb = 1.9  # Kpc, radius bulge\
        qb = 0.6    #ratio of the axis of the bulge\
        qh = 0.8    #ratio of the axis of the halo\
        z0 = 0.3  # kpc\
        z1 = 1  # kpc\
        Rd = 3.2  # kpc, disk radius\
        rho0b = 0.3 * 10 ** (-9)  # Mskpc^-3, normalization density of bulge\
        rho0h = 0.27 * 10 ** (-9)  # Mskpc^-3, normalization density of halo\
        sigma = 300 * 10 ** (-6)  # Mskpc^-2\
        alpha1 = 0.5\
        # parametes\
        mb = np.sqrt(R ** 2 + z ** 2 / qb ** 2)\
        mh = np.sqrt(R ** 2 + z ** 2 / qh ** 2)\
        alpha0 = 1 - alpha1\
\
        rhoh = rho0h * (mh / ah) ** alphah * (1 + mh / ah) ** (alphah - betah)\
        rhob = rho0b * (mb / ab) ** (-alphab) * np.exp(-mb / rb)\
        rhod = sigma * np.exp(R / Rd) * (\
        alpha0 / (2 * z0) * np.exp(-np.absolute(z) / z0) + alpha1 / (2 * z1) * np.exp(-np.absolute(z) / z1))\
        p_prob = np.array([rhoh, rhob, rhod]) / np.nansum(np.array([rhoh, rhob, rhod]), axis=0)\
        component = np.array(['H', 'B', 'D'])\
\
        idx = np.array(p_prob == np.nanmax(p_prob, axis=0))\
        res = np.array([component[idx[:, i]][0] for i in range(np.shape(idx)[1])])\
        return res\
\
    def DF(self, V_matrix, component, R, z):\
        '''retrive the probability distribution in the given region of the Galaxy. '''\
        P = np.empty(shape=(np.size(component), np.size(V_matrix[0, :])))\
        # Halo\
        iH = np.where(component == 'H')\
        if np.size(iH) > 0:\
            v = np.sqrt(V_matrix[0, :] ** 2 + V_matrix[1, :] ** 2 + V_matrix[2, :] ** 2)\
            vesc = 575  # km/s, escape velocity\
            vsun = 187.5  # km/s, velocity of the Sun\
            N = 1.003   # normalization constant\
            P[iH, :] = 4 * N / vsun / np.sqrt(np.pi) * (v / vesc) ** 2 * np.exp(-v ** 2 / vsun)  #probability velocity distribution function from Baushev et al 2012\
        # Bulge\
        iB = np.where(component == 'B')\
        if np.size(iB) > 0:\
            v = np.sqrt(V_matrix[0, :] ** 2 + V_matrix[1, :] ** 2 + V_matrix[2, :] ** 2)\
            disp = 140  # km/s, dispersion velocity at the center of the Galaxy\
            P[iB, :] = np.exp(-v ** 2 / 2 / disp ** 2) / np.sqrt(np.pi) / disp       #probability velocity distribution function from Zhou et al 2017\
        # Disk\
        iD = np.where(component == 'D')\
        if np.size(iD) > 0:\
            # parameters described Table 1 in Binney 2010\
            k = 0.25\
            q = 0.45\
            Rd = 3.2  # kpc\
            sigmaz0 = 19  # km/s\
            sigmar0 = 33.5  # km/s\
            beta = 0.33\
            L0 = 10  # km/s\
            sigma = 300 * 10 ** (-6)  # Ms/kpc**2\
            # parameters\
            v = V_matrix[2, :]\
            Jz = v ** 2 / 2 + v ** 2 / 2 * np.log(R[iD, np.newaxis] ** 2 + z[iD, np.newaxis] ** 2 / 0.8 ** 2)\
            Ar = k * Jz / np.exp(2 * q * (120 - R[iD, np.newaxis]) / Rd) / sigmar0\
            Az = k * Jz / np.exp(2 * q * (120 - R[iD, np.newaxis]) / Rd) / sigmaz0\
            P[iD, :] = v ** 2 / Jz * sigma * (1 + np.tanh(R[iD, np.newaxis] * v / L0)) * np.exp(\
                -k * Jz / (sigmar0 * np.exp(2 * q * (120 - R[iD, np.newaxis]) / Rd)) ** 2) / (\
                       np.pi * k * sigmar0 * np.exp(2 * q * (120 - R[iD, np.newaxis]) / Rd))        #probability velocity distribution function from Binney 2010\
        return P\
    def coor_gal(self,Ra,Dec,d):\
        c = SkyCoord(ra=Ra*u.degree, dec=Dec*u.degree, distance=d*u.pc)\
        z = c.cylindrical.z.value\
        R = c.cylindrical.rho.value\
        return z,R\
    def run(self, dataSlice, slicePoint=None):\
        np.random.seed(2500)\
        ''' simulation of the measured proper motion '''\
        #sigma_slope_ = np.vectorize(sigma_slope)\
        obs = np.where((dataSlice['filter'] == self.f) & (\
        dataSlice[self.mjdCol] < min(dataSlice[self.mjdCol]) + 365 * self.surveyduration))  #select the observations in the reference filter within the survey duration\
        d = np.array(self.simobj['d'])\
        M = np.array(self.simobj['MAG'])\
        fieldRA, fieldDec = np.radians(np.mean(dataSlice['fieldRA'])), np.radians(np.mean(dataSlice['fieldDec'])) #galaxtic coordinates\
        #z, R = d * np.sin(np.radians(fieldRA)), d * np.cos(np.radians(fieldRA)) #azimuthal and radial component of the 3Dvector for the position of the star\
        z,R = self.coor_gal(np.mean(dataSlice['fieldRA']),np.mean(dataSlice['fieldDec']),d)\
        gal_com = self.position_selection(R, z) #estimate if the star is in the Halo, the Bulge or Disk\
        mjd = dataSlice[self.mjdCol][obs]\
        fwhm = dataSlice[self.seeingCol][obs]\
        V_galactic = np.vstack((self.U, self.V, self.W))    #velocities array\
        Pv = self.DF(V_galactic, gal_com, R, z)             #estimate the velocity distibution function for each star -- because we do not know a-priori the position of the stars we estimate DF for all the positions--\
        marg_P = np.nanmean(Pv / np.nansum(Pv, axis=0), axis=0)  #marginalize the probability function for each star over the positions\
        marg_P /= np.nansum(marg_P) #normalize the marginalized distribution\
        vel_idx = np.random.choice(np.arange(0, len(V_galactic[0, :]), 1)[np.isfinite(marg_P)],\
                                   p=marg_P[np.isfinite(marg_P)], size=3)  #random selection of the velocity for each star following the velocity distribution function\
        vT = V_galactic[0, vel_idx][2]      #selection of the trasversal component\
\
        #selection of the transversal component for the proper motion of the stars from the unusual population\
        if self.prob_type == 'uniform':\
            v_unusual = np.random.uniform(-100, 100,size=(3, np.size(d)))\
            vT_unusual = v_unusual[2, :]\
        else:\
            p_vel_un = pd.read_csv(self.prob_type)\
            vel_idx = np.random.choice(p_vel_un['vel'], p=p_vel_un['fraction'] / np.sum(p_vel_un['fraction']), size=3)\
            vT_unusual = V_galactic[0, vel_idx][2]\
\
        direction = np.random.choice((-1, 1))  #select the direction of the proper motion\
        mu = direction * vT / 4.75 / d          #estimate the proper motion of the usual population\
        mu_unusual = direction * vT_unusual / 4.75 / d  #estimate the proper motion of the unusual population\
        snr = m52snr(M[:, np.newaxis], dataSlice[self.m5Col][obs])  # select objects above the limit magnitude threshold whatever the magnitude of the star is\
        row, col = np.where(snr > self.snr_lim) #select the snr above the threshold\
        precis = astrom_precision(dataSlice[self.seeingCol][obs], snr[row, :])  #estimate the uncertainties on the position\
        sigmapm = sigma_slope(dataSlice[self.mjdCol][obs], precis) * 365.25 * 1e3   #estimate the uncertainties on the proper motion\
        if np.size(mjd)>2:\
            Times = np.sort(mjd)\
            dt = np.array(list(combinations(Times, 2)))\
            DeltaTs = np.absolute(np.subtract(dt[:, 0], dt[:, 1])) #estimate all the possible time gaps given the dates of the observations\
            DeltaTs = np.unique(DeltaTs)\
            dt_pm = 0.05 * np.amin(dataSlice[self.seeingCol][obs]) / mu[np.unique(row)] # time gap of the motion given the proper motion\
            dt_pm_unusual = 0.05 * np.amin(dataSlice[self.seeingCol][obs]) / mu_unusual[np.unique(row)]\
            selection_usual = np.where((dt_pm > min(DeltaTs)) & (dt_pm < max(DeltaTs)) & (mu[np.unique(row)] > sigmapm)) #select measurable proper motions\
            selection_unusual = np.where((dt_pm_unusual > min(DeltaTs)) & (dt_pm_unusual < max(DeltaTs)) & (mu_unusual[np.unique(row)] > sigmapm)) \
           #select measurable proper motions\
\
            if (np.size(selection_usual)+np.size(selection_unusual) > 5000):\
                pa = np.random.uniform(0, 2 * np.pi, len(mu_unusual[selection_usual])) #select the poition on the inclination with respect the line of sight\
                pa_unusual = np.random.uniform(0, 2 * np.pi, len(mu_unusual[selection_unusual]))\
                pm_alpha, pm_delta = mu[selection_usual] * np.sin(pa), mu[selection_usual] * np.cos(pa) #estimate the components of the proper motion for the usual population\
                pm_un_alpha, pm_un_delta = mu_unusual[selection_unusual] * np.sin(pa_unusual), mu_unusual[selection_unusual] * np.cos(pa_unusual)   #estimate the components of the proper motion for the unusual population\
                mu = mu[selection_usual] * 1e3\
                mu_unusual = mu_unusual[selection_unusual] * 1e3\
                variance_k = np.array([np.std(mu[np.where(gal_com[selection_usual] == p)]) for p in ['H', 'D', 'B']])\
                variance_mu = np.std(mu)\
                sigmaL = np.sqrt(\
                    np.prod(variance_k, where=np.isfinite(variance_k)) ** 2 + variance_mu ** 2 + np.nanmedian(\
                        sigmapm) ** 2)  #estimate the variance of the likelihood distribution\
                unusual = np.where((mu_unusual < np.mean(mu_unusual) - self.sigma_threshold * sigmaL / 2) | (\
                mu_unusual > np.mean(mu_unusual) + self.sigma_threshold * sigmaL / 2))  #select the proper motion measurement outside the n*sigma limit\
                res = np.size(unusual) / np.size(selection_unusual) #estimate the fraction of unusual proper motion that we can identify as unusual\
\
                if self.dataout:\
                    dic = \{'detected': res,\
                           'pixID': radec2pix(nside=16, ra=np.radians(fieldRA), dec=np.radians(fieldDec)),\
                           'PM': pd.DataFrame(\{'pm_alpha': pm_alpha, 'pm_delta': pm_delta\}),\
                           'PM_un': pd.DataFrame(\{'pm_alpha': pm_un_alpha, 'pm_delta': pm_un_delta\})\}\
                    return dic\
                else:\
                    return res\
            else:\
                res=0\
                return res\
        else:\
                res=0\
                return res}