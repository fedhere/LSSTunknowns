import numpy as np\
import pandas as pd\
from scipy.stats import *\
from astropy import units as u\
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.coordinates import (CartesianRepresentation,
                                 CartesianDifferential, Galactic)
### LSST dependencies\
import rubin_sim.maf.db as db
from rubin_sim.maf.metrics import BaseMetric
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
from rubin_sim.maf.utils.mafUtils import radec2pix
from rubin_sim.maf.utils import m52snr, astrom_precision, sigma_slope
from rubin_sim.utils import hpid2RaDec, equatorialFromGalactic
import rubin_sim.maf.slicers as slicers
from rubin_sim.data import get_data_dir
from opsimUtils import *\

__all__ = [
    "generate_pm_slicer",
    "LSPMmetric",
    "microlensing_amplification",
    "microlensing_amplification_fsfb",
    "info_peak_before_t0",
    "fisher_matrix",
    "coefficients_pspl",
    "coefficients_fsfb",
]

class LSPMmetric(BaseMetric):
    def __init__(self, metricName='LSPMmetric', f='g', surveyduration=10, snr_lim=5., sigma_threshold=1,
                 m5Col='fiveSigmaDepth', prob_type='uniform', U=np.arange(-100, 100, 25), V=np.arange(-100, 100, 25),
                 W=np.arange(-100, 100, 25),
                 mjdCol='observationStartMJD', filterCol='filter', seeingCol='seeingFwhmGeom', dataout=False,
                 **kwargs):
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
        # simulation parameters
        self.U, self.V, self.W = U, V, W  # Galactic coordinate of the velocity in km/s
        self.prob_type = prob_type  # if not uniform, is the path to the file with the costumized velocity distribution function
        # output
        self.dataout = dataout  # to have as output all the simulated observed data set dataout=True, otherwise the relative error for

        sim = pd.read_csv('hyperstar_uniform.csv', usecols=['MAG', 'MODE', 'd', 'PM', 'PM_out']) #stellar population as described in Ragosta et al.
        self.simobj = sim
        if self.dataout:
            super(LSPMmetric, self).__init__(col=[self.mjdCol, self.filterCol, self.m5Col, self.seeingCol, 'night'],
                                             metricDtype='object', units='', metricName=metricName,
                                             **kwargs)
        else:
            super(LSPMmetric, self).__init__(col=[self.mjdCol, self.filterCol, self.m5Col, self.seeingCol, 'night'],
                                             units='Proper Motion relative error', metricName=metricName,
                                             **kwargs)

        np.seterr(over='ignore', invalid='ignore')
        # typical velocity distribution from litterature (Binney et Tremain- Galactic Dynamics)

    def position_selection(self,R, z):
        #from https://www.aanda.org/articles/aa/full_html/2017/02/aa27346-15/aa27346-15.html
        # costants\
        x0,y0,z0 = 0.68,0.28,0.26 #kpc
        M = 606*2.32*10**7 #Msum
        b = 0.39 #kpc
        r = np.sqrt(R**2+z**2)
        numb = 3*b**2*M 
        denb = 4*np.pi*(R**2+z**2)**(5/2)
        rhob = numb/denb
        
        Md,at,aT,bt,bT = 3960*2.32*10**7,5.31,2.0,0.25,0.8 # Msum, kpc
        num_thin= bt/4/np.pi*(at*R**2+(at+3*(z**2+bt**2)**(1/2)*(at+(z**2+bt**2)**(1/2))))*Md
        den_thin= (R**2+(at+(z**2+bt**2)**(1/2))**2)**(5/2)*(z**2+bt**2)**(3/2)
        num_thick= bT/4/np.pi*(aT*R**2+(aT+3*(z**2+bT**2)**(1/2)*(aT+(z**2+bT**2)**(1/2))))*Md
        den_thick= (R**2+(aT+(z**2+bT**2)**(1/2))**2)**(5/2)*(z**2+bT**2)**(3/2)
        rhothin = num_thin/den_thin
        rhothick = num_thick/den_thick
        rhod=  rhothin+rhothick

        Mh=4615*2.32*10**7
        ah=12
        C_rhoh = 1/4/np.pi/ah/r**2*Mh
        rhoh = C_rhoh*(r/ah)**(1.02)*((2.02+(r/ah)**(1.02))/(1+(r/ah)**(1.02))**2)
        
        p_prob = np.array([rhoh, rhob, rhod]) / np.nansum(np.array([rhoh, rhob, rhod]), axis=0)
        print(p_prob)
        component = np.array(['H', 'B', 'D'])

        idx = np.array(p_prob == np.nanmax(p_prob, axis=0))
        res = np.array([component[idx[:, i]][0] for i in range(np.shape(idx)[1])])
        return res
      
    def DF(self, V_matrix, component, R, z):
        '''retrive the probability distribution in the given region of the Galaxy. '''
        P = np.empty(shape=(np.size(component), np.shape(V_matrix)[1]))
        # Halo
        iH = np.where(component == 'H')
        if np.size(iH) > 0:
            v = np.sqrt(V_matrix[0, :] ** 2 + V_matrix[1, :] ** 2 + V_matrix[2, :] ** 2)
            vesc = 575  # km/s, escape velocity
            vsun = 220  # km/s, velocity of the Sun
            N = 1.003   # normalization constant
            P[iH, :] = 4 * N / vsun / np.sqrt(np.pi) * (v / vsun) ** 2 * np.exp(-v ** 2 / vsun)  #probability velocity distribution function from Baushev et al 2012
        # Bulge
        iB = np.where(component == 'B')
        if np.size(iB) > 0:
            v = np.sqrt(V_matrix[0, :] ** 2 + V_matrix[1, :] ** 2 + V_matrix[2, :] ** 2)
            disp = 140  # km/s, dispersion velocity at the center of the Galaxy
            P[iB, :] = np.exp(-v ** 2 / 2 / disp ** 2) / np.sqrt(np.pi) / disp       #probability velocity distribution function from Zhou et al 2017
        # Disk
        iD = np.where(component == 'D')
        if np.size(iD) > 0:
            # parameters described Table 1 in Binney 2010
            k = 0.25
            q = 0.45
            Rd = 3.2 *10**3 # pc
            sigmaz0 = 19  # km/s
            sigmar0 = 33.5  # km/s
            beta = 0.33
            L0 = 10  # km/s
            sigma = 300 * 10 ** (-6)  # Ms/kpc**2
            # parameters
            v = V_matrix[2, :]
            Jz = v ** 2 / 2 + v ** 2 / 2 * np.log(R[iD, np.newaxis] ** 2 + z[iD, np.newaxis] ** 2 / 0.8 ** 2)
            Ar = k * Jz / np.exp(2 * q * (120*10**3  - R[iD, np.newaxis]) / Rd) / sigmar0
            Az = k * Jz / np.exp(2 * q * (120*10**3  - R[iD, np.newaxis]) / Rd) / sigmaz0
            P[iD, :] = v ** 2 / Jz * sigma * (1 + np.tanh(R[iD, np.newaxis] * v / L0)) * np.exp(
                -k * Jz / (sigmar0 * np.exp(2 * q * (120*10**3  - R[iD, np.newaxis]) / Rd)) ** 2) / (
                       np.pi * k * sigmaz0 * np.exp(2 * q * (120*10**3  - R[iD, np.newaxis]) / Rd))        #probability velocity distribution function from Binney 2010
        return P
    def V_conversion(self,V_GC, ra,dec,d): #https://www.astro.utu.fi/~cflynn/galdyn/lecture7.html
        c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, distance=d*u.pc)
        cc = c.transform_to(coord.Galactocentric)
        U,V,W = V_GC
        gc = Galactic(u=cc.cartesian.x, v=cc.cartesian.y, w=cc.cartesian.z, 
                      U=U*u.km/u.s, V=V*u.km/u.s, W=W*u.km/u.s,representation_type=CartesianRepresentation, 
                      differential_type=CartesianDifferential)  
        pm  = gc.transform_to(coord.ICRS)
        return (pm.pm_ra_cosdec.value, pm.pm_dec.value)
    def coor_gal(self,Ra,Dec,d):
        c = SkyCoord(ra=Ra*u.degree, dec=Dec*u.degree, distance=d*u.pc)
        cc = c.transform_to(coord.Galactocentric)
        z = cc.cylindrical.z.value
        R = cc.cylindrical.rho.value
        return (z,R)
    def run(self, dataSlice, slicePoint=None):
        np.random.seed(2500)
        ''' simulation of the measured proper motion '''
        #sigma_slope_ = np.vectorize(sigma_slope)
        obs = np.where((dataSlice['filter'] == self.f) & (
        dataSlice[self.mjdCol] < min(dataSlice[self.mjdCol]) + 365 * self.surveyduration))  #select the observations in the reference filter within the survey duration
        d = np.array(self.simobj['d'])
        M = np.array(self.simobj['MAG'])
        fieldRA, fieldDec = np.radians(np.mean(dataSlice['fieldRA'])), np.radians(np.mean(dataSlice['fieldDec'])) #galaxtic coordinates
        #z, R = d * np.sin(np.radians(fieldRA)), d * np.cos(np.radians(fieldRA)) #azimuthal and radial component of the 3Dvector for the position of the star
        z,R = self.coor_gal(np.mean(dataSlice['fieldRA']),np.mean(dataSlice['fieldDec']),d)
        gal_com = self.position_selection(R, z) #estimate if the star is in the Halo, the Bulge or Disk
        mjd = dataSlice[self.mjdCol][obs]
        fwhm = dataSlice[self.seeingCol][obs]
        V_galactic = np.vstack((self.U, self.V, self.W))    #velocities array
        Pv = self.DF(V_galactic, gal_com, R, z)             #estimate the velocity distibution function for each star -- because we do not know a-priori the position of the stars we estimate DF for all the positions--
        marg_P = np.nanmean(Pv / np.nansum(Pv, axis=0), axis=0)  #marginalize the probability function for each star over the positions
        marg_P /= np.nansum(marg_P) #normalize the marginalized distribution
        vel_idx = np.random.choice(np.arange(0, np.shape(V_galactic)[1], 1)[np.isfinite(marg_P)],
                                   p=marg_P[np.isfinite(marg_P)], size=3)  #random selection of the velocity for each star following the velocity distribution function
        mu_components = np.array([self.V_conversion(np.array([V_galactic[i,vel_idx[i]] for i in range(np.shape(V_galactic)[0])]) ,np.mean(dataSlice['fieldRA']),np.mean(dataSlice['fieldDec']),dd)for dd in d])    #selection of the trasversal component
        mu_ra, mu_dec = mu_components[:,0],mu_components[:,1]
        mu = np.sqrt(mu_ra**2+mu_dec**2)
        #selection of the transversal component for the proper motion of the stars from the unusual population
        if self.prob_type == 'uniform':
            p_vel_unusual = uniform(-100, 100)
            v_unusual = p_vel_unusual.rvs(size=(3, np.size(d)))
            mu_ra_un, mu_dec_un= self.V_conversion(v_unusual,np.mean(dataSlice['fieldRA']),np.mean(dataSlice['fieldDec']),d) 
            mu_unusual = np.sqrt(mu_ra_un**2+ mu_dec_un**2)
        else:
            p_vel_un = pd.read_csv(self.prob_type)
            vel_idx = np.random.choice(p_vel_un['vel'], p=p_vel_un['fraction'] / np.sum(p_vel_un['fraction']), size=3)
            mu_un_components = np.array([self.V_conversion(np.array([V_galactic[i,vel_idx[i]] for i in range(np.shape(V_galactic)[0])]),
                                           np.mean(dataSlice['fieldRA']),np.mean(dataSlice['fieldDec']),dd) for dd in d] )
            mu_ra_un, mu_dec_un = mu_un_components[:,0], mu_un_components[:,1]
            mu_unusual = np.sqrt(mu_ra_un**2+ mu_dec_un**2) 
            
        #direction = np.random.choice((-1, 1))  #select the direction of the proper motion
        #mu = direction * vT / 4.75 / d  *1e3        #estimate the proper motion of the usual population
        #mu_unusual = direction * vT_unusual / 4.75 / d  *1e3 #estimate the proper motion of the unusual population
        snr = m52snr(M[:, np.newaxis], dataSlice[self.m5Col][obs])  # select objects above the limit magnitude threshold whatever the magnitude of the star is
        row, col = np.where(snr > self.snr_lim) #select the snr above the threshold
        precis = astrom_precision(dataSlice[self.seeingCol][obs], snr[row, :])  #estimate the uncertainties on the position
        sigmapm = sigma_slope(dataSlice[self.mjdCol][obs], precis) * 365.25 * 1e3   #estimate the uncertainties on the proper motion
        if np.size(mjd)>2:
            Times = np.sort(mjd)
            dt = np.array(list(combinations(Times, 2)))
            DeltaTs = np.absolute(np.subtract(dt[:, 0], dt[:, 1])) #estimate all the possible time gaps given the dates of the observations
            DeltaTs = np.unique(DeltaTs)
            dt_pm = 0.05 * np.amin(dataSlice[self.seeingCol][obs]) / np.absolute(mu[np.unique(row)]) # time gap of the motion given the proper motion
            dt_pm_unusual = 0.05 * np.amin(dataSlice[self.seeingCol][obs]) / np.absolute(mu_unusual[np.unique(row)])
            selection_usual = np.where((dt_pm > min(DeltaTs)) & (dt_pm < max(DeltaTs)) & (np.absolute(mu_unusual[np.unique(row)]) > sigmapm)) #select measurable proper motions
            selection_unusual = np.where((dt_pm_unusual > min(DeltaTs)) & (dt_pm_unusual < max(DeltaTs)) & (np.absolute(mu_unusual[np.unique(row)]) > sigmapm)) 
           #select measurable proper motions

            if (np.size(selection_usual)>0) and (np.size(selection_unusual) > 0):
                #pa = np.random.uniform(0, 2 * np.pi, len(mu_unusual[selection_usual])) #select the poition on the inclination with respect the line of sight
                #pa_unusual = np.random.uniform(0, 2 * np.pi, len(mu_unusual[selection_unusual]))
                #pm_alpha, pm_delta = mu[selection_usual] * np.sin(pa), mu[selection_usual] * np.cos(pa) #estimate the components of the proper motion for the usual population
                #pm_un_alpha, pm_un_delta = mu_unusual[selection_unusual] * np.sin(pa_unusual), mu_unusual[selection_unusual] * np.cos(pa_unusual)   #estimate the components of the proper motion for the unusual population
                pm_alpha, pm_delta = mu_ra[selection_unusual], mu_dec[selection_unusual]
                pm_un_alpha, pm_un_delta = mu_ra_un[selection_unusual], mu_dec_un[selection_unusual]
                mu = mu[selection_usual]
                mu_unusual = mu_unusual[selection_unusual]
                variance_k = np.array([np.std(mu[np.where(gal_com[selection_usual] == p)]) for p in ['H', 'D', 'B']])
                variance_mu = np.std(mu)
                sigmaL = np.sqrt(
                    np.prod(variance_k, where=np.isfinite(variance_k)) ** 2 + variance_mu ** 2 + np.nanmedian(
                        sigmapm) ** 2)  #estimate the variance of the likelihood distribution
                unusual = np.where((mu_unusual < np.mean(mu_unusual) - self.sigma_threshold * sigmaL / 2) | (
                mu_unusual > np.mean(mu_unusual) + self.sigma_threshold * sigmaL / 2))  #select the proper motion measurement outside the n*sigma limit
                res = np.size(unusual) / np.size(selection_unusual) #estimate the fraction of unusual proper motion that we can identify as unusual

                if self.dataout:
                    dic = {'detected': res,
                           'pixID': radec2pix(nside=16, ra=np.radians(fieldRA), dec=np.radians(fieldDec)),
                           'PM': pd.DataFrame({'pm_alpha': pm_alpha, 'pm_delta': pm_delta}),
                           'PM_un': pd.DataFrame({'pm_alpha': pm_un_alpha, 'pm_delta': pm_un_delta})}
                    return dic
                else:
                    return res
                    #print('the selection worked')
                    #print('res={}'.format(res))
            else:
                res=0
                #print('the selection did not work')
                #print('res={}'.format(res))
                return res
        else:
                res=0
                #print('not enough visits')
                #print('res={}'.format(res))
                return res
            
            
def generate_pm_slicer(
    n_events=10000,
    seed=42,
    nside=128,
    filtername="g",
):
    """
    Generate a UserPointSlicer with a population of proper motion. To be used with
    LS,TPM,CI metrics
    Parameters
    ----------
    n_events : int (10000)
        Number of star locations to generate
    seed : float (42)
        Random number seed
    nside : int (128)
        HEALpix nside, used to pick which stellar density map to load (TRImap folder has map only for nside=64,128)
    filtername : str ('r')
        The filter to use for the stellar density map
    """
    np.random.seed(seed)
    sim = pd.read_csv('hyperstar_uniform.csv', usecols=['MAG', 'MODE', 'd', 'PM', 'PM_out'])
    map_dir = os.path.join(get_data_dir(), "maps", "TriMaps")
    data = np.load(
        os.path.join(map_dir, "TRIstarDensity_%s_nside_%i.npz" % (filtername, nside))
    )
    star_density = data["starDensity"].copy()
    # magnitude bins
    bins = data["bins"].copy()
    data.close()

    star_mag = sim['MAG']
    bin_indx = np.where(bins[1:] >= star_mag[:,np.newaxis])[0].min()
    density_used = star_density[:, bin_indx].ravel()
    order = np.argsort(density_used)
    # I think the model might have a few outliers at the extreme, let's truncate it a bit
    density_used[order[-10:]] = density_used[order[-11]]

    # now, let's draw N from that distribution squared
    dist = density_used[order] ** 2
    cumm_dist = np.cumsum(dist)
    cumm_dist = cumm_dist / np.max(cumm_dist)
    uniform_draw = np.random.uniform(size=nevents)
    indexes = np.floor(np.interp(uniform_draw, cumm_dist, np.arange(cumm_dist.size)))
    hp_ids = order[indexes.astype(int)]
    gal_l, gal_b = hpid2RaDec(nside, hp_ids, nest=True)
    ra, dec = equatorialFromGalactic(gal_l, gal_b)

    # Set up the slicer to evaluate the catalog we just made
    slicer = slicers.UserPointsSlicer(ra, dec, lat_lon_deg=True, badval=0)
    # Add any additional information about each object to the slicer

    return slicer            
