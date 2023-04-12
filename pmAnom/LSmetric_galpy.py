import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.coordinates import (CartesianRepresentation,
                                 CartesianDifferential, Galactic)
from astropy.io import ascii
from scipy.stats import truncnorm
from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.utils.mafUtils import radec2pix
from rubin_sim.maf.utils import m52snr, astrom_precision, sigma_slope
from galpy.potential import NFWPotential, MiyamotoNagaiPotential, PowerSphericalPotentialwCutoff
from galpy.df import isotropicNFWdf, kingdf, dehnendf



class LSPMmetric(BaseMetric):
    def __init__(self, metricName='LSPMmetric', f='g', surveyduration=10, snr_lim=5., sigma_threshold=1,
                 m5Col='fiveSigmaDepth', mjdCol='observationStartMJD', filterCol='filter', seeingCol='seeingFwhmGeom', dataout=False,
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
        # output
        self.dataout = dataout  # to have as output all the simulated observed data set dataout=True, otherwise the relative error for

        if self.dataout:
            super(LSPMmetric, self).__init__(col=[self.mjdCol, self.filterCol, self.m5Col, self.seeingCol, 'night'],
                                             metricDtype='object', units='', metricName=metricName,
                                             **kwargs)
        else:
            super(LSPMmetric, self).__init__(col=[self.mjdCol, self.filterCol, self.m5Col, self.seeingCol, 'night'],
                                             units='Proper Motion relative error', metricName=metricName,
                                             **kwargs)

        np.seterr(over='ignore', invalid='ignore')
    
    def sample_simulation(self, R,z,gl, sample):
        """
        R = Galactocentric cylindrical radius
        z = Galactocentric cylindrical azimuth

        """
        # Gravitational Potential models for Disk, Halo, Bulge 
        #(We used recommendation parameters from https://arxiv.org/abs/1412.3451) 
        Disk_pot= MiyamotoNagaiPotential(a=3./8.,b=0.28/8.,normalize=.6)
        Halo_pot= NFWPotential(a=16/8.,normalize=.35)
        Bulge_pot= PowerSphericalPotentialwCutoff(alpha=1.8,rc=1.9/8.,normalize=0.05)
        # components desinties obtained from galpy through 
        # numerical integration of Poisson equation
        rho_disk = Disk_pot.dens(R/8.,z/8.)
        rho_halo = Halo_pot.dens(R/8.,z/8.)
        rho_bulge = Bulge_pot.dens(R/8.,z/8.)       
        Norm_along_R = np.sum([rho_bulge, rho_disk, rho_halo], axis = 0 )
        Bulge_fraction, Disk_fraction, Halo_fraction = np.array([rho_bulge, rho_disk, rho_halo]) / Norm_along_R
        Nbulge = np.array( list( map( int, Bulge_fraction*sample*rho_bulge ) ) ).sum()
        Nhalo = np.array( list( map( int, Halo_fraction*sample*rho_halo ) ) ).sum()
        Ndisk = np.array( list( map( int, Disk_fraction*sample*rho_disk ) ) ).sum() 
        #Distribution Functions inizialization (according to galpy recommendation from https://arxiv.org/abs/1412.3451)
        halo_df= isotropicNFWdf(pot=Halo_pot)    
        bulge_df = kingdf(M=2.3,rt=1.4,W0=3.)
        disk_df = dehnendf(beta=0.,profileParams=(1./4.,1.,0.2))

        #drawing samples from DF for each component
        sample_vh = halo_df.sample(R=R/8.,z=z/8.,n=Nhalo)
        sample_vb = bulge_df.sample(R=R/8.,z=z/8.,n=Nbulge)
        sample_vd = list(itertools.chain.from_iterable([disk_df.sample(n=Ndisk,los=l) for l in gl[np.isfinite(gl)]]))
        total_sample_v = list(itertools.chain.from_iterable([sample_vh,sample_vb,sample_vd]))
        self.galcomp = np.array(['H']*int(Nhalo) + ['B']*int(Nbulge) + ['D']* int(Ndisk*gl[np.isfinite(gl)].size))
        #BaSTI (https://arxiv.org/pdf/2111.09285.pdf) isochrones for magnitudes
        # Age Bulge (https://iopscience.iop.org/article/10.3847/1538-4357/abaeee)
        # Age Halo (https://doi.org/10.1038/nature11062)
        # Age Disk (https://arxiv.org/abs/1912.02816)
        d = np.sqrt(R**2+z**2)
        Bulge_magdist = np.array(ascii.read('Bulge.isc_sloan')['col6'])+5*np.log10(d[:,None]*10**2)+5 #for g band distribution
        Disk_magdist = np.array(ascii.read('Disk.isc_sloan')['col6'])+5*np.log10(d[:,None]*10**2)+5 #for g band distribution
        Halo_magdist = np.array(ascii.read('Halo.isc_sloan')['col6'])+5*np.log10(d[:,None]*10**2)+5 #for g band distribution

        #drawing samples from isochrones
        sample_mb = np.random.choice(Bulge_magdist.flatten(), Nbulge)
        sample_md = np.random.choice(Disk_magdist.flatten(), Ndisk*gl[np.isfinite(gl)].size)
        sample_mh = np.random.choice(Halo_magdist.flatten(), Nhalo)
        total_sample_m = list(itertools.chain.from_iterable([sample_mh,sample_mb,sample_md]))
        return (total_sample_m, total_sample_v)

    def coor_gal(self,Ra,Dec,d):
        '''
        function to convert RA, Dec to galactocetric coordinate
        ______________
        
        RA, Dec, array
        d, kpc, distance from the Sun
        '''
        c = SkyCoord(ra=Ra*u.degree, dec=Dec*u.degree, distance=d*1000*u.pc)
        cc = c.transform_to(coord.Galactocentric)
        z = cc.cylindrical.z.value/1000
        R = cc.cylindrical.rho.value/1000
        R0 = 8. # kpc, Sun's distance form the Galaxy center
        d_ = d*np.cos(cc.galactic.b.value)
        gl = np.arccos((-(R/R0)**2+1+(d_/R0)**2)/(2*d_/R0)) #equation to transform galactocentric R to galactic latitude
        self.coord_system = cc
        return (z,R,gl) 
    
    def V_conversion(self,V_GC, coor_system): #https://www.astro.utu.fi/~cflynn/galdyn/lecture7.html
        '''
        function to convert velocities to proper motion
        ______________
        
        V_GC, array with galpy object, if matrix: velocity in the cartesian coordinate system
        coor_system, astropy Skycoor object
        '''
        try:
            pm_ra_cosdec, pm_dec = np.array([x.pmra() for x in V_GC]), np.array([x.pmdec() for x in V_GC])            
        except:
            U,V,W = V_GC[:,0],V_GC[:,1],V_GC[:,2]
            gcs = Galactic(u=coor_system.cartesian.x, v=coor_system.cartesian.y, w=coor_system.cartesian.z, 
                      U=U*u.km/u.s, V=V*u.km/u.s, W=W*u.km/u.s,representation_type=CartesianRepresentation, 
                      differential_type=CartesianDifferential)
            pm_ra_cosdec ,pm_dec = [gc.transform_to(coord.ICRS).pm_ra_cosdec.value for gc in gcs],\
                [gc.transform_to(coord.ICRS).pm_dec.value for gc in gcs]
        return (np.array(pm_ra_cosdec), np.array(pm_dec))
    
    def run(self, dataSlice, slicePoint=None):
        np.random.seed(2500)
        ''' simulation of the measured proper motion '''
        #select the observations in the reference filter within the survey duration
        
        obs = np.where((dataSlice['filter'] == self.f) & (
        dataSlice[self.mjdCol] < min(dataSlice[self.mjdCol]) + 365 * self.surveyduration))  
        d = truncnorm.rvs(-1, 5,loc=8.2,scale=8,size = 5000)
        #galaxtic coordinates
        fieldRA, fieldDec = np.mean(dataSlice['fieldRA']), np.mean(dataSlice['fieldDec']) 
        mjd = dataSlice[self.mjdCol][obs]
        if np.size(mjd)>2:
            z,R, gl = self.coor_gal(np.mean(dataSlice['fieldRA']),np.mean(dataSlice['fieldDec']),d)
            mags, Vs = self.sample_simulation(R, z, gl,d.size)          
            mu_components =self.V_conversion(Vs,self.coord_system)   
            mu_ra, mu_dec = mu_components[0],mu_components[1]
            mu = np.sqrt(mu_ra**2+mu_dec**2)

            v_unusual = np.random.uniform(-100, 100 ,size=(np.size(d),3))

            mu_ra_un, mu_dec_un= self.V_conversion(v_unusual,self.coord_system) 
            mu_unusual = np.sqrt(mu_ra_un**2+ mu_dec_un**2)

            # select objects above the limit magnitude threshold whatever the magnitude of the star is
            snr = m52snr(np.array(mags)[:, np.newaxis], dataSlice[self.m5Col][obs])  
            #select the snr above the threshold
            row, col = np.where(snr > self.snr_lim) 
            #estimate the uncertainties on the position
            precis = astrom_precision(dataSlice[self.seeingCol][obs], snr[row, :])  
            #estimate the uncertainties on the proper motion
            sigmapm = sigma_slope(dataSlice[self.mjdCol][obs], precis) * 365.25 * 1e3   
        
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
            #select measurable proper motions
            if np.size(selection_usual)>0:# and 
                if np.size(selection_unusual) > 0:
                    pm_alpha, pm_delta = mu_ra[selection_unusual], mu_dec[selection_unusual]
                    pm_un_alpha, pm_un_delta = mu_ra_un[selection_unusual], mu_dec_un[selection_unusual]
                    mu = mu[selection_usual]
                    mu_unusual = mu_unusual[selection_unusual]
                    variance_k = np.array([np.std(mu[np.where(self.galcomp[selection_usual] == p)]) for p in ['H', 'D', 'B']])
                    variance_mu = np.std(mu)
                    #estimate the variance of the likelihood distribution
                    sigmaL = np.sqrt(
                        np.prod(variance_k, where=np.isfinite(variance_k)) ** 2 + variance_mu ** 2 + np.nanmedian(
                            sigmapm) ** 2)  
                    #select the proper motion measurement outside the n*sigma limit
                    unusual = np.where((mu_unusual < np.mean(mu_unusual) - self.sigma_threshold * sigmaL / 2) | (
                    mu_unusual > np.mean(mu_unusual) + self.sigma_threshold * sigmaL / 2))  
                    
                    #estimate the fraction of unusual proper motion that we can identify as unusual
                    res = np.size(unusual) / np.size(selection_unusual) 

                    if self.dataout:
                        dic = {'detected': res,
                               'pixID': radec2pix(nside=16, ra=np.radians(fieldRA), dec=np.radians(fieldDec)),
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
        #print('In RA= {}, DEC= {} fraction unknown = {}'.format(fieldRA, fieldDec, res))
        return res
