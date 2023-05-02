from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.coordinates import (CartesianRepresentation,
                                 CartesianDifferential, Galactic,ICRS)
from astropy.io import ascii
import itertools

def sample_simulation(x, y, z, d):
        """
        The sample_simulation method generates a sample of stars at a given Galactic location, where the input arguments R, z, and phi are the Galactocentric cylindrical coordinates, and d is the distance to the star. The method first computes the distance from the Galactic center for each star in the input sample, then selects the stars in the Gaia dataset whose distances are closest to the distance of the input sample, and finally, samples the velocities of these stars to generate the velocity components of the input sample. The method then computes the magnitude distribution for the stars in the Bulge, Disk, and Halo regions, using BaSTI isochrones, and draws samples from these distributions to generate the magnitudes of the input sample. The method returns the Cartesian coordinates, velocities, and magnitudes of the input sample.
        
        R = Galactocentric cylindrical radius
        z = Galactocentric cylindrical azimuth
        phi = Galactocentric cylindrical longitude
        """
        sample = d.size
        dposition = np.array(list(map(lambda xx,yy,zz : np.sqrt(xx**2+yy**2+zz**2),x, y,z)))
        index_sorted = np.argsort(d)
        d_sorted = d[index_sorted]
        idx1 = np.searchsorted(d_sorted, dposition)
        idx_close_position = np.clip(idx1 - 1, 0, len(d_sorted)-1)
        sample_vx = gaia['vx'].to_numpy()[idx_close_position]
        sample_vy = gaia['vy'].to_numpy()[idx_close_position]
        sample_vz = gaia['vz'].to_numpy()[idx_close_position]
        total_sample_v = [sample_vx,sample_vy,sample_vz]
        #BaSTI (https://arxiv.org/pdf/2111.09285.pdf) isochrones for magnitudes
        # Age Bulge (https://iopscience.iop.org/article/10.3847/1538-4357/abaeee)
        # Age Halo (https://doi.org/10.1038/nature11062)
        # Age Disk (https://arxiv.org/abs/1912.02816)
        Bulge_magdist = np.array(ascii.read('./Bulge.isc_sloan')['col6'])+5*np.log10(d[:,None]*10**2)+5 #for g band distribution
        Disk_magdist = np.array(ascii.read('./Disk.isc_sloan')['col6'])+5*np.log10(d[:,None]*10**2)+5 #for g band distribution
        Halo_magdist = np.array(ascii.read('./Halo.isc_sloan')['col6'])+5*np.log10(d[:,None]*10**2)+5 #for g band distribution

        #drawing samples from isochrones
        total_sample_m = np.random.choice(np.concatenate([Bulge_magdist.flatten(),
                                                   Disk_magdist.flatten(),
                                                   Disk_magdist.flatten()]), sample)
        return (total_sample_m, total_sample_v)

def population_synthesis(slicers, nslice
    slicer = slicers(nslice)
    with open('Gaiachallengedata.pkl', 'rb') as gaiadata:
            gaia = pickle.load(gaiadata)

    RA,Dec = slicer.slicePoints['ra'],slicer.slicePoints['dec']
    d = np.array(list(map(lambda x,y,z : np.sqrt(x**2+y**2+z**2), gaia['x'], gaia['y'], gaia['z'])))
    d_subpop = np.random.choice(d, 5000)

    c = SkyCoord(ra=np.degrees(RA)*u.degree, dec=np.degrees(Dec)*u.degree, distance=d_subpop[:,None]*1000*u.pc)
    cc = c.transform_to(coord.Galactocentric)
    x = cc.cartesian.x.value/1000
    y = cc.cartesian.y.value/1000
    z = cc.cartesian.z.value/1000

    mags, velocities = sample_simulation( x,y,z, d_subpop)

    population = defaultdict()
    population['mag'] = mags
    population['vx'],population['vy'],population['vz']= velocities[0],velocities[1],velocities[2]

    gcs = Galactic(u=cc.cartesian.x, v=cc.cartesian.y, w=cc.cartesian.z, U=population['vx']*u.km/u.s,V=population['vy']*u.km/u.s, W=population['vz']*u.km/u.s,representation_type=CartesianRepresentation, differential_type=CartesianDifferential)


    gcs = gcs.transform_to(ICRS)
    pm_ra_cosdec ,pm_dec = gcs.proper_motion.value

    population['pm_ra_cosdec'], population['pm_dec'] = pm_ra_cosdec ,pm_dec

    v_unusual = np.random.uniform(-500, 500 ,size=(np.size(d_subpop),3))
    gcs = Galactic(u=cc.cartesian.x[:,0], v=cc.cartesian.y[:,0], w=cc.cartesian.z[:,0], 
                      U=v_unusual[:,0]*u.km/u.s,
                   V=v_unusual[:,1]*u.km/u.s, W=v_unusual[:,2]*u.km/u.s,representation_type=CartesianRepresentation, 
                      differential_type=CartesianDifferential)
    gcs = gcs.transform_to(ICRS)
    pm_ra_cosdec ,pm_dec = gcs.proper_motion.value
    population['pm_un_ra_cosdec'], population['pm_un_dec'] = pm_ra_cosdec ,pm_dec

    population['RA'], population['dec'] = np.degrees(RA), np.degrees(Dec)
    return population

