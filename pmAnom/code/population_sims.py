import numpy as np
import LSmetric_population
import sys,os, glob, time, warnings, pickle
import rubin_sim.maf.slicers as slicers
from collections import defaultdict
from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.coordinates import (CartesianRepresentation,
                                 CartesianDifferential, Galactic,ICRS,FK5)
from astropy.io import ascii
import itertools
import healpy as hp



def RADec2pix(nside, ra, dec, degree=True):
    """
    Calculate the nearest healpixel ID of an RA/Dec array, assuming nside.

    Parameters
    ----------
    nside : int
        The nside value of the healpix grid.
    ra : numpy.ndarray
        The RA values to be converted to healpix ids, in degree by default.
    dec : numpy.ndarray
        The Dec values to be converted to healpix ids, in degree by default.

    Returns
    -------
    numpy.ndarray
        The healpix ids.
    """
    if degree:
        ra = np.radians(ra) # change to radians
        dec = np.radians(dec)
    
    lat = np.pi/2. - dec
    hpid = hp.ang2pix(nside, lat, ra )
    return hpid

def sims_pm(slicers, nside, templatefile):
    with open('templatefile.pkl', 'rb') as gaiadata:
        gaia = pickle.load(gaiadata)
    galactic_x = gaia['x'].to_numpy()*1000  # Replace with actual coordinates
    galactic_y = gaia['y'].to_numpy()*1000
    galactic_z = gaia['z'].to_numpy()*1000
    galactocentric_coords = CartesianRepresentation(x=galactic_x*u.pc, y=galactic_y*u.pc, z=galactic_z*u.pc)

    # Step 3: Create SkyCoord object with Galactocentric coordinates
    galactocentric_skycoord = SkyCoord(galactocentric_coords, frame=Galactic)

    # Step 4: Convert to Earth-centric coordinates
    earthcentric_coords = galactocentric_skycoord.transform_to(ICRS)

    # Step 5: Access RA and Dec values
    gaia_ra = earthcentric_coords.ra
    gaia_dec = earthcentric_coords.dec
    gaia_pix = RADec2pix(nside,gaia_ra.value,gaia_dec.value)
    gaia_distance = earthcentric_coords.distance.value
    population = {}
    population['ra']= []
    population['dec']= []
    population['pm_ra_cosdec'] = []
    population['pm_dec']= []
    population['pm_un_ra_cosdec'] = []
    population['pm_un_dec']= []
    population['mag'] = []
    population['galcomp'] = []
    for i in np.unique(gaia_pix):
        sample_pix = np.where(gaia_pix == i)
        population['ra'].append(np.random.choice(gaia_ra[sample_pix].value,5000))
        population['dec'].append(np.random.choice(gaia_dec[sample_pix].value,5000))
        x,y,z = galactic_x,galactic_y,galactic_z
        gcs = Galactic(u=x[sample_pix]*u.pc, v=y[sample_pix]*u.pc, w=z[sample_pix]*u.pc, 
                      U=gaia['vx'].to_numpy()[sample_pix]*u.km/u.s,V=gaia['vy'].to_numpy()[sample_pix]*u.km/u.s, W=gaia['vz'].to_numpy()[sample_pix]*u.km/u.s,
                      representation_type=CartesianRepresentation, 
                      differential_type=CartesianDifferential)
        gcs = gcs.transform_to(ICRS)
        pm_ra_cosdec, pm_dec = gcs.proper_motion.value

        v_unusual = np.random.uniform(-500, 500 ,size=(5000,3))
        dH, dbin = np.histogram(gaia_distance[sample_pix])
        dprobabilities = dH / np.sum(dH)
        d_random_index = np.random.choice(range(len(dprobabilities)), size=5000, p=dprobabilities)
        d_sample = dbin[d_random_index]
        catalog_target = SkyCoord(ra=np.mean(gaia_ra[sample_pix].value)*u.degree, 
                                  dec=np.mean(gaia_dec[sample_pix].value)*u.degree, distance=d_sample*u.pc)
        cc = catalog_target.transform_to(ICRS)
        x_unusual = cc.cartesian.x.value/1000
        y_unusual = cc.cartesian.y.value/1000
        z_unusual = cc.cartesian.z.value/1000
        gcs = Galactic(u=x_unusual*u.kpc, v=y_unusual*u.kpc, w=z_unusual*u.kpc, 
                    U=v_unusual[:,0]*u.km/u.s,V=v_unusual[:,1]*u.km/u.s, W=v_unusual[:,2]*u.km/u.s,
                    representation_type=CartesianRepresentation, differential_type=CartesianDifferential)
        gcs = gcs.transform_to(ICRS)
        pm_un_ra_cosdec ,pm_un_dec = gcs.proper_motion.value
        population['pm_un_ra_cosdec'].append(pm_un_ra_cosdec) 
        population['pm_un_dec'].append(pm_un_dec)


        H, pm_ra_cosdec_bins, pm_dec_bins = np.histogram2d(pm_ra_cosdec, pm_dec)
        probabilities = H / np.sum(H)

        # Reshape the arrays for numpy.random.choice()
        pm_ra_cosdec_bins_values_1d = pm_ra_cosdec_bins.reshape(-1)  # Flatten x-values to 1D array
        pm_dec_bins_values_1d = pm_dec_bins.reshape(-1)  # Flatten y-values to 1D array
        probabilities_1d = probabilities.reshape(-1)  # Flatten probabilities to 1D array

        # Randomly sample from the 2D histogram using numpy.random.choice()
        random_index = np.random.choice(range(len(probabilities_1d)), size=5000, p=probabilities_1d)
        random_x_index, random_y_index = np.unravel_index(random_index, probabilities.shape)
        random_x_value = pm_ra_cosdec_bins_values_1d[random_x_index]
        random_y_value = pm_dec_bins_values_1d[random_y_index]
        population['pm_ra_cosdec'].append(pm_ra_cosdec_bins_values_1d[random_x_index])
        population['pm_dec'].append(pm_dec_bins_values_1d[random_y_index])
        Bulge_magdist = np.random.choice(ascii.read('./Bulge.isc_sloan')['col6'].data, size=5000)
        Disk_magdist = np.random.choice(ascii.read('./Disk.isc_sloan')['col6'].data, size=5000)
        Halo_magdist = np.random.choice(ascii.read('./Halo.isc_sloan')['col6'].data, size=5000)
        total_sample_m = np.empty(5000)
        total_sample_m[distance/8200 < 0.3]=Bulge_magdist[distance/8200 < 0.3]
        total_sample_m[(distance/8200 > 0.3) & (z_unusual/8.2<0.3)]=Disk_magdist[(distance/8200 > 0.3) & (z_unusual/8.2<0.3)]
        total_sample_m[(distance/8200 > 0.3) & (z_unusual/8.2>0.3)]=Halo_magdist[(distance/8200 > 0.3) & (z_unusual/8.2>0.3)]
        population['mag'].append(total_sample_m+5*np.log10(distance)+5)
        comp = np.empty(5000,dtype=str)
        comp[distance/8200 < 0.3] = ['B']*np.sum(distance/8200 < 0.3)
        comp[(distance/8200 > 0.3) & (z_unusual/8.2<0.3)] = ['D']*np.sum((distance/8200 > 0.3) & (z_unusual/8.2<0.3))
        comp[(distance/8200 > 0.3) & (z_unusual/8.2>0.3)] = ['H']*np.sum((distance/8200 > 0.3) & (z_unusual/8.2>0.3))
        population['galcomp'].append(comp)
    return population
