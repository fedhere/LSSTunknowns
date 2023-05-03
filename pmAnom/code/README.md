populations_synthesis_pm.py-
The sample_simulation method generates a sample of stars at a given Galactic location, 
where the input arguments x,y,z are the Galactocentric cartesian coordinates,
and d is the distance to the star. The method first computes the distance 
from the Galactic center for each star in the input sample,
then selects the stars in the Gaia dataset whose distances are closest 
to the distance of theinput sample, and finally, 
samples the velocities of these stars to generate the velocity components of the input sample.
The method then computes the magnitude distribution for the stars in the Bulge, Disk, 
and Halo regions,
using BaSTI isochrones, and draws samples from these distributions to generate the magnitudes 
of the input sample. 
The method returns the Cartesian coordinates, velocities, and magnitudes of the input sample.
The method allows to change the tassellation of the Sky using the same slicer used in the MAF. 

To run the metrics we need Gaiachallengedata.pkl check it is in the data folder

LSmetric.py - 
The LSPMmetric class is a custom metric designed to evaluate the efficiency of 
the Large Synoptic Survey Telescope (LSST) in detecting high proper motion stars. 
The __init__ method initializes the instance variables of the class with the input 
values provided to the method.

TPMmetric.py - 
The TPMmetric class is a custom metric designed to evaluate the efficiency of 
the Large Synoptic Survey Telescope (LSST) in detecting photometric variability
of sources with measurable proper motion. 
The __init__ method initializes the instance variables of the class with the input 
values provided to the method.


To run the metrics we need population_nside32.p check it is in the data folder
