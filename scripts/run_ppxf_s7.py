import sys, os 

import numpy as np
from numpy.random import RandomState

from astropy.io import fits
from time import time

import multiprocessing

from ppxftests.run_ppxf import run_ppxf

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer 

"""
Running ppxf on the S7 galaxies 

Run on galaxies with 0 reddening (as from the data published by ADT et al.)
Does ppxf think there is nonzero reddening and/or an AGN continuum?
"""

##############################################################################
# User options
##############################################################################
isochrones = "Padova"
fit_agn_cont = False
fit_mpoly = False 
fit_A_V = True 
aperture_type = "Re"
grating ="COMB"
nthreads = 20

obj_name = sys.argv[1]

##############################################################################
assert not (fit_mpoly and fit_A_V),\
    "Only one of 'fit_mpoly' and 'fit_A_V' can be True!"

if fit_mpoly:
    reddening = None
    mdegree = 4 
else:
    reddening = 1.
    mdegree = -1

##############################################################################
# Paths and filenames
##############################################################################
assert "S7_DIR" in os.environ, 'S7_DIR environment variable is not defined! Make sure it is defined in your .bashrc file: export S7_DIR="/path/to/s7/data/"'
data_dir = os.environ["S7_DIR"]
if aperture_type == "Re":
    input_path = os.path.join(data_dir, "4_Nuclear_spectra_Re")
else:
    input_path = os.path.join(data_dir, "5_Full_field_spectra")
input_fits_fname = f"{obj_name}_{grating}_{aperture_type}.fits"

for path in [input_path]:
    assert os.path.exists(path), f"Directory {path} does not exist!"

##############################################################################
# Load the data 
##############################################################################
hdulist = fits.open(os.path.join(input_path, input_fits_fname))
spec = hdulist[0].data 
spec_err = hdulist[1].data

z = hdulist[0].header["Z"]

# Wavelength information
N_lambda = hdulist[0].header["NAXIS1"]
dlambda_A = hdulist[0].header["CDELT1"]
lambda_0_A = hdulist[0].header["CRVAL1"]
lambda_vals_A = np.array(range(N_lambda)) * dlambda_A + lambda_0_A

# Plot the spectrum to check that it looks OK
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
ax.errorbar(x=lambda_vals_A, y=spec, yerr=spec_err, color="k")
ax.set_xlabel("Wavelength (Å)")
ax.set_ylabel(r"Flux $F_\lambda$")

###########################################################################
# Run ppxf - regularisation
###########################################################################
t = time()
print(f"Gal {obj_name}: Regularisation: running ppxf on {nthreads} threads...")
pp_regul = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                    isochrones=isochrones,
                    z=z, ngascomponents=3,
                    fit_gas=True, tie_balmer=True,
                    fit_agn_cont=fit_agn_cont,
                    reddening=reddening, mdegree=mdegree,
                    regularisation_method="auto",
                    regul_nthreads=nthreads,
                    plotit=True)
print(f"Gal {obj_name}: Regularisation: total time in run_ppxf: {time() - t:.2f} seconds")

##############################################################################
# Run ppxf - MC
##############################################################################
def ppxf_helper(args):
    # Unpack arguments
    seed, spec, spec_err, lambda_vals_A, fit_agn_cont = args
    
    # Add "extra" noise to the spectrum
    rng = RandomState(seed)
    noise = rng.normal(scale=spec_err)
    spec_noise = spec + noise

    # Run ppxf
    pp = run_ppxf(spec=spec_noise, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                  isochrones=isochrones,
                  z=z, ngascomponents=2,
                  fit_gas=True, tie_balmer=True,
                  fit_agn_cont=fit_agn_cont,
                  reddening=reddening, mdegree=mdegree,
                  regularisation_method="none")
    return pp

# Input arguments
seeds = list(np.random.randint(low=0, high=100 * niters, size=niters))
args_list = [[s, spec, spec_err, lambda_vals_A, fit_agn_cont] for s in seeds]

# Run in parallel
print(f"Gal {obj_name}: MC simulations: running ppxf on {nthreads} threads...")
t = time()
with multiprocessing.Pool(nthreads) as pool:
    pp_mc_list = list(tqdm(pool.imap(ppxf_helper, args_list), total=niters))
print(f"Gal {obj_name}: MC simulations: total time in ppxf: {time() - t:.2f} s")


###########################################################################
# Look at the results
###########################################################################
# Plot the best-fit SFHs from each run 
