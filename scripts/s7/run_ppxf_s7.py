import matplotlib
matplotlib.use("agg")  # Use silent backend for running on avatar

import sys, os
import multiprocessing
import numpy as np
from numpy.random import RandomState
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
from time import time

from settings import CLEAN, ppxf_output_path, fig_path, Aperture, get_aperture_coords, gals_all
from load_galaxy_spectrum import load_galaxy_spectrum
from ppxftests.run_ppxf import run_ppxf, add_stuff_to_df

import matplotlib.pyplot as plt
plt.ion()   
plt.close("all")

from IPython.core.debugger import Tracer 

###########################################################################
# User options
###########################################################################
args = sys.argv
if args[1].upper() == "DEBUG":
    debug = True  # If true, run on only 20 threads & save to a new DataFrame with 'DEBUG' in the file name
    args = args[1:]  
else:
    debug = False
nthreads = 5 if debug else 56
niters = 10 if debug else 1000

# x and y coordinates in DataCube corresponding to the chosen aperture
aperture = Aperture[args[1]]
rr, cc = get_aperture_coords(aperture)

# List of Galaxies
if len(args) > 2:
    gals = args[2:]
else:
    gals = gals_all

###########################################################################
# Helper function for running MC simulations
###########################################################################
def ppxf_helper(args):
    # Unpack arguments
    seed, spec, spec_err, lambda_vals_rest_A, bad_pixel_ranges_A = args
    
    # Add "extra" noise to the spectrum
    rng = RandomState(seed)
    noise = rng.normal(scale=spec_err)
    spec_noise = spec + noise

    # This is to mitigate the "edge effects" of the convolution with the LSF
    spec_noise[0] = -9999
    spec_noise[-1] = -9999

    # Make sure there are no NaNs in the input!
    nan_mask = np.isnan(spec_noise)
    nan_mask |= np.isnan(spec_err)
    spec_noise[nan_mask] = -9999
    spec_err[nan_mask] = -9999

    # Run ppxf
    pp = run_ppxf(spec=spec_noise, spec_err=spec_err, lambda_vals_A=lambda_vals_rest_A,
                  isochrones="Padova",
                  bad_pixel_ranges_A=bad_pixel_ranges_A,
                  z=0.0, ngascomponents=1,
                  fit_gas=False, tie_balmer=False,
                  fit_agn_cont=True,
                  clean=CLEAN,
                  reddening=1.0, mdegree=-1,
                  regularisation_method="none")
    return pp

##############################################################################
# Run ppxf for each galaxy & append to DataFrame
for gal in gals:
    # Create a new DataFrame for this galaxy
    df_fname = f"s7_ppxf_DEBUG_{gal}_{aperture.name}.hd5" if debug else f"s7_ppxf_{gal}_{aperture.name}.hd5"
    df = pd.DataFrame(dtype="object")
    if not debug:
        print(f"Now running on {gal} (aperture {aperture})...")
    else:
        print(f"Now running on {gal} (aperture {aperture}) in DEBUG mode...")

    #############################################################################
    # Load the aperture spectrum
    lambda_vals_obs_A, lambda_vals_rest_A, spec, spec_cont_only, spec_err, norm, bad_pixel_ranges_A =\
        load_galaxy_spectrum(gal, aperture)

    ##############################################################################
    # PPXF: MC simulations
    # Input arguments
    seeds = list(np.random.randint(low=0, high=100 * niters, size=niters))
    args_list = [[s, spec_cont_only * norm, spec_err * norm, lambda_vals_rest_A, bad_pixel_ranges_A] for s in seeds]

    # Run in parallel
    print(f"Gal {gal}: MC simulations: running ppxf on {nthreads} threads...")
    t = time()
    with multiprocessing.Pool(nthreads) as pool:
        pp_mc_list = list(tqdm(pool.imap(ppxf_helper, args_list), total=niters))
    print(f"Gal {gal}: MC simulations: total time in ppxf: {time() - t:.2f} s")

    ##############################################################################
    # Extract information for DataFrame
    plot_fname = f"MC_iter_{gal}_{aperture.name}_debug.pdf" if debug else f"MC_iter_{gal}_{aperture.name}.pdf"
    try:
        thisrow = add_stuff_to_df(pp_mc_list, plotit=True, fig_path=fig_path, plot_fname=plot_fname, savefig=True, gal=gal)
    except:
        print(f"WARNING: plotting failed for {gal}!")
        thisrow = add_stuff_to_df(pp_mc_list)
    thisrow["Galaxy"] = gal

    df = df.append(thisrow, ignore_index=True)

    # Save 
    df.to_hdf(os.path.join(ppxf_output_path, df_fname), key=aperture.name)
    print(f"Finished processing {gal}!")
