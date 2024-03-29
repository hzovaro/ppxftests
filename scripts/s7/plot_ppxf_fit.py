import sys, os 

import multiprocessing
import numpy as np
from numpy.random import RandomState
from tqdm import tqdm

from ppxftests.run_ppxf import run_ppxf
from ppxftests.ppxf_plot import ppxf_plot, plot_sfh_light_weighted

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.ion()   
plt.close("all")

from settings import CLEAN, fig_path, Aperture, gals_all, ages, metallicities, get_aperture_label
from load_galaxy_spectrum import load_galaxy_spectrum

###########################################################################
# Helper function for running MC simulations
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
                  plotit=False,
                  regularisation_method="none")
    return pp

###########################################################################
# User options
savefigs = True

# List of Galaxies
if len(sys.argv) > 2:
    gals = sys.argv[2:]
else:
    gals = gals_all

aperture = Aperture[sys.argv[1]]

###########################################################################
# Run ppxf and plot 
for gal in tqdm(gals):

    # Load spectrum with additional wavelength regions masked
    lambda_vals_obs_A, lambda_vals_rest_A, spec, spec_cont_only, spec_err, norm, bad_pixel_ranges_masked_A =\
        load_galaxy_spectrum(gal, aperture, plotit=False)

    # Run ppxf niters times w/ random noise    
    seeds = list(np.random.randint(low=0, high=100 * 1, size=1))
    pp = ppxf_helper([seeds[0], spec_cont_only * norm, spec_err * norm, lambda_vals_rest_A, bad_pixel_ranges_masked_A])

    # Plot the fit 
    fig, ax = plt.subplots(nrows=1, figsize=(10, 3.333))
    ppxf_plot(pp, ax=ax)
    ax.set_title(f"{gal} - {get_aperture_label(aperture)}")
    if savefigs:
        plt.gcf().savefig(os.path.join(fig_path, f"{gal}_MC_fit_example_{aperture.name}.pdf"), format="pdf", bbox_inches="tight")

    # Plot the SFH
    fig, ax = plt.subplots(nrows=1, figsize=(10, 2.5))
    plot_sfh_light_weighted(pp.weights_light_weighted, ages, metallicities, ax=ax)
    if savefigs:
        plt.gcf().savefig(os.path.join(fig_path, f"{gal}_MC_sfh_example_{aperture.name}.pdf"), format="pdf", bbox_inches="tight")
