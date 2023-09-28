"""
PROBLEM: 
In the process of making figures for the paper showing fits to individual galaxies, it's become clear that we have 2 problems:

1) the fit itself is not very good for a handful of galaxies, and
2) the CLEAN keyword is masking out a *lot* of pixels in some galaxies (although by eye this doesn't appear to affect the fit all that much??)

TODO:
1. In the documentation, it says only to use the CLEAN keyword if the reduced-chi2 of the fit is ~1. So, for each galaxy, check pp.chi2. If they are all >1, then it's probably best to repeat everything with CLEAN turned off.
2. How does the chi2 change from MC run to MC run? Is there a lot of variation? Test: run ~20 iterations for say ~3 galaxies & look at the MC plots.

A note on indexing and age cutoffs:
    in compute_lw_age():
        input age_thresh_upper (= tau_cutoff)
        LW ages computed in templates up to but NOT including tau_cutoff.
        so if tau_cutoff = ages[idx] the LW age is computed from the templates from ages[0] to ages[idx - 1]

        so in the array of LW ages:
            idx 0 --> tau_cutoff idx = 1
            idx 1 --> tau_cutoff idx = 2
            ...
            idx N --> tau_cutoff idx = N + 1
            so for tau_cutoff = 100 Myr,
            idx <100 Myr idx - 1> --> tau_cutoff idx = <100 Myr idx>.

TODO: need to repeat all of this for the other apertures, too.

"""
import sys, os 

import multiprocessing
import numpy as np
from numpy.random import RandomState
import pandas as pd
from astropy.io import fits
from time import time
from tqdm import tqdm

from ppxftests.run_ppxf import run_ppxf, add_stuff_to_df, ppxf_plot
from ppxftests.sfhutils import ages_padova as ages
from ppxftests.sfhutils import compute_lw_age, compute_mw_age
isochrones="Padova"

import matplotlib.pyplot as plt
plt.ion()   
plt.close("all")

from IPython.core.debugger import Tracer 

# Paths
lzifu_input_path = "/priv/meggs3/u5708159/S7/mar23/LZIFU/data/"
lzifu_output_path = "/priv/meggs3/u5708159/S7/mar23/LZIFU/products/"
s7_data_path = "/priv/meggs3/u5708159/S7/mar23/"
fig_path = "/priv/meggs3/u5708159/S7/mar23/ppxf/figs/"

# Aperture type
from enum import Enum
class Aperture(Enum):
    RE1 = 0
    FOURAS = 1
    SDSS = 2
    ONEKPC = 3

# Helper function for running MC simulations
def ppxf_helper(args):
    # Unpack arguments
    seed, spec, spec_err, lambda_vals_rest_A, bad_pixel_ranges_A, clean = args
    
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
                  clean=clean,
                  reddening=1.0, mdegree=-1,
                  plotit=False,
                  regularisation_method="none")
    return pp

###########################################################################
# User options
savefigs = True

# List of Galaxies
gals = [g.strip("\n") for g in open(os.path.join(s7_data_path, "gal_list.txt")).readlines()]

# x and y coordinates in DataCube corresponding to the chosen aperture
aperture = Aperture[sys.argv[1]]
rr, cc = np.unravel_index(aperture.value, (2, 2))

# DataFrame storing reduced-chi2 for each galaxy 
df = pd.DataFrame(index=gals, columns=["pp (clean)", "pp (no clean)", "reduced-chi2 (clean)", "reduced-chi2 (no clean)"])

###########################################################################
# Run ppxf and plot 
for gal in tqdm(gals[2:4]):

    # Load the aperture spectrum
    hdulist_in = fits.open(os.path.join(lzifu_input_path, f"{gal}.fits.gz"))
    
    spec = hdulist_in["PRIMARY"].data[:, rr, cc]
    spec_err = np.sqrt(hdulist_in["VARIANCE"].data[:, rr, cc])
    norm = hdulist_in["NORM"].data[rr, cc]
    z = hdulist_in[0].header["Z"]

    N_lambda = hdulist_in[0].header["NAXIS3"]
    dlambda_A = hdulist_in[0].header["CDELT3"]
    lambda_0_A = hdulist_in[0].header["CRVAL3"]
    lambda_vals_obs_A = np.array(list(range(N_lambda))) * dlambda_A + lambda_0_A
    lambda_vals_rest_A = lambda_vals_obs_A / (1 + z)

    # Define bad pixel ranges
    bad_pixel_ranges_A = [
         [(6300 - 10) / (1 + z), (6300 + 10) / (1 + z)], # Sky line at 6300
         [(5577 - 10) / (1 + z), (5577 + 10) / (1 + z)], # Sky line at 5577
         [(6360 - 10) / (1 + z), (6360 + 10) / (1 + z)], # Sky line at 6360
         [(5700 - 10) / (1 + z), (5700 + 10) / (1 + z)], # Sky line at 5700                              
         [(5889 - 30), (5889 + 20)], # Interstellar Na D + He line 
         [(5889 - 10) / (1 + z), (5889 + 10) / (1 + z)], # solar NaD line
    ]

    # Load the LZIFU fit
    lzifu_fname = f"{gal}_merge_comp.fits"
    hdulist = fits.open(os.path.join(lzifu_output_path, lzifu_fname))
    spec_cont_lzifu = hdulist["CONTINUUM"].data[:, rr, cc] 
    spec_elines = hdulist["LINE"].data[:, rr, cc]
    spec_cont_only = spec - spec_elines

    ##############################################################################
    # PPXF: MC simulations
    # Input arguments
    """
    seeds = list(np.random.randint(low=0, high=100 * 1, size=1))
    pp_noclean = ppxf_helper([seeds[0], spec_cont_only * norm, spec_err * norm, lambda_vals_rest_A, bad_pixel_ranges_A, False])
    pp_clean = ppxf_helper([seeds[0], spec_cont_only * norm, spec_err * norm, lambda_vals_rest_A, bad_pixel_ranges_A, True])

    df.loc[gal, "pp (clean)"] = pp_clean
    df.loc[gal, "pp (no clean)"] = pp_noclean
    df.loc[gal, "reduced-chi2 (clean)"] = pp_clean.chi2
    df.loc[gal, "reduced-chi2 (no clean)"] = pp_noclean.chi2
    df.loc[gal, "Number of pixels masekd out with CLEAN"] = len(pp_noclean.goodpixels) - len(pp_clean.goodpixels)
    """

    ##############################################################################
    # Input arguments
    niters = 100
    nthreads = 20
    seeds = list(np.random.randint(low=0, high=100 * niters, size=niters))
    args_list = [[s, spec_cont_only * norm, spec_err * norm, lambda_vals_rest_A, bad_pixel_ranges_A, False] for s in seeds]

    # Run in parallel
    print(f"Gal {gal}: MC simulations: running ppxf on {nthreads} threads...")
    t = time()
    with multiprocessing.Pool(nthreads) as pool:
        pp_mc_list = list(tqdm(pool.imap(ppxf_helper, args_list), total=niters))
    print(f"Gal {gal}: MC simulations: total time in ppxf: {time() - t:.2f} s")
    thisrow = add_stuff_to_df(pp_mc_list, pp_mc_list[0], gal=gal, plotit=True, savefig=True, plot_fname=f"{gal}_MC_check_chi2.pdf")

    ##############################################################################
    # # Plot the actual fit.
    # fig, axs = plt.subplots(nrows=2, figsize=(10, 7.5))
    # ppxf_plot(pp_noclean, ax=axs[0])
    # ppxf_plot(pp_clean, ax=axs[1])
    # axs[0].set_title(gal)
    # # For now, saving to the regular directory, not the paper one
    # if savefigs:
    #     plt.gcf().savefig(os.path.join(fig_path, f"{gal}_clean_vs_noclean_{aperture.name}.pdf"), format="pdf", bbox_inches="tight")


sys.exit()

"""
Gals w/ bad fits:
IC4329A
FAIRALL49
NGC7469
NGC1667*
MCG-03-34-064
NGC6860

Gals w/ lots of pixels masked out:
IC4329A
NGC1068
NGC613
ESO362-G018
NGC1320
NGC7469
NGC1667*
MCG-03-34-064
NGC6860

Gals w/ bad eline residuals:
IC4329A
NGC1068
NGC7469
NGC424
MCG-03-34-064

"""

# TODO: histogram showing % of pixels masked out when CLEAN is used.
# TODO: look at the chi2 of the fits: perhaps save these in the MC output?

# Compute stuff. Note: pass pp_mc_list[0] instead of pp_regul since we're not bothering with it anymore.
thisrow = add_stuff_to_df([pp_mc], pp_mc, plotit=False, gal=gal)

# Plot distributions in various quantities for each iteration
fig_hist, axs_hist = plt.subplots(nrows=4, ncols=2, figsize=(12, 20))
fig_hist.subplots_adjust(hspace=0.3, left=0.1, right=0.9, top=0.95, bottom=0.05)
ax_iter = 0

# Plot the age distributions overlaid with quantiles to see what they look like & check that everything looks right 
for age_idx in [28, 48, len(ages) - 2]:
    mw_ages = [compute_mw_age(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=ages[age_idx + 1])[0] for pp in pp_mc_list]
    if not all(np.isnan(mw_ages)):
        axs_hist.flat[ax_iter].hist(mw_ages, bins=15)
    axs_hist.flat[ax_iter].axvline(thisrow["Mass-weighted age vs. age cutoff (MC 16th percentile)"][age_idx], ls="--", color="grey")
    axs_hist.flat[ax_iter].axvline(thisrow["Mass-weighted age vs. age cutoff (MC 50th percentile)"][age_idx], color="grey")
    axs_hist.flat[ax_iter].axvline(thisrow["Mass-weighted age vs. age cutoff (MC 84th percentile)"][age_idx], ls="--", color="grey")
    axs_hist.flat[ax_iter].set_title(f"Mass-weighted age below {ages[age_idx + 1] / 1e9:.2f} Gyr")
    axs_hist.flat[ax_iter].set_xlabel("Mass-weighted age (log yr)")
    ax_iter += 1

    lw_ages = [compute_lw_age(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=ages[age_idx + 1])[0] for pp in pp_mc_list]
    if not all(np.isnan(lw_ages)):
        axs_hist.flat[ax_iter].hist(lw_ages, bins=15)
    axs_hist.flat[ax_iter].axvline(thisrow["Light-weighted age vs. age cutoff (MC 16th percentile)"][age_idx], ls="--", color="grey")
    axs_hist.flat[ax_iter].axvline(thisrow["Light-weighted age vs. age cutoff (MC 50th percentile)"][age_idx], color="grey")
    axs_hist.flat[ax_iter].axvline(thisrow["Light-weighted age vs. age cutoff (MC 84th percentile)"][age_idx], ls="--", color="grey")
    axs_hist.flat[ax_iter].set_title(f"Light-weighted age below {ages[age_idx + 1] / 1e9:.2f} Gyr")
    axs_hist.flat[ax_iter].set_xlabel("Light-weighted age (log yr)")
    ax_iter += 1

# Extinction and/or polynomial fits
R_V = 4.05
a_v_vals = R_V * np.array([pp.reddening for pp in pp_mc_list])
if not all(np.isnan(a_v_vals)):
    axs_hist.flat[ax_iter].hist(a_v_vals, bins=15)
axs_hist.flat[ax_iter].axvline(thisrow["A_V (MC 16th percentile)"], ls="--", color="grey")
axs_hist.flat[ax_iter].axvline(thisrow["A_V (MC 50th percentile)"], color="grey")
axs_hist.flat[ax_iter].axvline(thisrow["A_V (MC 84th percentile)"], ls="--", color="grey")
axs_hist.flat[ax_iter].set_title(r"$A_V$")
axs_hist.flat[ax_iter].set_xlabel(r"$A_V$ (mag)")
ax_iter += 1

# Plot x_AGN distribution overlaid w/ quantiles etc. to check that it looks right
x_AGN_vals = np.array([np.nansum(pp.weights_agn) / np.nansum(pp.weights_stellar) for pp in pp_mc_list])
if not all(np.isnan(x_AGN_vals)):
    axs_hist.flat[ax_iter].hist(x_AGN_vals, bins=15)
axs_hist.flat[ax_iter].axvline(thisrow["x_AGN (total, MC 16th percentile)"], ls="--", color="grey")
axs_hist.flat[ax_iter].axvline(thisrow["x_AGN (total, MC 50th percentile)"], color="grey")
axs_hist.flat[ax_iter].axvline(thisrow["x_AGN (total, MC 84th percentile)"], ls="--", color="grey")
axs_hist.flat[ax_iter].axvline(thisrow["x_AGN (total, MC mean)"], color="k")
axs_hist.flat[ax_iter].axvline(thisrow["x_AGN (total, regularised)"], color="red")
axs_hist.flat[ax_iter].axvspan(xmin=thisrow["x_AGN (total, MC mean)"] - thisrow["x_AGN (total, MC error)"], 
            xmax=thisrow["x_AGN (total, MC mean)"] + thisrow["x_AGN (total, MC error)"], color="pink", alpha=0.3)
axs_hist.flat[ax_iter].set_title(r"$x_{\rm AGN}$")
axs_hist.flat[ax_iter].set_xlabel(r"$x_{\rm AGN}$")
ax_iter += 1

# if savefig:
    # fig_hist.savefig(os.path.join("/priv/meggs3/u5708159/S7/mar23/ppxf/figs/", plot_fname), bbox_inches="tight", format="pdf")

