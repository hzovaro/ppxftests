import sys, os 

import numpy as np
from numpy.random import RandomState

from astropy.io import fits
from time import time

from ppxftests.sfhutils import compute_mw_age, compute_lw_age

import multiprocessing

from ppxftests.run_ppxf import run_ppxf

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer 


"""
Fit the continuum in two ways:
1. using LZIFU, in order to fit & subtract emission lines 
2. using ppxf w/ regularisation, in order to analyse the SFH
Are the continua fitted via each of these methods similar? 
"""
lzifu_input_path = "/priv/meggs3/u5708159/S7/LZIFU/data/"
lzifu_output_path = "/priv/meggs3/u5708159/S7/LZIFU/products/"
gal = sys.argv[1]

nthreads = 30

##############################################################################
# Load the aperture spectrum
##############################################################################
hdulist_in = fits.open(os.path.join(lzifu_input_path, f"{gal}.fits.gz"))
spec = hdulist_in[0].data[:, 0, 0]
spec_err = np.sqrt(hdulist_in[1].data[:, 0, 0])
norm = hdulist_in[0].header["NORM"]
z = hdulist_in[0].header["Z"]

N_lambda = hdulist_in[0].header["NAXIS3"]
dlambda_A = hdulist_in[0].header["CDELT3"]
lambda_0_A = hdulist_in[0].header["CRVAL3"]
lambda_vals_obs_A = np.array(list(range(N_lambda))) * dlambda_A + lambda_0_A
lambda_vals_rest_A = lambda_vals_obs_A / (1 + z)

##############################################################################
# Load the LZIFU fit
##############################################################################
lzifu_fname = f"{gal}_merge_comp.fits"
hdulist = fits.open(os.path.join(lzifu_output_path, lzifu_fname))
spec_cont_lzifu = hdulist["CONTINUUM"].data[:, 0, 0] 
spec_elines = hdulist["LINE"].data[:, 0, 0]

spec_cont_only = spec - spec_elines

# Plot to check
fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(x=lambda_vals_obs_A, y=spec, yerr=spec_err, color="k", label="Data")
ax.plot(lambda_vals_obs_A, spec_cont_lzifu, color="green", label="Continuum fit")
ax.plot(lambda_vals_obs_A, spec_elines, color="magenta", label="Emission line fit")
ax.plot(lambda_vals_obs_A, spec_cont_lzifu + spec_elines, color="orange", label="Total fit")
ax.plot(lambda_vals_obs_A, spec_cont_only, color="red", label="Data minus emission lines")
ax.set_xlabel("Wavelength (Å)")
ax.set_ylabel(r"Normalised flux ($F_\lambda$)")
ax.legend()

##############################################################################
# Run ppxf with regularisation
##############################################################################
bad_pixel_ranges_A = [
     [(6300 - 10) / (1 + z), (6300 + 10) / (1 + z)], # Sky line at 6300
     [(5577 - 10) / (1 + z), (5577 + 10) / (1 + z)], # Sky line at 5577
     [(6360 - 10) / (1 + z), (6360 + 10) / (1 + z)], # Sky line at 6360
     [(5700 - 10) / (1 + z), (5700 + 10) / (1 + z)], # Sky line at 5700                              
     [(5889 - 30), (5889 + 20)], # Interstellar Na D + He line 
     [(5889 - 10) / (1 + z), (5889 + 10) / (1 + z)], # solar NaD line
]

t = time()
print(f"Gal {gal}: Regularisation: running ppxf on {nthreads} threads...")
pp_regul_nolines = run_ppxf(spec=spec_cont_only * norm, spec_err=spec_err * norm, lambda_vals_A=lambda_vals_rest_A,
                    isochrones="Padova", z=0.0, 
                    fit_gas=False, ngascomponents=0,
                    fit_agn_cont=True,
                    reddening=1.0, mdegree=-1,
                    bad_pixel_ranges_A=bad_pixel_ranges_A,
                    regularisation_method="auto",
                    regul_nthreads=nthreads, interactive_mode=False,
                    plotit=False)
print(f"Gal {gal}: Regularisation: total time in run_ppxf: {time() - t:.2f} seconds")

t = time()
print(f"Gal {gal}: Regularisation: running ppxf on {nthreads} threads...")
pp_regul_lines = run_ppxf(spec=spec * norm, spec_err=spec_err * norm, lambda_vals_A=lambda_vals_rest_A,
                    isochrones="Padova", z=0.0, 
                    fit_gas=True, ngascomponents=3,
                    fit_agn_cont=True,
                    reddening=1.0, mdegree=-1,
                    bad_pixel_ranges_A=bad_pixel_ranges_A,
                    regularisation_method="auto",
                    regul_nthreads=nthreads, interactive_mode=False,
                    plotit=False)
print(f"Gal {gal}: Regularisation: total time in run_ppxf: {time() - t:.2f} seconds")

##############################################################################
# Plot: compare the LZIFU and PPXF continuum fits 
##############################################################################
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(lambda_vals_rest_A, spec, label="Input", color="k")
ax.plot(pp_regul_nolines.lam, pp_regul_nolines.bestfit, label="ppxf fit (no emission lines)")
ax.plot(pp_regul_lines.lam, pp_regul_lines.bestfit, label="ppxf fit (with emission lines)")
ax.plot(lambda_vals_rest_A, spec_cont_lzifu, label="lzifu fit")
ax.legend()
ax.set_xlabel("Wavelength (Å)")
ax.set_ylabel(r"Normalised flux ($F_\lambda$)")

##############################################################################
# Does the prior subtraction of emission lines make any difference to the 
# quantities that we're interested in?
##############################################################################
ages = pp_regul_lines.ages
# Compare the LW/MW SFHs of each
fig, ax = plt.subplots(nrows=1, figsize=(15, 5))
ax.step(ages, pp_regul_lines.sfh_mw_1D, label="ppxf fit (with emission lines)")
ax.step(ages, pp_regul_nolines.sfh_mw_1D, label="ppxf fit (no emission lines)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Age (yr)")
ax.set_ylabel(r"Mass in each bin ($\rm M_\odot$)")
ax.grid()
ax.legend()
ax.set_title(gal)

# Plot the MW/LW age as a function of age cutoff for each fit 
ages_mw_lines = np.array([compute_mw_age(pp_regul_lines.sfh_mw_1D, isochrones="Padova", age_thresh_upper=a)[0] for a in ages[1:]])
ages_mw_nolines = np.array([compute_mw_age(pp_regul_nolines.sfh_mw_1D, isochrones="Padova", age_thresh_upper=a)[0] for a in ages[1:]])
ages_lw_lines = np.array([compute_lw_age(pp_regul_lines.sfh_lw_1D, isochrones="Padova", age_thresh_upper=a)[0] for a in ages[1:]])
ages_lw_nolines = np.array([compute_lw_age(pp_regul_nolines.sfh_lw_1D, isochrones="Padova", age_thresh_upper=a)[0] for a in ages[1:]])

fig, axs = plt.subplots(ncols=2, figsize=(15, 5))
axs[0].step(ages[1:], ages_mw_lines, label="ppxf fit (with emission lines)")
axs[0].step(ages[1:], ages_mw_nolines, label="ppxf fit (no emission lines)")
axs[0].set_xlabel("Cutoff age (yr)")
axs[0].set_ylabel("Mean MW age below cutoff (log yr)")
axs[0].set_xscale("log")
axs[0].grid()
axs[1].step(ages[1:], ages_lw_lines, label="ppxf fit (with emission lines)")
axs[1].step(ages[1:], ages_lw_nolines, label="ppxf fit (no emission lines)")
axs[1].set_xlabel("Cutoff age (yr)")
axs[1].set_ylabel("Mean LW age below cutoff (log yr)")
axs[1].set_xscale("log")
axs[1].grid()
axs[1].legend()
fig.suptitle(gal)

