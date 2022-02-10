import os, sys
import numpy as np
from time import time 
from tqdm import tqdm
from itertools import product
import multiprocessing
import pandas as pd
import extinction

from astropy.io import fits

from ppxftests.run_ppxf import run_ppxf
from ppxftests.ssputils import load_ssp_templates
from ppxftests.mockspec import create_mock_spectrum
from ppxftests.sfhutils import load_sfh, compute_mw_age, compute_lw_age, compute_cumulative_mass, compute_cumulative_light
from ppxftests.sfhutils import compute_mean_age, compute_mean_mass, compute_mean_sfr, compute_mean_1D_sfh
from ppxftests.ppxf_plot import plot_sfh_mass_weighted, plot_sfh_light_weighted

# import matplotlib
# matplotlib.use("agg")

from IPython.core.debugger import Tracer

"""

An interactive script for figuring out how & when the weird "jumps" in 
delta-delta-chi2 occur. 
It seems to occur when tie_balmer=True, but not always.

"""

regul_nthreads = 25
isochrones = "Padova"
gal = 10
SNR = 100
z = 0.0
R_V = 4.05

sfh_mw_input, sfh_lw_input, sfr_avg_input, sigma_star_kms = load_sfh(gal, plotit=False)
        
##############################################################################
# TEST 1: A_V = 0
# Do the weights sum to 1.0? 
##############################################################################
# Create spectrum
spec, spec_err, lambda_vals_A = create_mock_spectrum(
    sfh_mass_weighted=sfh_mw_input,
    agn_continuum=True, alpha_nu=1.0, x_AGN=0.5,
    ngascomponents=1, sigma_gas_kms=[70], v_gas_kms=[0], eline_model=["HII"], L_Ha_erg_s=[1e41],
    isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
    A_V=0.0, seed=0, plotit=False)

# run ppxf
t = time()
pp = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
            isochrones=isochrones,
            z=z, fit_gas=True, ngascomponents=1,
            fit_agn_cont=True,
            reddening=1.0, mdegree=-1,
            regularisation_method="none",
            regul_nthreads=regul_nthreads,
            plotit=False, interactive_mode=False)
np.nansum(pp.weights[~pp.gas_component])
pp.weights_agn

##############################################################################
# TEST 2: A_V = 1.0
# Do the weights sum to 1.0? 
##############################################################################
# Create spectrum
spec, spec_err, lambda_vals_A = create_mock_spectrum(
    sfh_mass_weighted=sfh_mw_input,
    agn_continuum=True, alpha_nu=1.0, x_AGN=0.5,
    ngascomponents=1, sigma_gas_kms=[70], v_gas_kms=[0], eline_model=["HII"], L_Ha_erg_s=[1e41],
    isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
    A_V=2.0, seed=0, plotit=False)

# run ppxf
t = time()
pp = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
            isochrones=isochrones,
            z=z, fit_gas=True, ngascomponents=1,
            fit_agn_cont=True,
            reddening=1.0, mdegree=-1,
            regularisation_method="none",
            regul_nthreads=regul_nthreads,
            plotit=False, interactive_mode=False)
np.nansum(pp.weights[~pp.gas_component])
pp.weights_agn

A_lambda_ref = extinction.calzetti00(wave=np.array([4020.]), a_v=pp_z.reddening * R_V, r_v=R_V)
ext_factor_theoretical = 10**(-0.4 * A_lambda_ref)

lambda_norm_idx = np.nanargmin(np.abs(pp.lam - 4020))
ext_factor_pp = pp.mpoly[lambda_norm_idx]

# These weights DO sum to 1
np.nansum(pp.weights[~pp.gas_component]) * ext_factor_pp

##############################################################################
# TEST 2: A_V = 1.0, z = 0.05
# Do the weights sum to 1.0? 
##############################################################################
# Create spectrum
z = 0.05
spec, spec_err, lambda_vals_A = create_mock_spectrum(
    sfh_mass_weighted=sfh_mw_input,
    agn_continuum=True, alpha_nu=0.3, x_AGN=1.0,
    ngascomponents=1, sigma_gas_kms=[70], v_gas_kms=[0], eline_model=["HII"], L_Ha_erg_s=[1e41],
    isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
    A_V=4.0, seed=0, plotit=True)

# run ppxf
# Need to de-redshift the spectrum first!!!
pp_z = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A / (1 + z),
            isochrones=isochrones,
            z=0.0, fit_gas=True, ngascomponents=1,
            fit_agn_cont=True,
            reddening=1.0, mdegree=-1,
            regularisation_method="none",
            regul_nthreads=regul_nthreads,
            plotit=True, interactive_mode=False)

lambda_norm_idx = np.nanargmin(np.abs(pp_z.lam - 4020))
ext_factor_pp = pp_z.mpoly[lambda_norm_idx]
print(f"Sum of ALL non-gas component weights: {np.nansum(pp_z.weights[~pp_z.gas_component]):.2f}")
print(f"Sum of ALL non-gas component weights * predicted reddening curve: {np.nansum(pp_z.weights[~pp_z.gas_component]) * ext_factor_pp:.2f}")
print(f"x_AGN = {np.nansum(pp_z.weights_agn) / np.nansum(pp_z.weights_stellar):.2f}")
print(f"A_V = {pp_z.reddening * R_V:.2f}")
print(f"pp.sol = [{pp_z.sol[0][0]:.2f}, {pp_z.sol[0][1]:.2f}]")

# THIS WORKS NOW!!!


# Earlier when I looked at pp.sol, the values looked maxed-out, which was odd.
# Does it have something to do with being redshifted?
# If this is causing issues, then I can always run ppxf on the spectrum
# after it has been de-redshifted, since we know the redshifts of these 
# sources quite well anyway.

# A_lambda_ref = extinction.calzetti00(wave=np.array([4020.]), a_v=pp.reddening * R_V, r_v=R_V)
# ext_factor_theoretical = 10**(-0.4 * A_lambda_ref)

# lambda_norm_idx = np.nanargmin(np.abs(pp.lam - 4020))
# ext_factor_pp = pp.mpoly[lambda_norm_idx]

# # These weights DO NOT sum to 1: why????
# np.nansum(pp.weights[~pp.gas_component]) * ext_factor_pp


##############################################################################

##############################################################################

# Need to figure out how/why the weights are weird 
# Is it because the weights correspond to the spectrum BEFORE the extinction
# is applied?
A_lambda_ref = extinction.calzetti00(wave=np.array([4020]), a_v=pp.reddening * R_V, r_v=R_V)
# ppxf weights SHOULD sum to 1 when multiplied by A_lambda_ref
pp.weights_agn * A_lambda_ref


"""
from time import sleep

import multiprocessing

nthreads = 5

def my_function(a):
    print(a + 5)
    sleep(5)
    return 

print("Hello world!")

with multiprocessing.Pool(nthreads) as pool:
    res_list = list(pool.imap(my_function, [1, 2, 3, 4, 5]))

print("Done!")
"""