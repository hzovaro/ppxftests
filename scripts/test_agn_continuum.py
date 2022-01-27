import os, sys
import numpy as np
from numpy.random import RandomState
from time import time 
from tqdm import tqdm
from itertools import product
import multiprocessing
import pandas as pd

from astropy.io import fits

from ppxftests.run_ppxf import run_ppxf
from ppxftests.ssputils import load_ssp_templates, get_bin_edges_and_widths
from ppxftests.mockspec import create_mock_spectrum
from ppxftests.sfhutils import load_sfh, compute_mw_age, compute_lw_age, compute_mass
from ppxftests.sfhutils import compute_mean_age, compute_mean_mass, compute_mean_sfr, compute_mean_1D_sfh
from ppxftests.ppxf_plot import plot_sfh_mass_weighted, plot_sfh_light_weighted

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

fig_path = "/priv/meggs3/u5708159/ppxftests/figs/"
data_path = "/priv/meggs3/u5708159/ppxftests/"

###############################################################################
# Settings
###############################################################################
isochrones = "Padova"
SNR = 100
z = 0.01

# Get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates(isochrones)
N_ages = len(ages)
N_metallicities = len(metallicities)
bin_edges, bin_widths = get_bin_edges_and_widths(isochrones=isochrones)

# ppxf settings
niters = 100
nthreads = 25

# For analysis
lambda_norm_A = 5000
age_thresh_pairs = [
    (ages[0], 1e7),
    (ages[0], 1e8),
    (ages[0], 1e9),
    (1e9, ages[-1]),
    (ages[0], ages[-1]),
]

age_thresh_vals = [None, 1e9, None]

# Parameters
# alpha_nu_vals = np.linspace(0.3, 2.1, 5)  # What is a typical value for low-z Seyfert galaxies?
# log_L_NT_vals = np.linspace(42, 44, 5)
alpha_nu_vals = [0.3, 2.0]
log_L_NT_vals = [42, 43, 44]

###########################################################################
# Helper function for running MC simulations
###########################################################################
def ppxf_helper(args):
    # Unpack arguments
    seed, spec, spec_err, lambda_vals_A = args
    
    # Add "extra" noise to the spectrum
    rng = RandomState(seed)
    noise = rng.normal(scale=spec_err)
    spec_noise = spec + noise

    # This is to mitigate the "edge effects" of the convolution with the LSF
    spec_noise[0] = -9999
    spec_noise[-1] = -9999

    # Run ppxf
    pp = run_ppxf(spec=spec_noise, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                  z=z, ngascomponents=1,
                  regularisation_method="none", 
                  isochrones="Padova",
                  fit_gas=False, tie_balmer=True,
                  plotit=False, savefigs=False, interactive_mode=False)
    return pp

###############################################################################
# Load a realistic SFH
###############################################################################
gals = [int(g) for g in sys.argv[1:]]
for gal in gals:
    try:
        sfh_mw_input, sfh_lw_input, sfr_avg_input, sigma_star_kms = load_sfh(gal, plotit=True)
    except:
        print(f"ERROR: unable to load galaxy {gal}. Skipping...")
        continue

    ###############################################################################
    # Run ppxf without an AGN continuum added as a "control"
    ###############################################################################
    # Create spectrum
    spec, spec_err, lambda_vals_A = create_mock_spectrum(
        sfh_mass_weighted=sfh_mw_input,
        agn_continuum=False,
        isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
        plotit=False)

    ###########################################################################
    # Run ppxf WITHOUT regularisation, using a MC approach
    ###########################################################################
    # Input arguments
    seeds = list(np.random.randint(low=0, high=100 * niters, size=niters))
    args_list = [[s, spec, spec_err, lambda_vals_A] for s in seeds]

    # Run in parallel
    print(f"Gal {gal:004}: MC simulations: running ppxf on {nthreads} threads...")
    t = time()
    with multiprocessing.Pool(nthreads) as pool:
        pp_mc_list = list(tqdm(pool.imap(ppxf_helper, args_list), total=niters))
    print(f"Gal {gal:004}: MC simulations: total time in ppxf: {time() - t:.2f} s")

    ###########################################################################
    # Run ppxf with regularisation
    ###########################################################################
    t = time()
    print(f"Gal {gal:004}: Regularisation: running ppxf on {nthreads} threads...")
    pp_regul = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                  z=z, ngascomponents=1,
                  regularisation_method="auto",
                  isochrones=isochrones,
                  fit_gas=False, tie_balmer=True,
                  delta_regul_min=1, regul_max=5e4, delta_delta_chi2_min=1,
                  regul_nthreads=nthreads,
                  plotit=False, savefigs=False, interactive_mode=False)
    print(f"Gal {gal:004}: Regularisation: total time in run_ppxf: {time() - t:.2f} seconds")

    ###########################################################################
    # Create an empty DataFrame for storing results in
    ###########################################################################
    df = pd.DataFrame()

    ###########################################################################
    # Compute mean quantities from the ppxf runs
    ###########################################################################
    thisrow = {}  # Row to append to DataFrame
    thisrow["AGN continuum"] = False
    thisrow["alpha_nu"] = np.nan
    thisrow["log L_NT"] = np.nan

    # Compute the mean SFH and SFR from the lists of MC runs
    sfh_MC_lw_1D_mean = compute_mean_1D_sfh(pp_mc_list, isochrones, "lw")
    sfh_MC_mw_1D_mean = compute_mean_1D_sfh(pp_mc_list, isochrones, "mw")
    sfr_avg_MC = compute_mean_sfr(pp_mc_list, isochrones)
    sfh_regul_mw_1D = pp_regul.sfh_mw_1D
    sfh_regul_lw_1D = pp_regul.sfh_lw_1D
    sfr_avg_regul = pp_regul.sfr_mean

    # Compute the mean mass- and light-weighted ages plus the total mass in a series of age ranges
    for age_thresh_pair in age_thresh_pairs:
        age_thresh_lower, age_thresh_upper = age_thresh_pair
        age_str = f"{np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f}"
            
        # MC runs: compute the mean mass- and light-weighted ages plus the total mass in this age range
        age_lw_mean, age_lw_std = compute_mean_age(pp_mc_list, isochrones, "lw", age_thresh_lower, age_thresh_upper)
        age_mw_mean, age_mw_std = compute_mean_age(pp_mc_list, isochrones, "mw", age_thresh_lower, age_thresh_upper)
        mass_mean, mass_std = compute_mean_mass(pp_mc_list, isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)
        
        # Regul run: compute the mean mass- and light-weighted ages plus the total mass in this age range
        age_mw_regul = 10**compute_mw_age(sfh_regul_mw_1D, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)[0]
        age_lw_regul = 10**compute_lw_age(sfh_regul_lw_1D, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)[0]
        mass_regul = compute_mass(sfh_regul_mw_1D, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)

        # Input: compute the mean mass- and light-weighted ages plus the total mass in this age range
        age_mw_input = 10**compute_mw_age(sfh_mw_input, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)[0]
        age_lw_input = 10**compute_lw_age(sfh_lw_input, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)[0]
        mass_input = compute_mass(sfh_mw_input, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)

        # Put in DataFrame
        thisrow[f"MW age {age_str} (input)"] = age_mw_input
        thisrow[f"LW age {age_str} (input)"] = age_lw_input
        thisrow[f"Mass {age_str} (input)"] = mass_input
        thisrow[f"MW age {age_str} (MC) mean"] = age_mw_mean
        thisrow[f"LW age {age_str} (MC) mean"] = age_lw_mean
        thisrow[f"Mass {age_str} (MC) mean"] = mass_mean
        thisrow[f"MW age {age_str} (MC) std. dev."] = age_mw_std
        thisrow[f"LW age {age_str} (MC) std. dev."] = age_lw_std
        thisrow[f"Mass {age_str} (MC) std. dev."] = mass_std
        thisrow[f"MW age {age_str} (regularised)"] = age_mw_regul
        thisrow[f"LW age {age_str} (regularised)"] = age_lw_regul
        thisrow[f"Mass {age_str} (regularised)"] = mass_regul

    df = df.append(thisrow, ignore_index=True)

    ###############################################################################
    # The effect of the strength and exponent of the AGN continuum on the recovered SFH
    ###############################################################################
    for alpha_nu, log_L_NT in tqdm(product(alpha_nu_vals, log_L_NT_vals), total=len(alpha_nu_vals) * len(log_L_NT_vals)):
        # Create spectrum
        spec, spec_err, lambda_vals_A = create_mock_spectrum(
            sfh_mass_weighted=sfh_mw_input,
            agn_continuum=True, L_NT_erg_s=10**log_L_NT, alpha_nu=alpha_nu,
            isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
            plotit=False)

        ###########################################################################
        # Run ppxf WITHOUT regularisation, using a MC approach
        ###########################################################################
        # Input arguments
        seeds = list(np.random.randint(low=0, high=100 * niters, size=niters))
        args_list = [[s, spec, spec_err, lambda_vals_A] for s in seeds]

        # Run in parallel
        print(f"Gal {gal:004}: MC simulations: running ppxf on {nthreads} threads...")
        t = time()
        with multiprocessing.Pool(nthreads) as pool:
            pp_mc_list = list(tqdm(pool.imap(ppxf_helper, args_list), total=niters))
        print(f"Gal {gal:004}: MC simulations: total time in ppxf: {time() - t:.2f} s")

        ###########################################################################
        # Run ppxf with regularisation
        ###########################################################################
        t = time()
        print(f"Gal {gal:004}: Regularisation: running ppxf on {nthreads} threads...")
        pp_regul = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                      z=z, ngascomponents=1,
                      regularisation_method="auto",
                      isochrones=isochrones,
                      fit_gas=False, tie_balmer=True,
                      delta_regul_min=1, regul_max=5e4, delta_delta_chi2_min=1,
                      regul_nthreads=nthreads,
                      plotit=False, savefigs=False, interactive_mode=False)
        print(f"Gal {gal:004}: Regularisation: total time in run_ppxf: {time() - t:.2f} seconds")

        ###########################################################################
        # Compute mean quantities from the ppxf runs
        ###########################################################################
        thisrow = {}  # Row to append to DataFrame
        thisrow["AGN continuum"] = True
        thisrow["alpha_nu"] = alpha_nu
        thisrow["log L_NT"] = log_L_NT

        # Compute the mean SFH and SFR from the lists of MC runs
        sfh_MC_lw_1D_mean = compute_mean_1D_sfh(pp_mc_list, isochrones, "lw")
        sfh_MC_mw_1D_mean = compute_mean_1D_sfh(pp_mc_list, isochrones, "mw")
        sfr_avg_MC = compute_mean_sfr(pp_mc_list, isochrones)
        sfh_regul_mw_1D = pp_regul.sfh_mw_1D
        sfh_regul_lw_1D = pp_regul.sfh_lw_1D
        sfr_avg_regul = pp_regul.sfr_mean

        # Compute the mean mass- and light-weighted ages plus the total mass in a series of age ranges
        for age_thresh_pair in age_thresh_pairs:
            age_thresh_lower, age_thresh_upper = age_thresh_pair
            age_str = f"{np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f}"

            # MC runs: compute the mean mass- and light-weighted ages plus the total mass in this age range
            age_lw_mean, age_lw_std = compute_mean_age(pp_mc_list, isochrones, "lw", age_thresh_lower, age_thresh_upper)
            age_mw_mean, age_mw_std = compute_mean_age(pp_mc_list, isochrones, "mw", age_thresh_lower, age_thresh_upper)
            mass_mean, mass_std = compute_mean_mass(pp_mc_list, isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)

            # Regul run: compute the mean mass- and light-weighted ages plus the total mass in this age range
            age_mw_regul = 10**compute_mw_age(sfh_regul_mw_1D, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)[0]
            age_lw_regul = 10**compute_lw_age(sfh_regul_lw_1D, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)[0]
            mass_regul = compute_mass(sfh_regul_mw_1D, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)

            # Input: compute the mean mass- and light-weighted ages plus the total mass in this age range
            age_mw_input = 10**compute_mw_age(sfh_mw_input, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)[0]
            age_lw_input = 10**compute_lw_age(sfh_lw_input, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)[0]
            mass_input = compute_mass(sfh_mw_input, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)

            # Put in DataFrame
            thisrow[f"MW age {age_str} (input)"] = age_mw_input
            thisrow[f"LW age {age_str} (input)"] = age_lw_input
            thisrow[f"Mass {age_str} (input)"] = mass_input
            thisrow[f"MW age {age_str} (MC) mean"] = age_mw_mean
            thisrow[f"LW age {age_str} (MC) mean"] = age_lw_mean
            thisrow[f"Mass {age_str} (MC) mean"] = mass_mean
            thisrow[f"MW age {age_str} (MC) std. dev."] = age_mw_std
            thisrow[f"LW age {age_str} (MC) std. dev."] = age_lw_std
            thisrow[f"Mass {age_str} (MC) std. dev."] = mass_std
            thisrow[f"MW age {age_str} (regularised)"] = age_mw_regul
            thisrow[f"LW age {age_str} (regularised)"] = age_lw_regul
            thisrow[f"Mass {age_str} (regularised)"] = mass_regul

        df = df.append(thisrow, ignore_index=True)

    ###############################################################################
    # Save DataFrame to file 
    ###############################################################################
    # Add metadata 
    df["SNR"] = SNR
    df["niters"] = niters
    df["nthreads"] = nthreads
    df["z"] = z
    df["Emission lines"] = False
    df["isochrones"] = isochrones
    df["sigma_star_kms"] = sigma_star_kms

    # Save
    df.to_hdf(os.path.join(data_path, f"ga{gal}_agncont.hd5"), key="agn")
