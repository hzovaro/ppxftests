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
from ppxftests.ssputils import load_ssp_templates
from ppxftests.mockspec import create_mock_spectrum
from ppxftests.sfhutils import load_sfh, compute_mw_age, compute_lw_age, compute_cumulative_mass, compute_cumulative_light
from ppxftests.sfhutils import compute_mean_age, compute_mean_mass, compute_mean_sfr, compute_mean_1D_sfh
from ppxftests.ppxf_plot import plot_sfh_mass_weighted, plot_sfh_light_weighted

import matplotlib
matplotlib.use("agg")

from IPython.core.debugger import Tracer

data_path = "/priv/meggs3/u5708159/ppxftests/ext_and_agn/"

"""
We have found that ppxf seems to do OK at estimating the extinction 
when an AGN continuum is present. 
Here: run ppxf on an input SFH with varying AGN continua and varying 
stellar extinction.
"""
t_start = time()

###############################################################################
# Settings
###############################################################################
debug = False
isochrones = "Padova"
SNR = 100
z = 0.02  # Typical S7 galaxy refshift
R_V = 4.05  # For the Calzetti+2000 reddening curve

# Get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates(isochrones)

# ppxf settings
if not debug:
    niters = 500
    nthreads = 56
    delta_delta_chi2_min = 1.0
else:
    niters = 25
    nthreads = 25
    delta_delta_chi2_min = 1e3
fit_agn_cont = True
mpoly = -1  # If -1, use the reddening keyword to fit an extinction curve.
reddening = 1.0 if mpoly == -1 else None

# Variables
if not debug:
    alpha_nu_vals = [np.nan, 0.3, 0.5, 1.0, 2.0]
    x_AGN_vals = [np.nan, 0.1, 0.2, 0.5, 1.0, 2.0] # NOTE: NaNs are included to create a "special case" where we run without the AGN continuum
    A_V_vals = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
else:
    alpha_nu_vals = [0.3]
    x_AGN_vals = [1.5]
    A_V_vals = [2.5]

###########################################################################
# Helper function for running MC simulations
###########################################################################
def ppxf_helper(args):
    # Unpack arguments
    seed, spec, spec_err, lambda_vals_rest_A, fit_agn_cont = args
    
    # Add "extra" noise to the spectrum
    rng = RandomState(seed)
    noise = rng.normal(scale=spec_err)
    spec_noise = spec + noise

    # This is to mitigate the "edge effects" of the convolution with the LSF
    spec_noise[0] = -9999
    spec_noise[-1] = -9999

    # Run ppxf
    pp = run_ppxf(spec=spec_noise, spec_err=spec_err, lambda_vals_A=lambda_vals_rest_A,
                  isochrones=isochrones,
                  z=0.0, ngascomponents=1,
                  fit_gas=False, tie_balmer=False,
                  fit_agn_cont=fit_agn_cont,
                  reddening=reddening, mdegree=mpoly,
                  regularisation_method="none")
    return pp


###########################################################################
# Function for 
###########################################################################
def add_stuff_to_df(pp_mc_list, pp_regul, ages, isochrones):
    # STUFF TO STORE IN THE DATA FRAME
    thisrow = {}  # Row to append to DataFrame

    # 1-dimensional weights
    thisrow["Stellar template weights (MC mean)"] =[np.nanmean([pp.weights_stellar for pp in pp_mc_list], axis=0)]
    thisrow["Stellar template weights (MC error)"] = [np.nanstd([pp.weights_stellar for pp in pp_mc_list], axis=0)]
    thisrow["Stellar template weights (regularised)"] = [pp_regul.weights_stellar]

    # Kinematic solution
    thisrow["v_* (MC mean)"] = np.nanmean([pp.sol[0] for pp in pp_mc_list])
    thisrow["v_* (MC error)"] = np.nanstd([pp.sol[0] for pp in pp_mc_list])
    thisrow["v_* (regularised)"] = pp_regul.sol[0]
    thisrow["sigma_* (MC mean)"] = np.nanmean([pp.sol[1] for pp in pp_mc_list])
    thisrow["sigma_* (MC error)"] = np.nanstd([pp.sol[1] for pp in pp_mc_list])
    thisrow["sigma_* (regularised)"] = pp_regul.sol[1]

    # Light-weighted SFH (1D)
    thisrow["SFH LW 1D (MC mean)"] = [np.nanmean(np.array([pp.sfh_lw_1D for pp in pp_mc_list]), axis=0)]
    thisrow["SFH LW 1D (MC error)"] = [np.nanstd(np.array([pp.sfh_lw_1D for pp in pp_mc_list]), axis=0)]
    thisrow["SFH LW 1D (regularised)"] = [pp_regul.sfh_lw_1D]

    # Mass-weighted SFH (1D)
    thisrow["SFH MW 1D (MC mean)"] = [np.nanmean(np.array([pp.sfh_mw_1D for pp in pp_mc_list]), axis=0)]
    thisrow["SFH MW 1D (MC error)"] = [np.nanstd(np.array([pp.sfh_mw_1D for pp in pp_mc_list]), axis=0)]
    thisrow["SFH MW 1D (regularised)"] = [pp_regul.sfh_mw_1D]

    # Cumulative mass (log)
    thisrow["Cumulative mass vs. age cutoff (MC mean)"] = [np.nanmean(np.array([[compute_cumulative_mass(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), axis=0)]
    thisrow["Cumulative mass vs. age cutoff (MC error)"] = [np.nanstd(np.array([[compute_cumulative_mass(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), axis=0)]
    thisrow["Cumulative mass vs. age cutoff (regularised)"] = [np.array([compute_cumulative_mass(pp_regul.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]])]

    # Cumulative light (log)
    thisrow["Cumulative light vs. age cutoff (MC mean)"] = [np.nanmean(np.array([[compute_cumulative_light(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), axis=0)]
    thisrow["Cumulative light vs. age cutoff (MC error)"] = [np.nanstd(np.array([[compute_cumulative_light(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), axis=0)]
    thisrow["Cumulative light vs. age cutoff (regularised)"] = [np.array([compute_cumulative_light(pp_regul.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]])]

    # Mass-weighted age as a function of time (log)
    thisrow["Mass-weighted age vs. age cutoff (MC mean)"] = [np.nanmean(np.array([[compute_mw_age(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), axis=0)]
    thisrow["Mass-weighted age vs. age cutoff (MC error)"] = [np.nanstd(np.array([[compute_mw_age(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), axis=0)]
    thisrow["Mass-weighted age vs. age cutoff (regularised)"] = [np.array([compute_mw_age(pp_regul.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]])]

    # Light-weighted age as a function of time (log)
    thisrow["Light-weighted age vs. age cutoff (MC mean)"] = [np.nanmean(np.array([[compute_lw_age(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), axis=0)]
    thisrow["Light-weighted age vs. age cutoff (MC error)"] = [np.nanstd(np.array([[compute_lw_age(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), axis=0)]
    thisrow["Light-weighted age vs. age cutoff (regularised)"] = [np.array([compute_lw_age(pp_regul.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]])]

    # Extinction and/or polynomial fits
    if reddening is not None:
        thisrow["A_V (MC mean)"] = R_V * np.nanmean([pp.reddening for pp in pp_mc_list])
        thisrow["A_V (MC error)"] = R_V * np.nanstd([pp.reddening for pp in pp_mc_list])
        thisrow["A_V (regularised)"] = pp_regul.reddening * R_V
        thisrow["10^-0.4A(lambda) (MC mean)"] = [np.nanmean(np.array([pp.mpoly for pp in pp_mc_list]), axis=0)]
        thisrow["10^-0.4A(lambda) (MC error)"] = [np.nanstd(np.array([pp.mpoly for pp in pp_mc_list]), axis=0)]
        thisrow["10^-0.4A(lambda) (regularised)"] = [pp_regul.mpoly]
    else:
        thisrow["Multiplicative polynomial (MC mean)"] = [np.nanmean(np.array([pp.mpoly for pp in pp_mc_list]), axis=0)]
        thisrow["Multiplicative polynomial (MC error)"] = [np.nanstd(np.array([pp.mpoly for pp in pp_mc_list]), axis=0)]
        thisrow["Multiplicative polynomial (regularised)"] = [pp_regul.mpoly]

    # AGN template weights 
    if fit_agn_cont:
        thisrow["ppxf alpha_nu_vals"] = [pp_regul.alpha_nu_vals]
        thisrow["AGN template weights (MC mean)"] =[np.nanmean([pp.weights_agn for pp in pp_mc_list], axis=0)]
        thisrow["AGN template weights (MC error)"] = [np.nanstd([pp.weights_agn for pp in pp_mc_list], axis=0)]
        thisrow["AGN template weights (regularised)"] = [pp_regul.weights_agn]
        # Express as a fraction of the total stellar weights (accounts for extinction)
        thisrow["x_AGN (total, MC mean)"] = np.nanmean([np.nansum(pp.weights_agn) / np.nansum(pp.weights_stellar) for pp in pp_mc_list])
        thisrow["x_AGN (total, MC error)"] = np.nanstd([np.nansum(pp.weights_agn) / np.nansum(pp.weights_stellar) for pp in pp_mc_list])
        thisrow["x_AGN (total, regularised)"] = np.nansum(pp_regul.weights_agn) / np.nansum(pp_regul.weights_stellar)
        thisrow["x_AGN (individual, MC mean)"] = np.nanmean([pp.weights_agn / np.nansum(pp.weights_stellar) for pp in pp_mc_list], axis=0)
        thisrow["x_AGN (individual, MC error)"] = np.nanstd([pp.weights_agn / np.nansum(pp.weights_stellar) for pp in pp_mc_list], axis=0)
        thisrow["x_AGN (individual, regularised)"] = pp_regul.weights_agn / np.nansum(pp_regul.weights_stellar)

    return thisrow

###############################################################################
# Load a realistic SFH
###############################################################################
gals = [int(g) for g in sys.argv[1:]]
for gal in gals:
    # Filename 
    if mpoly == -1:
        df_fname = f"ga{gal:004d}_ext.hd5" if not debug else f"ga{gal:004d}_DEBUG.hd5"
    else:
        df_fname = f"ga{gal:004d}_mpoly.hd5" if not debug else f"ga{gal:004d}_DEBUG.hd5"

    if os.path.exists(os.path.join(data_path, df_fname)):
        # Load the existing DF
        print(f"WARNING: file '{os.path.join(data_path, df_fname)}' exists - appending to existing file!")
        df = pd.read_hdf(os.path.join(data_path, df_fname))
        df_exists = True
    else:
        df = pd.DataFrame(dtype="object")
        df_exists = False

    try:
        sfh_mw_input, sfh_lw_input, sfr_avg_input, sigma_star_kms = load_sfh(gal, plotit=False)
    except:
        print(f"ERROR: unable to load galaxy {gal}. Skipping...")
        sys.exit()

    ###############################################################################
    # DEFINE THE INPUT SFH
    # COMPUTE THE TRUTH VALUES 
    # STORE THESE SOMEWHERE SO THEY CAN LATER BE ADDED TO THE DATAFRAME
    ###############################################################################
    truth_dict = {}
    truth_dict["SFH LW 1D (input)"] = np.nansum(sfh_lw_input, axis=0)
    truth_dict["SFH MW 1D (input)"] = np.nansum(sfh_mw_input, axis=0)
    truth_dict["Cumulative mass vs. age cutoff (input)"] = np.array([compute_cumulative_mass(np.nansum(sfh_mw_input, axis=0), isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]])
    truth_dict["Cumulative light vs. age cutoff (input)"] = np.array([compute_cumulative_light(np.nansum(sfh_lw_input, axis=0), isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]])
    truth_dict["Mass-weighted age vs. age cutoff (input)"] = np.array([compute_mw_age(np.nansum(sfh_mw_input, axis=0), isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]]) 
    truth_dict["Light-weighted age vs. age cutoff (input)"] = np.array([compute_lw_age(np.nansum(sfh_lw_input, axis=0), isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]]) 

    ###############################################################################
    # RUN PPXF ON SPECTRA W/ EACH COMBINATION OF ALPHA_NU AND X_AGN
    # ONE ROW FOR EACH COMBINATION
    # APPEND TO DATA FRAME
    ###############################################################################
    for alpha_nu, x_AGN, A_V in product(alpha_nu_vals, x_AGN_vals, A_V_vals):
        # IF both are NaN, then do NOT add an AGN continuum
        if np.isnan(x_AGN) and np.isnan(alpha_nu):
            add_agn_continuum = False 
        elif np.isnan(x_AGN) or np.isnan(alpha_nu):
            continue
        else:
            add_agn_continuum = True

        # If this combination already exists in the DataFrame, then skip
        if df_exists:
            if np.isnan(x_AGN) and np.isnan(alpha_nu):
                cond = np.isnan(df["alpha_nu (input)"]) & np.isnan(df["x_AGN (input)"]) & (df["A_V (input)"] == A_V)
            else:
                cond = (df["alpha_nu (input)"] == alpha_nu) & (df["x_AGN (input)"] == x_AGN) & (df["A_V (input)"] == A_V)
            if np.any(cond):
                print(f"Combination alpha_nu = {alpha_nu:.1f}, x_AGN = {x_AGN:.1f}, A_V = {A_V:.1f} already exists in DataFrame - skipping...")
                continue

        t0 = time()
        print("***************************************************************")
        print(f"Beginning new iteration:")
        print(f"A_V = {A_V}, x_AGN = {x_AGN}, alpha_nu = {alpha_nu}")
        print("***************************************************************")

        # Create spectrum
        spec, spec_err, lambda_vals_obs_A = create_mock_spectrum(
            sfh_mass_weighted=sfh_mw_input,
            agn_continuum=add_agn_continuum, x_AGN=x_AGN, alpha_nu=alpha_nu,
            isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
            A_V=A_V, seed=0,
            plotit=False)
        lambda_vals_rest_A = lambda_vals_obs_A / (1 + z)

        ###########################################################################
        # Run ppxf WITHOUT regularisation, using a MC approach
        ###########################################################################
        # Input arguments
        seeds = list(np.random.randint(low=0, high=100 * niters, size=niters))
        args_list = [[s, spec, spec_err, lambda_vals_rest_A, fit_agn_cont] for s in seeds]

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
        pp_regul = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_rest_A,
                            isochrones=isochrones,
                            z=0.0, ngascomponents=1,
                            fit_gas=False, tie_balmer=False,
                            fit_agn_cont=fit_agn_cont,
                            reddening=reddening, mdegree=mpoly,
                            regularisation_method="auto", delta_delta_chi2_min=delta_delta_chi2_min,
                            regul_nthreads=nthreads)
        print(f"Gal {gal:004}: Regularisation: total time in run_ppxf: {time() - t:.2f} seconds")

        ###########################################################################
        # Compute mean quantities from the ppxf runs
        ###########################################################################
        thisrow = add_stuff_to_df(pp_mc_list, pp_regul, ages, isochrones)
        thisrow["AGN continuum in input?"] = add_agn_continuum
        thisrow["alpha_nu (input)"] = alpha_nu
        thisrow["x_AGN (input)"] = x_AGN
        thisrow["A_V (input)"] = A_V

        df = df.append({**thisrow, **truth_dict}, ignore_index=True)
        
        # Save every iteration in case it shits itself 
        df.to_hdf(os.path.join(data_path, df_fname), key="ext_and_agn")

        t1 = time()
        print("***************************************************************")
        print(f"Iteration complete: total time = {(t1 - t0) / 60.:.2f} minutes")
        print("***************************************************************")

    ###############################################################################
    # Save DataFrame to file 
    ###############################################################################
    # Add metadata 
    df["Wavelength values (rest frame, Å)"] = [lambda_vals_rest_A]
    df["Wavelength values (observer frame, Å)"] = [lambda_vals_obs_A]
    df["SNR"] = SNR
    df["niters"] = niters
    df["nthreads"] = nthreads
    df["z"] = z
    df["Emission lines?"] = False
    df["AGN continuum included in fit?"] = fit_agn_cont
    df["Extinction curve included in fit?"] = True if reddening is not None else False
    df["Multiplicative polynomial included in fit?"] = True if mpoly != -1 else False
    df["isochrones"] = isochrones
    df["sigma_* (input)"] = sigma_star_kms

    # Save
    df.to_hdf(os.path.join(data_path, df_fname), key="ext_and_agn")
    
t_end = time()
print("***************************************************************")
print(f"Total execution time = {(t_end - t_start) / 60.:.2f} minutes")
print("***************************************************************")

