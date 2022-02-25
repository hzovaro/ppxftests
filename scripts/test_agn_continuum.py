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
# import matplotlib.pyplot as plt
# plt.ion()
# plt.close("all")

from IPython.core.debugger import Tracer

data_path = "/priv/meggs3/u5708159/ppxftests/"

###############################################################################
# Settings
###############################################################################
isochrones = "Padova"
SNR = 100
z = 0

# Get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates(isochrones)

# ppxf settings
niters = 100
nthreads = 25 if niters > 25 else niters
fit_agn_cont = True  # NOTE: run this TWICE - once with AGN cont. in the fit, once without.

# Parameters
# alpha_nu_vals = np.linspace(0.3, 2.1, 5)  # What is a typical value for low-z Seyfert galaxies?
alpha_nu_vals = [np.nan, 0.3, 0.5, 1.0, 2.0]
x_AGN_vals = [np.nan, 0.1, 0.5, 1.0, 2.0] # NOTE: NaNs are included to create a "special case" where we run without the AGN continuum

###########################################################################
# Helper function for running MC simulations
###########################################################################
def ppxf_helper(args):
    # Unpack arguments
    seed, spec, spec_err, lambda_vals_A, fit_agn_cont = args
    
    # Add "extra" noise to the spectrum
    rng = RandomState(seed)
    noise = rng.normal(scale=spec_err)
    spec_noise = spec + noise

    # This is to mitigate the "edge effects" of the convolution with the LSF
    spec_noise[0] = -9999
    spec_noise[-1] = -9999

    # Run ppxf
    pp = run_ppxf(spec=spec_noise, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                  z=z, isochrones="Padova",
                  fit_gas=False, tie_balmer=True, ngascomponents=1,
                  fit_agn_cont=fit_agn_cont,
                  reddening=1.0, mdegree=-1,
                  regularisation_method="none", 
                  plotit=False, savefigs=False, interactive_mode=False)
    return pp


###########################################################################
# Function for 
###########################################################################
def add_stuff_to_df(pp_mc_list, pp_regul, ages, isochrones):
    # STUFF TO STORE IN THE DATA FRAME
    thisrow = {}  # Row to append to DataFrame

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

    # "Sky" template weights 
    if fit_agn_cont:
        thisrow["ppxf alpha_nu_vals"] = [pp_regul.alpha_nu_vals]
        thisrow["AGN template weights (MC mean)"] =[np.nanmean([pp.weights_agn for pp in pp_mc_list], axis=0)]
        thisrow["AGN template weights (MC error)"] = [np.nanstd([pp.weights_agn for pp in pp_mc_list], axis=0)]
        thisrow["AGN template weights (regularised)"] = [pp_regul.weights_agn]

    return thisrow

###############################################################################
# Load a realistic SFH
###############################################################################
gals = [int(g) for g in sys.argv[1:]]
for gal in gals:
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
    df = pd.DataFrame(dtype="object")
    for alpha_nu, x_AGN in tqdm(product(alpha_nu_vals, x_AGN_vals), total=len(alpha_nu_vals) * len(x_AGN_vals)):
        # IF both are NaN, then do NOT add an AGN continuum
        if np.isnan(x_AGN) and np.isnan(alpha_nu):
            add_agn_continuum = False 
        elif np.isnan(x_AGN) or np.isnan(alpha_nu):
            continue
        else:
            add_agn_continuum = True
        print(f"{x_AGN}, {alpha_nu}")

        # Create spectrum
        spec, spec_err, lambda_vals_A = create_mock_spectrum(
            sfh_mass_weighted=sfh_mw_input,
            agn_continuum=add_agn_continuum, x_AGN=x_AGN, alpha_nu=alpha_nu,
            isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
            plotit=False)

        ###########################################################################
        # Run ppxf with regularisation
        ###########################################################################
        t = time()
        print(f"Gal {gal:004}: Regularisation: running ppxf on {nthreads} threads...")
        pp_regul = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                      z=z, isochrones=isochrones,
                      fit_gas=False, tie_balmer=True, ngascomponents=1,
                      fit_agn_cont=fit_agn_cont,
                      reddening=1.0, mdegree=-1,
                      regularisation_method="auto",
                      delta_regul_min=1, regul_max=5e4, delta_delta_chi2_min=1,
                      regul_nthreads=nthreads,
                      plotit=False, savefigs=False, interactive_mode=False)
        print(f"Gal {gal:004}: Regularisation: total time in run_ppxf: {time() - t:.2f} seconds")

        ###########################################################################
        # Run ppxf WITHOUT regularisation, using a MC approach
        ###########################################################################
        # Input arguments
        seeds = list(np.random.randint(low=0, high=100 * niters, size=niters))
        args_list = [[s, spec, spec_err, lambda_vals_A, fit_agn_cont] for s in seeds]

        # Run in parallel
        print(f"Gal {gal:004}: MC simulations: running ppxf on {nthreads} threads...")
        t = time()
        with multiprocessing.Pool(nthreads) as pool:
            pp_mc_list = list(tqdm(pool.imap(ppxf_helper, args_list), total=niters))
        print(f"Gal {gal:004}: MC simulations: total time in ppxf: {time() - t:.2f} s")

        ###########################################################################
        # Compute mean quantities from the ppxf runs
        ###########################################################################
        thisrow = add_stuff_to_df(pp_mc_list, pp_regul, ages, isochrones)
        thisrow["AGN continuum in input?"] = add_agn_continuum
        thisrow["alpha_nu"] = alpha_nu
        thisrow["x_AGN"] = x_AGN

        df = df.append({**thisrow, **truth_dict}, ignore_index=True)
        
        # Add metadata 
        df["SNR"] = SNR
        df["niters"] = niters
        df["nthreads"] = nthreads
        df["z"] = z
        df["Emission lines?"] = False
        df["AGN continuum included in fit?"] = fit_agn_cont
        df["isochrones"] = isochrones
        df["sigma_star_kms"] = sigma_star_kms

        # Save every iteration in case it shits itself 
        df.to_hdf(os.path.join(data_path, f"ga{gal}_agncont.hd5"), key="agn")

    ###############################################################################
    # Save DataFrame to file 
    ###############################################################################
    df.to_hdf(os.path.join(data_path, f"ga{gal}_agncont.hd5"), key="agn")
