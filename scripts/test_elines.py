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

# import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

data_path = "/priv/meggs3/u5708159/ppxftests/elines/"

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
debug = True
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
alpha_nu = np.nan
x_AGN = np.nan
A_V = 0.0

# Emission line parameters
# ngascomponents_vals = list(range(4))
# sigma_gas_vals = [[40] * nn for nn in range(4)]
# v_gas_vals = [[], [0], [-100, +100], [-200, 0, +200]]
# eline_model_vals = [["HII"] * nn for nn in range(4)]
# L_Ha_vals = [[1e41] * nn for nn in range(4)]

# For now: 2 runs only - 1 w/o emisison lines, the other with many complex lines.
ngascomponents_in_spec_vals = [4]
sigma_gas_vals = [[40, 125, 300, 5e3]]
v_gas_vals = [[-10, 0, -500, -1000]]
eline_model_vals = [["HII", "AGN", "AGN", "BLR"]]
L_Ha_vals = [[1e40, 3e40, 1e40, 1.5e40]]
ngascomponents_in_fit_vals = [3]

###########################################################################
# Helper function for running MC simulations
###########################################################################
def ppxf_helper(args):
    # Unpack arguments
    seed, spec, spec_err, lambda_vals_rest_A, ngascomponents = args
    
    # Add "extra" noise to the spectrum
    rng = RandomState(seed)
    noise = rng.normal(scale=spec_err)
    spec_noise = spec + noise

    # This is to mitigate the "edge effects" of the convolution with the LSF
    spec_noise[0] = -9999
    spec_noise[-1] = -9999

    # Run ppxf
    pp = run_ppxf(spec=spec_noise, spec_err=spec_err, lambda_vals_A=lambda_vals_rest_A,
                  isochrones=isochrones, z=0.0,
                  ngascomponents=ngascomponents, fit_gas=True if ngascomponents > 0 else False, tie_balmer=False,
                  fit_agn_cont=True,
                  reddening=reddening, mdegree=mpoly,
                  regularisation_method="none")
    return pp


###########################################################################
# Function for 
###########################################################################
def add_stuff_to_df(pp_mc_list, pp_regul):

    """
    Given a list of MC ppxf instances plus an instance of ppxf from a 
    regularised run, return a dict containing quantities derived the ppxf 
    runs.
    """
    
    isochrones = pp_regul.isochrones
    ages = pp_regul.ages

    # STUFF TO STORE IN THE DATA FRAME
    thisrow = {}  # Row to append to DataFrame

    # Fit parameters
    thisrow["Emission lines included in fit?"] = pp_regul.fit_gas
    thisrow["AGN continuum included in fit?"] = pp_regul.fit_agn_cont
    thisrow["Extinction curve included in fit?"] = True if pp_regul.reddening is not None else False
    thisrow["Multiplicative polynomial included in fit?"] = True if pp_regul.mdegree > 0 else False
    thisrow["Isochrones"] = isochrones

    # 1-dimensional weights
    thisrow["Stellar template weights (MC mean)"] = np.nanmean([pp.weights_stellar for pp in pp_mc_list], axis=0)
    thisrow["Stellar template weights (MC error)"] = np.nanstd([pp.weights_stellar for pp in pp_mc_list], axis=0)
    thisrow["Stellar template weights (regularised)"] = pp_regul.weights_stellar

    # Stellar kinematics
    if pp_regul.fit_gas:
        thisrow["v_* (MC mean)"] = np.nanmean([pp.sol[0][0] for pp in pp_mc_list])
        thisrow["v_* (MC error)"] = np.nanstd([pp.sol[0][0] for pp in pp_mc_list])
        thisrow["v_* (regularised)"] = pp_regul.sol[0][0]
        thisrow["sigma_* (MC mean)"] = np.nanmean([pp.sol[0][1] for pp in pp_mc_list])
        thisrow["sigma_* (MC error)"] = np.nanstd([pp.sol[0][1] for pp in pp_mc_list])
        thisrow["sigma_* (regularised)"] = pp_regul.sol[0][1]
    else:
        thisrow["v_* (MC mean)"] = np.nanmean([pp.sol[0] for pp in pp_mc_list])
        thisrow["v_* (MC error)"] = np.nanstd([pp.sol[0] for pp in pp_mc_list])
        thisrow["v_* (regularised)"] = pp_regul.sol[0]
        thisrow["sigma_* (MC mean)"] = np.nanmean([pp.sol[1] for pp in pp_mc_list])
        thisrow["sigma_* (MC error)"] = np.nanstd([pp.sol[1] for pp in pp_mc_list])
        thisrow["sigma_* (regularised)"] = pp_regul.sol[1]

    # Emisson lines: fluxes and kinematics
    if pp_regul.fit_gas:
        ngascomponents = np.nanmax(pp_regul.component)
        thisrow["Number of emission line components in fit"] = ngascomponents
        thisrow["Emission lines fitted"] = pp_regul.gas_names
        thisrow["F_gas erg/s (regularised)"] = pp_regul.gas_flux * pp_regul.norm
        thisrow["F_gas erg/s (MC mean)"] = np.nanmean([pp.gas_flux * pp.norm for pp in pp_mc_list], axis=0)
        thisrow["F_gas erg/s (MC error)"] = np.nanstd([pp.gas_flux * pp.norm for pp in pp_mc_list], axis=0)

        thisrow["v_gas (regularised)"] = pp_regul.v_gas
        thisrow["v_gas (MC mean)"] = np.nanmean([pp.v_gas for pp in pp_mc_list], axis=0)
        thisrow["v_gas (MC error)"] = np.nanstd([pp.v_gas for pp in pp_mc_list], axis=0)
        thisrow["sigma_gas (regularised)"] = pp_regul.sigma_gas
        thisrow["sigma_gas (MC mean)"] = np.nanmean([pp.sigma_gas for pp in pp_mc_list], axis=0)
        thisrow["sigma_gas (MC error)"] = np.nanstd([pp.sigma_gas for pp in pp_mc_list], axis=0)
    else:
        thisrow["Number of emission line components in fit"] = 0
  
    # Light-weighted SFH (1D)
    thisrow["SFH LW 1D (MC mean)"] = np.nanmean(np.array([pp.sfh_lw_1D for pp in pp_mc_list]), axis=0)
    thisrow["SFH LW 1D (MC error)"] = np.nanstd(np.array([pp.sfh_lw_1D for pp in pp_mc_list]), axis=0)
    thisrow["SFH LW 1D (regularised)"] = pp_regul.sfh_lw_1D

    # Mass-weighted SFH (1D)
    thisrow["SFH MW 1D (MC mean)"] = np.nanmean(np.array([pp.sfh_mw_1D for pp in pp_mc_list]), axis=0)
    thisrow["SFH MW 1D (MC error)"] = np.nanstd(np.array([pp.sfh_mw_1D for pp in pp_mc_list]), axis=0)
    thisrow["SFH MW 1D (regularised)"] = pp_regul.sfh_mw_1D

    # Cumulative mass (log)
    thisrow["Cumulative mass vs. age cutoff (MC mean)"] = np.nanmean(np.array([[compute_cumulative_mass(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Cumulative mass vs. age cutoff (MC error)"] = np.nanstd(np.array([[compute_cumulative_mass(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Cumulative mass vs. age cutoff (regularised)"] = np.array([compute_cumulative_mass(pp_regul.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]])

    # Cumulative light (log)
    thisrow["Cumulative light vs. age cutoff (MC mean)"] = np.nanmean(np.array([[compute_cumulative_light(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Cumulative light vs. age cutoff (MC error)"] = np.nanstd(np.array([[compute_cumulative_light(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Cumulative light vs. age cutoff (regularised)"] = np.array([compute_cumulative_light(pp_regul.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]])

    # Mass-weighted age as a function of time (log)
    thisrow["Mass-weighted age vs. age cutoff (MC mean)"] = np.nanmean(np.array([[compute_mw_age(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Mass-weighted age vs. age cutoff (MC error)"] = np.nanstd(np.array([[compute_mw_age(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Mass-weighted age vs. age cutoff (regularised)"] = np.array([compute_mw_age(pp_regul.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]])

    # Light-weighted age as a function of time (log)
    thisrow["Light-weighted age vs. age cutoff (MC mean)"] = np.nanmean(np.array([[compute_lw_age(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Light-weighted age vs. age cutoff (MC error)"] = np.nanstd(np.array([[compute_lw_age(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Light-weighted age vs. age cutoff (regularised)"] = np.array([compute_lw_age(pp_regul.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]])

    # Extinction and/or polynomial fits
    if pp_regul.reddening is not None:
        thisrow["A_V (MC mean)"] = R_V * np.nanmean([pp.reddening for pp in pp_mc_list])
        thisrow["A_V (MC error)"] = R_V * np.nanstd([pp.reddening for pp in pp_mc_list])
        thisrow["A_V (regularised)"] = pp_regul.reddening * R_V
        thisrow["10^-0.4A(lambda) (MC mean)"] = np.nanmean(np.array([pp.mpoly for pp in pp_mc_list]), axis=0)
        thisrow["10^-0.4A(lambda) (MC error)"] = np.nanstd(np.array([pp.mpoly for pp in pp_mc_list]), axis=0)
        thisrow["10^-0.4A(lambda) (regularised)"] = pp_regul.mpoly
    else:
        thisrow["Multiplicative polynomial (MC mean)"] = np.nanmean(np.array([pp.mpoly for pp in pp_mc_list]), axis=0)
        thisrow["Multiplicative polynomial (MC error)"] = np.nanstd(np.array([pp.mpoly for pp in pp_mc_list]), axis=0)
        thisrow["Multiplicative polynomial (regularised)"] = pp_regul.mpoly
    thisrow["Wavelength (rest frame, Å, log-rebinned)"] = pp_regul.lam

    # AGN template weights 
    if pp_regul.fit_agn_cont:
        thisrow["ppxf alpha_nu_vals"] = pp_regul.alpha_nu_vals
        thisrow["AGN template weights (MC mean)"] = np.nanmean([pp.weights_agn for pp in pp_mc_list], axis=0)
        thisrow["AGN template weights (MC error)"] = np.nanstd([pp.weights_agn for pp in pp_mc_list], axis=0)
        thisrow["AGN template weights (regularised)"] = pp_regul.weights_agn
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
    for nn in range(len(ngascomponents_in_spec_vals)):
        ngascomponents_in_spec = ngascomponents_in_spec_vals[nn]
        ngascomponents_in_fit = ngascomponents_in_fit_vals[nn]

        # IF both are NaN, then do NOT add an AGN continuum
        if np.isnan(x_AGN) and np.isnan(alpha_nu):
            add_agn_continuum = False 
        elif np.isnan(x_AGN) or np.isnan(alpha_nu):
            continue
        else:
            add_agn_continuum = True

        # If this combination already exists in the DataFrame, then skip
        if df_exists and not debug:
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
            ngascomponents=ngascomponents_in_spec, v_gas_kms=v_gas_vals[nn], sigma_gas_kms=sigma_gas_vals[nn], 
            eline_model=eline_model_vals[nn], L_Ha_erg_s=L_Ha_vals[nn],
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
        args_list = [[s, spec, spec_err, lambda_vals_rest_A, ngascomponents_in_fit] for s in seeds]

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
                            isochrones=isochrones, z=0.0, 
                            fit_gas=True if ngascomponents_in_fit > 0 else False, ngascomponents=ngascomponents_in_fit, tie_balmer=False,
                            fit_agn_cont=fit_agn_cont,
                            reddening=reddening, mdegree=mpoly,
                            regularisation_method="auto", delta_delta_chi2_min=delta_delta_chi2_min,
                            regul_nthreads=nthreads,
                            plotit=True if debug else False)
        print(f"Gal {gal:004}: Regularisation: total time in run_ppxf: {time() - t:.2f} seconds")   

        ###########################################################################
        # Compute mean quantities from the ppxf runs
        ###########################################################################
        thisrow = add_stuff_to_df(pp_mc_list, pp_regul)
        thisrow["Number of emission line components in input"] = ngascomponents_in_spec
        thisrow["Emission lines included in input?"] = True if ngascomponents_in_spec > 0 else False
        thisrow["AGN continuum included in input?"] = add_agn_continuum
        thisrow["alpha_nu (input)"] = alpha_nu
        thisrow["x_AGN (input)"] = x_AGN
        thisrow["A_V (input)"] = A_V
        thisrow["sigma_* (input)"] = sigma_star_kms
        thisrow["SNR"] = SNR
        thisrow["niters"] = niters
        thisrow["nthreads"] = nthreads
        thisrow["z"] = z

        # Info about emission lines
        thisrow["v_gas (input)"] = v_gas_vals[nn]
        thisrow["sigma_gas (input)"] = sigma_gas_vals[nn]
        thisrow["L(Ha) (input)"] = L_Ha_vals[nn]
        thisrow["Emission line model (input)"] = eline_model_vals[nn]
        
        df = df.append({**thisrow, **truth_dict}, ignore_index=True)

        ###############################################################################
        # Save DataFrame to file 
        ###############################################################################
        # Save every iteration in case it shits itself 
        df.to_hdf(os.path.join(data_path, df_fname), key="elines")

        t1 = time()
        print("***************************************************************")
        print(f"Iteration complete: total time = {(t1 - t0) / 60.:.2f} minutes")
        print("***************************************************************")

t_end = time()
print("***************************************************************")
print(f"Total execution time = {(t_end - t_start) / 60.:.2f} minutes")
print("***************************************************************")

