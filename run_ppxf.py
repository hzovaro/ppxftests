###############################################################################
#
#   File:       ppxf_integrated.py
#   Author:     Henry Zovaro
#   Email:      henry.zovaro@anu.edu.au
#
#   Description:
#   Run ppxf on a spectrum extracted from the central regions of an S7 datacube
#   to determine the star formation history and emission line properties.
#
#   Copyright (C) 2021 Henry Zovaro
#
###############################################################################
#!/usr/bin/env python
import sys, os

import matplotlib

from time import time

from scipy import constants
import numpy as np
from numpy.random import RandomState
import extinction
from itertools import product
import multiprocessing
import pandas as pd

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

from cosmocalc import get_dist
from log_rebin_errors import log_rebin_errors

from ppxftests.ssputils import load_ssp_templates, get_bin_edges_and_widths, log_rebin_and_convolve_stellar_templates
from ppxftests.sfhutils import load_sfh, compute_mw_age, compute_lw_age, compute_cumulative_mass, compute_cumulative_light
from ppxftests.mockspec import FWHM_WIFES_INST_A
from ppxftests.ppxf_plot import ppxf_plot, plot_sfh_mass_weighted

from IPython.core.debugger import Tracer

##############################################################################
# Plotting settings
##############################################################################
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rc("font", size=11)
matplotlib.rc("text", usetex=False)
# matplotlib.rc("font", **{"family": "serif"})
matplotlib.rc("image", interpolation="nearest")
matplotlib.rc("image", origin="lower")
# plt.close("all")
plt.ion()

##############################################################################
# For interactive execution
##############################################################################
def hit_key_to_continue():
    key = input("Hit a key to continue or q to quit...")
    if key == 'q':
        sys.exit()
    return

##############################################################################
# For fitting extinction in ppxf
##############################################################################
def reddening_fm07(lam, ebv):
    # lam in Angstroms
    # Need to derive A(lambda) from E(B-V)
    # fm07 takes as input lambda and A_V, so we first need to convert E(B-V)
    # into A_V
    R_V = 3.1
    A_V = R_V * ebv
    A_lambda = extinction.fm07(wave=lam, a_v=A_V, unit='aa')
    fact = 10**(-0.4 * A_lambda)  # Need a minus sign here!
    return fact

def reddening_calzetti00(lam, ebv):
    # lam in Angstroms
    # Need to derive A(lambda) from E(B-V)
    # calzetti00 takes as input lambda and A_V, so we first need to convert E(B-V)
    # into A_V
    R_V = 4.05
    A_V = R_V * ebv
    A_lambda = extinction.calzetti00(wave=lam, a_v=A_V, r_v=R_V, unit='aa')
    fact = 10**(-0.4 * A_lambda)  # Need a minus sign here!
    return fact

##############################################################################
# For printing emission line fluxes
##############################################################################
def sigfig(number, err):
    # Truncate error to one significant figure
    ndigits = int(np.floor(np.log10(err)))
    err_truncated = np.round(err, -ndigits)
    number_truncated = np.round(number, -ndigits)

    return number_truncated, err_truncated, -ndigits

def sci_notation(num, err):
    if num != 0:
        exp = int(np.floor(np.log10(np.abs(num))))
    else:
        exp = 0

    mantissa = num / float(10**exp)
    mantissa_err = err / float(10**exp)
    mantissa_dp, mantissa_err_dp, ndigits = sigfig(mantissa, mantissa_err)

    if exp != 0:
        s = "$ {:." + str(ndigits) + "f} \\pm {:." + str(ndigits) +\
            "f} \\times 10^{{{:d}}} $"
        return s.format(mantissa_dp, mantissa_err_dp, exp)
    else:
        s = "$ {:." + str(ndigits) + "f} \\pm {:." + str(ndigits) + "f}$"
        return s.format(mantissa_dp, mantissa_err_dp)

##############################################################################
# For doing MC simulations
##############################################################################
def ppxf_mc_helper(args):
    """
    Helper function used in ppxf_mc() for running Monte Carlo simulations 
    with ppxf. Note that no regularisation is used.

    Inputs:
    args        list containing the following:
        seed            integer required to seed the RNG for computing the extra 
                        noise to be added to the spectrum
        spec            input (noisy) spectrum
        spec_err        corresponding 1-sigma errors 
        lambda_vals_A   wavelength values (Angstroms)

    """

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
                  clean=False,
                  fit_gas=False, tie_balmer=True,
                  plotit=False, savefigs=False, interactive_mode=False)
    return pp

def ppxf_mc(spec, spec_err, lambda_vals_A,
            niters, nthreads):
    """
    Run Monte-Carlo simulations with ppxf.

    Run ppxf a total of niters times on the input spectrum (spec), each time 
    adding additional random noise goverend by the 1-sigma errors on the 
    input spectrum (spec_err)

    Inputs:
    spec            Input spectrum
    spec_err        Corresponding 1-sigma errors
    lambda_vals_A   Wavelength values for the spectrum (Angstroms)
    niters          Total number of MC iterations 
    nthreads        Number of threads used
    
    """
    # Input arguments
    seeds = list(np.random.randint(low=0, high=100 * niters, size=niters))
    args_list = [[s, spec, spec_err, lambda_vals_A] for s in seeds]

    # Run in parallel
    print(f"ppxf_mc(): running ppxf on {nthreads} threads...")
    t = time()
    with multiprocessing.Pool(nthreads) as pool:
        pp_list = list(tqdm(pool.imap(ppxf_mc_helper, args_list), total=niters))
    print(f"ppxf_mc(): elapsed time = {time() - t:.2f} s")

    return pp_list

##############################################################################
# For running in parallel
##############################################################################
def ppxf_helper(args):
    # Parse arguments
    templates, spec_log, spec_err_log, noise_scaling_factor,\
        velscale, start_kin, good_px, nmoments, adegree,\
        mdegree, dv, lambda_vals_log, regul, clean, reddening, reddening_calzetti00,\
        reg_dim, kinematic_components, gas_component, gas_names,\
        gas_reddening = args

    # Run ppxf
    pp_age_met = ppxf(templates=templates,
          galaxy=spec_log, noise=spec_err_log * noise_scaling_factor,
          velscale=np.squeeze(velscale), start=start_kin,
          goodpixels=good_px,
          moments=nmoments, degree=adegree, mdegree=mdegree,
          vsyst=dv,
          lam=np.exp(lambda_vals_log),
          regul=regul,
          clean=clean,
          reddening=reddening, reddening_func=reddening_calzetti00,
          reg_dim=reg_dim,
          component=kinematic_components, gas_component=gas_component,
          gas_names=gas_names, gas_reddening=gas_reddening, method="capfit",
          quiet=True)

    # Return
    return pp_age_met

###########################################################################
# Function for saving variables to a DataFrame from multiple ppxf runs
###########################################################################
def add_stuff_to_df(pp_mc_list, pp_regul,
                    R_V=4.05, 
                    plotit=False,
                    plot_fname=None,  # Only used for plotting
                    savefig=False,
                    gal=None,  # Only used for printing error messages
                    ):

    """
    Given a list of MC ppxf instances plus an instance of ppxf from a 
    regularised run, return a dict containing quantities derived the ppxf 
    runs.
    """
    isochrones = pp_regul.isochrones
    ages = pp_regul.ages

    # STUFF TO STORE IN THE DATA FRAME
    thisrow = {}  # Row to append to DataFrame

    if plotit:
        fig_hist, axs_hist = plt.subplots(nrows=5, ncols=2, figsize=(12, 20))
        fig_hist.subplots_adjust(hspace=0.3, left=0.1, right=0.9, top=0.95, bottom=0.05)
        if plot_fname is not None:
            fig_hist.suptitle(plot_fname.replace("_", " "))
        ax_iter = 0

    # Fit parameters
    thisrow["Emission lines included in fit?"] = pp_regul.fit_gas
    thisrow["AGN continuum included in fit?"] = pp_regul.fit_agn_cont
    thisrow["Extinction curve included in fit?"] = True if pp_regul.reddening is not None else False
    thisrow["Multiplicative polynomial included in fit?"] = True if pp_regul.mdegree > 0 else False
    thisrow["Isochrones"] = isochrones

    # 1-dimensional weights
    thisrow["Stellar template weights (MC mean)"] = np.nanmean([pp.weights_stellar for pp in pp_mc_list], axis=0)
    thisrow["Stellar template weights (MC error)"] = np.nanstd([pp.weights_stellar for pp in pp_mc_list], axis=0)
    thisrow["Stellar template weights (MC 50th percentile)"] = np.nanquantile(a=[pp.weights_stellar for pp in pp_mc_list], q=0.5, axis=0)
    thisrow["Stellar template weights (MC 16th percentile)"] = np.nanquantile(a=[pp.weights_stellar for pp in pp_mc_list], q=0.16, axis=0)
    thisrow["Stellar template weights (MC 84th percentile)"] = np.nanquantile(a=[pp.weights_stellar for pp in pp_mc_list], q=0.84, axis=0)
    thisrow["Stellar template weights (regularised)"] = pp_regul.weights_stellar

    # Stellar kinematics
    if pp_regul.fit_gas:
        thisrow["v_* (MC mean)"] = np.nanmean([pp.sol[0][0] for pp in pp_mc_list])
        thisrow["v_* (MC error)"] = np.nanstd([pp.sol[0][0] for pp in pp_mc_list])
        thisrow["v_* (MC 50th percentile)"] = np.nanquantile(a=[pp.sol[0][0] for pp in pp_mc_list], q=0.5)
        thisrow["v_* (MC 16th percentile)"] = np.nanquantile(a=[pp.sol[0][0] for pp in pp_mc_list], q=0.16)
        thisrow["v_* (MC 84th percentile)"] = np.nanquantile(a=[pp.sol[0][0] for pp in pp_mc_list], q=0.84)
        thisrow["v_* (regularised)"] = pp_regul.sol[0][0]
        thisrow["sigma_* (MC mean)"] = np.nanmean([pp.sol[0][1] for pp in pp_mc_list])
        thisrow["sigma_* (MC error)"] = np.nanstd([pp.sol[0][1] for pp in pp_mc_list])
        thisrow["sigma_* (MC 50th percentile)"] = np.nanquantile(a=[pp.sol[0][1] for pp in pp_mc_list], q=0.5)
        thisrow["sigma_* (MC 16th percentile)"] = np.nanquantile(a=[pp.sol[0][1] for pp in pp_mc_list], q=0.16)
        thisrow["sigma_* (MC 84th percentile)"] = np.nanquantile(a=[pp.sol[0][1] for pp in pp_mc_list], q=0.84)
        thisrow["sigma_* (regularised)"] = pp_regul.sol[0][1]
    else:
        thisrow["v_* (MC mean)"] = np.nanmean([pp.sol[0] for pp in pp_mc_list])
        thisrow["v_* (MC error)"] = np.nanstd([pp.sol[0] for pp in pp_mc_list])
        thisrow["v_* (MC 50th percentile)"] = np.nanquantile(a=[pp.sol[0] for pp in pp_mc_list], q=0.5)
        thisrow["v_* (MC 16th percentile)"] = np.nanquantile(a=[pp.sol[0] for pp in pp_mc_list], q=0.16)
        thisrow["v_* (MC 84th percentile)"] = np.nanquantile(a=[pp.sol[0] for pp in pp_mc_list], q=0.84)
        thisrow["v_* (regularised)"] = pp_regul.sol[0]
        thisrow["sigma_* (MC mean)"] = np.nanmean([pp.sol[1] for pp in pp_mc_list])
        thisrow["sigma_* (MC error)"] = np.nanstd([pp.sol[1] for pp in pp_mc_list])
        thisrow["sigma_* (MC 50th percentile)"] = np.nanquantile(a=[pp.sol[1] for pp in pp_mc_list], q=0.5)
        thisrow["sigma_* (MC 16th percentile)"] = np.nanquantile(a=[pp.sol[1] for pp in pp_mc_list], q=0.16)
        thisrow["sigma_* (MC 84th percentile)"] = np.nanquantile(a=[pp.sol[1] for pp in pp_mc_list], q=0.84)
        thisrow["sigma_* (regularised)"] = pp_regul.sol[1]

    # Emisson lines: fluxes and kinematics
    if pp_regul.fit_gas:
        ngascomponents = np.nanmax(pp_regul.component)
        thisrow["Number of emission line components in fit"] = ngascomponents
        thisrow["Emission lines fitted"] = pp_regul.gas_names
        thisrow["F_gas erg/s (regularised)"] = pp_regul.gas_flux * pp_regul.norm
        thisrow["F_gas erg/s (MC mean)"] = np.nanmean([pp.gas_flux * pp.norm for pp in pp_mc_list], axis=0)
        thisrow["F_gas erg/s (MC error)"] = np.nanstd([pp.gas_flux * pp.norm for pp in pp_mc_list], axis=0)
        thisrow["F_gas erg/s (MC 50th percentile)"] = np.nanquantile(a=[pp.gas_flux * pp.norm for pp in pp_mc_list], q=0.5, axis=0)
        thisrow["F_gas erg/s (MC 16th percentile)"] = np.nanquantile(a=[pp.gas_flux * pp.norm for pp in pp_mc_list], q=0.16, axis=0)
        thisrow["F_gas erg/s (MC 84th percentile)"] = np.nanquantile(a=[pp.gas_flux * pp.norm for pp in pp_mc_list], q=0.84, axis=0)

        thisrow["v_gas (regularised)"] = pp_regul.v_gas
        thisrow["v_gas (MC mean)"] = np.nanmean([pp.v_gas for pp in pp_mc_list], axis=0)
        thisrow["v_gas (MC error)"] = np.nanstd([pp.v_gas for pp in pp_mc_list], axis=0)
        thisrow["v_gas (MC 50th percentile)"] = np.nanquantile(a=[pp.v_gas for pp in pp_mc_list], q=0.5, axis=0)
        thisrow["v_gas (MC 16th percentile)"] = np.nanquantile(a=[pp.v_gas for pp in pp_mc_list], q=0.16, axis=0)
        thisrow["v_gas (MC 84th percentile)"] = np.nanquantile(a=[pp.v_gas for pp in pp_mc_list], q=0.84, axis=0)
        thisrow["sigma_gas (regularised)"] = pp_regul.sigma_gas
        thisrow["sigma_gas (MC mean)"] = np.nanmean([pp.sigma_gas for pp in pp_mc_list], axis=0)
        thisrow["sigma_gas (MC error)"] = np.nanstd([pp.sigma_gas for pp in pp_mc_list], axis=0)
        thisrow["sigma_gas (MC 50th percentile)"] = np.nanquantile(a=[pp.sigma_gas for pp in pp_mc_list], q=0.5, axis=0)
        thisrow["sigma_gas (MC 16th percentile)"] = np.nanquantile(a=[pp.sigma_gas for pp in pp_mc_list], q=0.16, axis=0)
        thisrow["sigma_gas (MC 84th percentile)"] = np.nanquantile(a=[pp.sigma_gas for pp in pp_mc_list], q=0.84, axis=0)
    else:
        thisrow["Number of emission line components in fit"] = 0
  
    # Light-weighted SFH (1D)
    thisrow["SFH LW 1D (MC mean)"] = np.nanmean(np.array([pp.sfh_lw_1D for pp in pp_mc_list]), axis=0)
    thisrow["SFH LW 1D (MC error)"] = np.nanstd(np.array([pp.sfh_lw_1D for pp in pp_mc_list]), axis=0)
    thisrow["SFH LW 1D (MC 50th percentile)"] = np.nanquantile(a=np.array([pp.sfh_lw_1D for pp in pp_mc_list]), q=0.5, axis=0)
    thisrow["SFH LW 1D (MC 16th percentile)"] = np.nanquantile(a=np.array([pp.sfh_lw_1D for pp in pp_mc_list]), q=0.16, axis=0)
    thisrow["SFH LW 1D (MC 84th percentile)"] = np.nanquantile(a=np.array([pp.sfh_lw_1D for pp in pp_mc_list]), q=0.84, axis=0)
    thisrow["SFH LW 1D (regularised)"] = pp_regul.sfh_lw_1D

    # Mass-weighted SFH (1D)
    thisrow["SFH MW 1D (MC mean)"] = np.nanmean(np.array([pp.sfh_mw_1D for pp in pp_mc_list]), axis=0)
    thisrow["SFH MW 1D (MC error)"] = np.nanstd(np.array([pp.sfh_mw_1D for pp in pp_mc_list]), axis=0)
    thisrow["SFH MW 1D (MC 50th percentile)"] = np.nanquantile(a=np.array([pp.sfh_mw_1D for pp in pp_mc_list]), q=0.5, axis=0)
    thisrow["SFH MW 1D (MC 16th percentile)"] = np.nanquantile(a=np.array([pp.sfh_mw_1D for pp in pp_mc_list]), q=0.16, axis=0)
    thisrow["SFH MW 1D (MC 84th percentile)"] = np.nanquantile(a=np.array([pp.sfh_mw_1D for pp in pp_mc_list]), q=0.84, axis=0)
    thisrow["SFH MW 1D (regularised)"] = pp_regul.sfh_mw_1D

    # Cumulative mass (log)
    thisrow["Cumulative mass vs. age cutoff (MC mean)"] = np.nanmean(np.array([[compute_cumulative_mass(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Cumulative mass vs. age cutoff (MC error)"] = np.nanstd(np.array([[compute_cumulative_mass(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Cumulative mass vs. age cutoff (MC 50th percentile)"] = np.nanquantile(a=np.array([[compute_cumulative_mass(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), q=0.5, axis=0)
    thisrow["Cumulative mass vs. age cutoff (MC 16th percentile)"] = np.nanquantile(a=np.array([[compute_cumulative_mass(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), q=0.16, axis=0)
    thisrow["Cumulative mass vs. age cutoff (MC 84th percentile)"] = np.nanquantile(a=np.array([[compute_cumulative_mass(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), q=0.84, axis=0)
    thisrow["Cumulative mass vs. age cutoff (regularised)"] = np.array([compute_cumulative_mass(pp_regul.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]])

    # Cumulative light (log)
    thisrow["Cumulative light vs. age cutoff (MC mean)"] = np.nanmean(np.array([[compute_cumulative_light(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Cumulative light vs. age cutoff (MC error)"] = np.nanstd(np.array([[compute_cumulative_light(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Cumulative light vs. age cutoff (MC 50th percentile)"] = np.nanquantile(a=np.array([[compute_cumulative_light(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), q=0.5, axis=0)
    thisrow["Cumulative light vs. age cutoff (MC 16th percentile)"] = np.nanquantile(a=np.array([[compute_cumulative_light(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), q=0.16, axis=0)
    thisrow["Cumulative light vs. age cutoff (MC 84th percentile)"] = np.nanquantile(a=np.array([[compute_cumulative_light(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]] for pp in pp_mc_list]), q=0.84, axis=0)
    thisrow["Cumulative light vs. age cutoff (regularised)"] = np.array([compute_cumulative_light(pp_regul.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a) for a in ages[1:]])

    # Cumulative mass fraction (log)
    M_tot_vals = np.zeros(len(pp_mc_list))
    M_cum_vals = np.zeros((len(pp_mc_list), len(ages[:-1])))
    M_frac_vals = np.zeros((len(pp_mc_list), len(ages[:-1])))
    fig, ax = plt.subplots(figsize=(10, 5)) if plotit else (None, None)
    for ii, pp in enumerate(pp_mc_list):
        M_tot = np.nansum(pp.sfh_mw_1D)
        M_cum = np.array([compute_cumulative_mass(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a, log_result=False) for a in ages[1:]])
        M_frac = M_cum / M_tot
        # Store in arrays
        M_tot_vals[ii] = M_tot
        M_cum_vals[ii] = M_cum
        M_frac_vals[ii, :] = M_frac
        ax.step(ages[:-1], M_frac, color="k", alpha=0.4, where="mid") if plotit else None

    # Compute mean/std values
    M_tot_MC_mean = np.nanmean(M_tot_vals)
    M_tot_MC_err = np.nanstd(M_tot_vals)
    M_frac_MC_mean = np.nanmean(M_frac_vals, axis=0)
    M_frac_MC_err = np.nanstd(M_frac_vals, axis=0)
    M_frac_MC_50th = np.nanquantile(a=M_frac_vals, axis=0, q=0.50)
    M_frac_MC_16th = np.nanquantile(a=M_frac_vals, axis=0, q=0.16)
    M_frac_MC_84th = np.nanquantile(a=M_frac_vals, axis=0, q=0.84)
    M_tot_regul = np.nansum(pp_regul.sfh_mw_1D)
    M_cum_regul = np.array([compute_cumulative_mass(pp_regul.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a, log_result=False) for a in ages[1:]])
    M_frac_regul = M_cum_regul / M_tot_regul

    if plotit:
        ax.step(ages[:-1], M_frac_MC_mean, linewidth=2.0, where="mid", color="b", label="MC")
        ax.step(ages[:-1], M_frac_regul, linewidth=2.0, where="mid", color="r", label="Regularised")
        ax.fill_between(x=ages[:-1], y1=M_frac_MC_mean - M_frac_MC_err, y2=M_frac_MC_mean + M_frac_MC_err, color="b", alpha=0.4, step="mid")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim([1e-6, None])
        ax.set_title("Cumulative Mass fraction")
        ax.set_ylabel("Cumulative mass fraction")
        ax.set_xlabel("Age")
        ax.legend()

    # Store in dict
    thisrow["Cumulative mass fraction vs. age cutoff (MC mean)"] = M_frac_MC_mean
    thisrow["Cumulative mass fraction vs. age cutoff (MC error)"] = M_frac_MC_err
    thisrow["Cumulative mass fraction vs. age cutoff (regularised)"] =  M_frac_regul

    # Cumulative light fraction (log)
    L_tot_vals = np.zeros(len(pp_mc_list))
    L_cum_vals = np.zeros((len(pp_mc_list), len(ages[:-1])))
    L_frac_vals = np.zeros((len(pp_mc_list), len(ages[:-1])))
    fig, ax = plt.subplots(figsize=(10, 5)) if plotit else (None, None)
    for ii, pp in enumerate(pp_mc_list):
        L_tot = np.nansum(pp.sfh_lw_1D)
        L_cum = np.array([compute_cumulative_light(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a, log_result=False) for a in ages[1:]])
        L_frac = L_cum / L_tot
        # Store in arrays
        L_tot_vals[ii] = L_tot
        L_cum_vals[ii] = L_cum
        L_frac_vals[ii, :] = L_frac
        ax.step(ages[:-1], L_frac, color="k", alpha=0.4, where="mid") if plotit else None

    # Compute mean/std values
    L_tot_MC_mean = np.nanmean(L_tot_vals)
    L_tot_MC_err = np.nanstd(L_tot_vals)
    L_tot_MC_50th = np.nanquantile(a=L_tot_vals, q=0.50)
    L_tot_MC_16th = np.nanquantile(a=L_tot_vals, q=0.16)
    L_tot_MC_84th = np.nanquantile(a=L_tot_vals, q=0.84)
    L_frac_MC_mean = np.nanmean(L_frac_vals, axis=0)
    L_frac_MC_err = np.nanstd(L_frac_vals, axis=0)
    L_frac_MC_50th = np.nanquantile(a=L_frac_vals, axis=0, q=0.50)
    L_frac_MC_16th = np.nanquantile(a=L_frac_vals, axis=0, q=0.16)
    L_frac_MC_84th = np.nanquantile(a=L_frac_vals, axis=0, q=0.84)
    L_tot_regul = np.nansum(pp_regul.sfh_lw_1D)
    L_cum_regul = np.array([compute_cumulative_light(pp_regul.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a, log_result=False) for a in ages[1:]])
    L_frac_regul = L_cum_regul / L_tot_regul

    if plotit:
        ax.step(ages[:-1], L_frac_MC_mean, linewidth=2.0, where="mid", color="b", label="MC")
        ax.step(ages[:-1], L_frac_regul, linewidth=2.0, where="mid", color="r", label="Regularised")
        ax.fill_between(x=ages[:-1], y1=L_frac_MC_mean - L_frac_MC_err, y2=L_frac_MC_mean + L_frac_MC_err, color="b", alpha=0.4, step="mid")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim([1e-6, None])
        ax.set_title("Cumulative light fraction")
        ax.set_ylabel("Cumulative light fraction")
        ax.set_xlabel("Age")
        ax.legend()

    # Store in dict
    thisrow["Cumulative light fraction vs. age cutoff (MC mean)"] = L_frac_MC_mean
    thisrow["Cumulative light fraction vs. age cutoff (MC error)"] = L_frac_MC_err
    thisrow["Cumulative light fraction vs. age cutoff (MC 50th percentile)"] = L_frac_MC_50th
    thisrow["Cumulative light fraction vs. age cutoff (MC 16th percentile)"] = L_frac_MC_16th
    thisrow["Cumulative light fraction vs. age cutoff (MC 84th percentile)"] = L_frac_MC_84th
    thisrow["Cumulative light fraction vs. age cutoff (regularised)"] = L_frac_regul

    # Mass-weighted age as a function of time (log)
    thisrow["Mass-weighted age vs. age cutoff (MC mean)"] = np.nanmean(np.array([[compute_mw_age(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Mass-weighted age vs. age cutoff (MC error)"] = np.nanstd(np.array([[compute_mw_age(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Mass-weighted age vs. age cutoff (MC 50th percentile)"] = np.nanquantile(a=np.array([[compute_mw_age(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), q=0.5, axis=0)
    thisrow["Mass-weighted age vs. age cutoff (MC 16th percentile)"] = np.nanquantile(a=np.array([[compute_mw_age(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), q=0.16, axis=0)
    thisrow["Mass-weighted age vs. age cutoff (MC 84th percentile)"] = np.nanquantile(a=np.array([[compute_mw_age(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), q=0.84, axis=0)
    thisrow["Mass-weighted age vs. age cutoff (regularised)"] = np.array([compute_mw_age(pp_regul.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]])

    # Light-weighted age as a function of time (log)
    thisrow["Light-weighted age vs. age cutoff (MC mean)"] = np.nanmean(np.array([[compute_lw_age(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Light-weighted age vs. age cutoff (MC error)"] = np.nanstd(np.array([[compute_lw_age(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), axis=0)
    thisrow["Light-weighted age vs. age cutoff (MC 50th percentile)"] = np.nanquantile(a=np.array([[compute_lw_age(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), q=0.5, axis=0)
    thisrow["Light-weighted age vs. age cutoff (MC 16th percentile)"] = np.nanquantile(a=np.array([[compute_lw_age(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), q=0.16, axis=0)
    thisrow["Light-weighted age vs. age cutoff (MC 84th percentile)"] = np.nanquantile(a=np.array([[compute_lw_age(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]] for pp in pp_mc_list]), q=0.84, axis=0)
    thisrow["Light-weighted age vs. age cutoff (regularised)"] = np.array([compute_lw_age(pp_regul.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=a)[0] for a in ages[1:]])

    # Plot the age distributions overlaid with quantiles to see what they look like & check that everything looks right 
    if plotit:
        for age_idx in [28, 48, len(ages) - 2]:
            mw_ages = [compute_mw_age(pp.sfh_mw_1D, isochrones=isochrones, age_thresh_upper=ages[age_idx])[0] for pp in pp_mc_list]
            if not all(np.isnan(mw_ages)):
                axs_hist.flat[ax_iter].hist(mw_ages, bins=15)
            axs_hist.flat[ax_iter].axvline(thisrow["Mass-weighted age vs. age cutoff (MC 16th percentile)"][age_idx - 1], ls="--", color="grey")
            axs_hist.flat[ax_iter].axvline(thisrow["Mass-weighted age vs. age cutoff (MC 50th percentile)"][age_idx - 1], color="grey")
            axs_hist.flat[ax_iter].axvline(thisrow["Mass-weighted age vs. age cutoff (MC 84th percentile)"][age_idx - 1], ls="--", color="grey")
            axs_hist.flat[ax_iter].axvline(thisrow["Mass-weighted age vs. age cutoff (MC mean)"][age_idx - 1], color="k")
            axs_hist.flat[ax_iter].axvline(thisrow["Mass-weighted age vs. age cutoff (regularised)"][age_idx - 1], color="red")
            axs_hist.flat[ax_iter].axvspan(xmin=thisrow["Mass-weighted age vs. age cutoff (MC mean)"][age_idx - 1] - thisrow["Mass-weighted age vs. age cutoff (MC error)"][age_idx - 1], 
                        xmax=thisrow["Mass-weighted age vs. age cutoff (MC mean)"][age_idx - 1] + thisrow["Mass-weighted age vs. age cutoff (MC error)"][age_idx - 1], color="pink", alpha=0.3)
            axs_hist.flat[ax_iter].set_title(f"Mass-weighted age below {ages[age_idx] / 1e9:.2f} Gyr")
            axs_hist.flat[ax_iter].set_xlabel("Mass-weighted age (log yr)")
            ax_iter += 1

            lw_ages = [compute_lw_age(pp.sfh_lw_1D, isochrones=isochrones, age_thresh_upper=ages[age_idx])[0] for pp in pp_mc_list]
            if not all(np.isnan(lw_ages)):
                axs_hist.flat[ax_iter].hist(lw_ages, bins=15)
            axs_hist.flat[ax_iter].axvline(thisrow["Light-weighted age vs. age cutoff (MC 16th percentile)"][age_idx - 1], ls="--", color="grey")
            axs_hist.flat[ax_iter].axvline(thisrow["Light-weighted age vs. age cutoff (MC 50th percentile)"][age_idx - 1], color="grey")
            axs_hist.flat[ax_iter].axvline(thisrow["Light-weighted age vs. age cutoff (MC 84th percentile)"][age_idx - 1], ls="--", color="grey")
            axs_hist.flat[ax_iter].axvline(thisrow["Light-weighted age vs. age cutoff (MC mean)"][age_idx - 1], color="k")
            axs_hist.flat[ax_iter].axvline(thisrow["Light-weighted age vs. age cutoff (regularised)"][age_idx - 1], color="red")
            axs_hist.flat[ax_iter].axvspan(xmin=thisrow["Light-weighted age vs. age cutoff (MC mean)"][age_idx - 1] - thisrow["Light-weighted age vs. age cutoff (MC error)"][age_idx - 1], 
                        xmax=thisrow["Light-weighted age vs. age cutoff (MC mean)"][age_idx - 1] + thisrow["Light-weighted age vs. age cutoff (MC error)"][age_idx - 1], color="pink", alpha=0.3)
            axs_hist.flat[ax_iter].set_title(f"Light-weighted age below {ages[age_idx] / 1e9:.2f} Gyr")
            axs_hist.flat[ax_iter].set_xlabel("Light-weighted age (log yr)")
            ax_iter += 1

    # Extinction and/or polynomial fits
    if pp_regul.reddening is not None:
        thisrow["A_V (MC mean)"] = R_V * np.nanmean([pp.reddening for pp in pp_mc_list])
        thisrow["A_V (MC error)"] = R_V * np.nanstd([pp.reddening for pp in pp_mc_list])
        thisrow["A_V (MC 50th percentile)"] = R_V * np.nanquantile(a=[pp.reddening for pp in pp_mc_list], q=0.5)
        thisrow["A_V (MC 16th percentile)"] = R_V * np.nanquantile(a=[pp.reddening for pp in pp_mc_list], q=0.16)
        thisrow["A_V (MC 84th percentile)"] = R_V * np.nanquantile(a=[pp.reddening for pp in pp_mc_list], q=0.84)
        thisrow["A_V (regularised)"] = pp_regul.reddening * R_V
        thisrow["10^-0.4A(lambda) (MC mean)"] = np.nanmean(np.array([pp.mpoly for pp in pp_mc_list]), axis=0)
        thisrow["10^-0.4A(lambda) (MC error)"] = np.nanstd(np.array([pp.mpoly for pp in pp_mc_list]), axis=0)
        thisrow["10^-0.4A(lambda) (MC 50th percentile)"] = np.nanquantile(a=np.array([pp.mpoly for pp in pp_mc_list]), q=0.5, axis=0)
        thisrow["10^-0.4A(lambda) (MC 16th percentile)"] = np.nanquantile(a=np.array([pp.mpoly for pp in pp_mc_list]), q=0.16, axis=0)
        thisrow["10^-0.4A(lambda) (MC 84th percentile)"] = np.nanquantile(a=np.array([pp.mpoly for pp in pp_mc_list]), q=0.84, axis=0)
        thisrow["10^-0.4A(lambda) (regularised)"] = pp_regul.mpoly

        # Plot the A_V distribution overlaid w/ quantiles etc. to check that it looks right
        if plotit:
            a_v_vals = R_V * np.array([pp.reddening for pp in pp_mc_list])
            if not all(np.isnan(a_v_vals)):
                axs_hist.flat[ax_iter].hist(a_v_vals, bins=15)
            axs_hist.flat[ax_iter].axvline(thisrow["A_V (MC 16th percentile)"], ls="--", color="grey")
            axs_hist.flat[ax_iter].axvline(thisrow["A_V (MC 50th percentile)"], color="grey")
            axs_hist.flat[ax_iter].axvline(thisrow["A_V (MC 84th percentile)"], ls="--", color="grey")
            axs_hist.flat[ax_iter].axvline(thisrow["A_V (MC mean)"], color="k")
            axs_hist.flat[ax_iter].axvline(thisrow["A_V (regularised)"], color="red")
            axs_hist.flat[ax_iter].axvspan(xmin=thisrow["A_V (MC mean)"] - thisrow["A_V (MC error)"], 
                        xmax=thisrow["A_V (MC mean)"] + thisrow["A_V (MC error)"], color="pink", alpha=0.3)
            axs_hist.flat[ax_iter].set_title(r"$A_V$")
            axs_hist.flat[ax_iter].set_xlabel(r"$A_V$ (mag)")
            ax_iter += 1

    else:
        if pp_regul.mpoly is not None:
            thisrow["Multiplicative polynomial (MC mean)"] = np.nanmean(np.array([pp.mpoly for pp in pp_mc_list]), axis=0)
            thisrow["Multiplicative polynomial (MC error)"] = np.nanstd(np.array([pp.mpoly for pp in pp_mc_list]), axis=0)
            thisrow["Multiplicative polynomial (MC 50th percentile)"] = np.nanquantile(a=np.array([pp.mpoly for pp in pp_mc_list]), q=0.5, axis=0)
            thisrow["Multiplicative polynomial (MC 16th percentile)"] = np.nanquantile(a=np.array([pp.mpoly for pp in pp_mc_list]), q=0.16, axis=0)
            thisrow["Multiplicative polynomial (MC 84th percentile)"] = np.nanquantile(a=np.array([pp.mpoly for pp in pp_mc_list]), q=0.84, axis=0)
            thisrow["Multiplicative polynomial (regularised)"] = pp_regul.mpoly
        elif pp_regul.apoly is not None:
            thisrow["Additive polynomial (MC mean)"] = np.nanmean(np.array([pp.apoly for pp in pp_mc_list]), axis=0)
            thisrow["Additive polynomial (MC error)"] = np.nanstd(np.array([pp.apoly for pp in pp_mc_list]), axis=0)
            thisrow["Additive polynomial (MC 50th percentile)"] = np.nanquantile(a=np.array([pp.apoly for pp in pp_mc_list]), q=0.5, axis=0)
            thisrow["Additive polynomial (MC 16th percentile)"] = np.nanquantile(a=np.array([pp.apoly for pp in pp_mc_list]), q=0.16, axis=0)
            thisrow["Additive polynomial (MC 84th percentile)"] = np.nanquantile(a=np.array([pp.apoly for pp in pp_mc_list]), q=0.84, axis=0)
            thisrow["Additive polynomial (regularised)"] = pp_regul.apoly
    thisrow["Wavelength (rest frame, Å, log-rebinned)"] = pp_regul.lam

    # AGN template weights 
    if pp_regul.fit_agn_cont:
        thisrow["ppxf alpha_nu_vals"] = pp_regul.alpha_nu_vals
        thisrow["AGN template weights (MC mean)"] = np.nanmean([pp.weights_agn for pp in pp_mc_list], axis=0)
        thisrow["AGN template weights (MC error)"] = np.nanstd([pp.weights_agn for pp in pp_mc_list], axis=0)
        thisrow["AGN template weights (MC 50th percentile)"] = np.nanquantile(a=[pp.weights_agn for pp in pp_mc_list], q=0.5, axis=0)
        thisrow["AGN template weights (MC 16th percentile)"] = np.nanquantile(a=[pp.weights_agn for pp in pp_mc_list], q=0.16, axis=0)
        thisrow["AGN template weights (MC 84th percentile)"] = np.nanquantile(a=[pp.weights_agn for pp in pp_mc_list], q=0.84, axis=0)
        thisrow["AGN template weights (regularised)"] = pp_regul.weights_agn
        
        # Express as a fraction of the total stellar weights (accounts for extinction)
        thisrow["x_AGN (total, MC mean)"] = np.nanmean([np.nansum(pp.weights_agn) / np.nansum(pp.weights_stellar) for pp in pp_mc_list])
        thisrow["x_AGN (total, MC error)"] = np.nanstd([np.nansum(pp.weights_agn) / np.nansum(pp.weights_stellar) for pp in pp_mc_list])
        thisrow["x_AGN (total, MC 50th percentile)"] = np.nanquantile(a=[np.nansum(pp.weights_agn) / np.nansum(pp.weights_stellar) for pp in pp_mc_list], q=0.50)
        thisrow["x_AGN (total, MC 16th percentile)"] = np.nanquantile(a=[np.nansum(pp.weights_agn) / np.nansum(pp.weights_stellar) for pp in pp_mc_list], q=0.16)
        thisrow["x_AGN (total, MC 84th percentile)"] = np.nanquantile(a=[np.nansum(pp.weights_agn) / np.nansum(pp.weights_stellar) for pp in pp_mc_list], q=0.84)
        thisrow["x_AGN (total, regularised)"] = np.nansum(pp_regul.weights_agn) / np.nansum(pp_regul.weights_stellar)
        
        # Plot x_AGN A_V distribution overlaid w/ quantiles etc. to check that it looks right
        if plotit:
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
        
        thisrow["x_AGN (individual, MC mean)"] = np.nanmean([pp.weights_agn / np.nansum(pp.weights_stellar) for pp in pp_mc_list], axis=0)
        thisrow["x_AGN (individual, MC error)"] = np.nanstd([pp.weights_agn / np.nansum(pp.weights_stellar) for pp in pp_mc_list], axis=0)
        thisrow["x_AGN (individual, MC 50th percentile)"] = np.nanquantile(a=[pp.weights_agn / np.nansum(pp.weights_stellar) for pp in pp_mc_list], axis=0, q=0.50)
        thisrow["x_AGN (individual, MC 16th percentile)"] = np.nanquantile(a=[pp.weights_agn / np.nansum(pp.weights_stellar) for pp in pp_mc_list], axis=0, q=0.16)
        thisrow["x_AGN (individual, MC 84th percentile)"] = np.nanquantile(a=[pp.weights_agn / np.nansum(pp.weights_stellar) for pp in pp_mc_list], axis=0, q=0.84)
        thisrow["x_AGN (individual, regularised)"] = pp_regul.weights_agn / np.nansum(pp_regul.weights_stellar)

    # Quality of fit 
    thisrow["Reduced-chi2 (MC mean)"] = np.nanmean([pp.chi2 for pp in pp_mc_list])
    thisrow["Reduced-chi2 (MC error)"] = np.nanstd([pp.chi2 for pp in pp_mc_list])
    thisrow["Reduced-chi2 (MC 50th percentile)"] = np.nanquantile(a=[pp.chi2 for pp in pp_mc_list], q=0.5)
    thisrow["Reduced-chi2 (MC 16th percentile)"] = np.nanquantile(a=[pp.chi2 for pp in pp_mc_list], q=0.16)
    thisrow["Reduced-chi2 (MC 84th percentile)"] = np.nanquantile(a=[pp.chi2 for pp in pp_mc_list], q=0.84)
    thisrow["Reduced-chi2 (regularised)"] = pp_regul.chi2
    if plotit:
        chi2_vals = np.array([pp.chi2 for pp in pp_mc_list])
        if not all(np.isnan(chi2_vals)):
            axs_hist.flat[ax_iter].hist(chi2_vals, bins=15)
        axs_hist.flat[ax_iter].axvline(thisrow["Reduced-chi2 (MC 16th percentile)"], ls="--", color="grey")
        axs_hist.flat[ax_iter].axvline(thisrow["Reduced-chi2 (MC 50th percentile)"], color="grey")
        axs_hist.flat[ax_iter].axvline(thisrow["Reduced-chi2 (MC 84th percentile)"], ls="--", color="grey")
        axs_hist.flat[ax_iter].axvline(thisrow["Reduced-chi2 (MC mean)"], color="k")
        axs_hist.flat[ax_iter].axvline(thisrow["Reduced-chi2 (regularised)"], color="red")
        axs_hist.flat[ax_iter].axvspan(xmin=thisrow["Reduced-chi2 (MC mean)"] - thisrow["Reduced-chi2 (MC error)"], xmax=thisrow["Reduced-chi2 (MC mean)"] + thisrow["Reduced-chi2 (MC error)"], color="pink", alpha=0.3)
        axs_hist.flat[ax_iter].set_title(r"$\chi^2/{\rm DOF}$")
        axs_hist.flat[ax_iter].set_xlabel(r"$\chi^2/{\rm DOF}$")
        ax_iter += 1

    if plotit and savefig:
        fig_hist.savefig(os.path.join("/priv/meggs3/u5708159/S7/mar23/ppxf/figs/", plot_fname), bbox_inches="tight", format="pdf")

    # Double-check that the quantile measurements are consistent with the means 
    def check_for_nans(r, col):
        if f"{col} (MC mean)" not in r:
            print(f"NaN check: {col} not found!")
            return
        mean_is_nan = np.isnan(r[f"{col} (MC mean)"])
        std_is_nan = np.isnan(r[f"{col} (MC error)"])
        _50th_is_nan = np.isnan(r[f"{col} (MC 50th percentile)"])
        _16th_is_nan = np.isnan(r[f"{col} (MC 16th percentile)"])
        _84th_is_nan = np.isnan(r[f"{col} (MC 84th percentile)"])
        if type(mean_is_nan) == np.bool_:
            if not mean_is_nan == std_is_nan:
                print(f"NaN consistency error found in column {col} for {gal}!")
            if not mean_is_nan == _50th_is_nan:
                print(f"NaN consistency error found in column {col} for {gal}!")
            if not mean_is_nan == _16th_is_nan:
                print(f"NaN consistency error found in column {col} for {gal}!")
            if not mean_is_nan == _84th_is_nan:
                print(f"NaN consistency error found in column {col} for {gal}!")
        else:
            if not all(mean_is_nan == std_is_nan):
                print(f"NaN consistency error found in column {col} for {gal}!")
            if not all(mean_is_nan == _50th_is_nan):
                print(f"NaN consistency error found in column {col} for {gal}!")
            if not all(mean_is_nan == _16th_is_nan):
                print(f"NaN consistency error found in column {col} for {gal}!")
            if not all(mean_is_nan == _84th_is_nan):
                print(f"NaN consistency error found in column {col} for {gal}!")
        print(f"NaN check: {col} is OK for {gal}")
    for col in ["Stellar template weights", "v_*", "sigma_*", "v_*", "sigma_*", "v_*", "sigma_*", "F_gas erg/s", "v_gas", "sigma_gas", "SFH LW 1D", "SFH MW 1D", "Cumulative mass vs. age cutoff", "Cumulative light vs. age cutoff", "Cumulative light fraction vs. age cutoff", "Mass-weighted age vs. age cutoff", "Light-weighted age vs. age cutoff", "Mass-weighted age vs. age cutoff", "Light-weighted age vs. age cutoff", "A_V", "10^-0.4A(lambda)", "A_V", "Multiplicative polynomial", "Additive polynomial", "AGN template weights",]:
        check_for_nans(thisrow, col)

    return thisrow

##############################################################################
# START FUNCTION DEFINITION
##############################################################################
def run_ppxf(spec, spec_err, lambda_vals_A, z, 
             isochrones, metals_to_use=None,
             fit_agn_cont=False, alpha_nu_vals=[0.5, 1.0, 1.5, 2.0],
             fit_gas=True, ngascomponents=0, tie_balmer=False,
             mdegree=4, adegree=-1, reddening=None,
             FWHM_inst_A=FWHM_WIFES_INST_A,
             bad_pixel_ranges_A=[],
             lambda_norm_A=4020,
             regularisation_method="auto", regul_fixed=0, 
             auto_adjust_regul=False, regul_nthreads=20,
             clean=False,
             regul_start=0, regul_span=1e4,
             delta_regul_min=1, regul_max=10e4, delta_delta_chi2_min=1,
             interactive_mode=False,
             plotit=False, savefigs=False, fname_str="ppxftests",
             reg_dim=None):
    """
    Wrapper function for calling ppxf.

    Inputs:
    spec                Input spectrum to fit to, on a linear wavelength scale
    spec_err            Corresponding 1-sigma errors 
    lambda_vals_A       Wavelength values, in Angstroms
    FWHM_inst_A         Instrumental resoltuion in Angstroms - defaults to WiFeS 
                        instrumental resolution for the B3000 grating
    z                   Galaxy redshift
    bad_pixel_ranges_A  Spectral regions to mask out (in Angstroms). 
                        format: [[lambda_1, lambda_2], ...]
    ngascomponents      Number of kinematic components to be fitted to the 
                        emission lines
    fit_gas             Whether to fit emission lines 
    mdegree             Degree of multiplicative polynomial to use in the fit.
                        Should be -1 for kinematic fitting
    adegree             Degree of additive polynomial to use in the fit.
                        Should be -1 for age + metallicity fitting
    reddening           Initial estimate of the E(B - V) >= 0 to be used to 
                        simultaneously fit a reddening curve to the stellar
                        continuum. NOTE: this cannot be used simultaneously
                        with mdegree.
    fit_agn_cont        Whether to include a set of AGN power-law continuum
                        templates in the fit. NOTE: for convenience, these use 
                        the "sky" input argument to ppxf - which means that 
                        if gas_reddening is used (instead of mpoly), ppxf will
                        NOT apply an extinction correction to the templates.
    alpha_nu_vals       Range of exponents to use in the power-law continuum.
    isochrones          Set of isochrones to use - must be Padova or Geneva
    tie_balmer          If true, use the Ha/Hb ratio to measure gas reddening.
    lambda_norm_A       Normalisation wavelength for computing the light-
                        ages.
    regularisation_method       
                        Method to use to determine the regularisation. 
                        Options: "none", "fixed", "auto", "interactive"
    regul_fixed         Value of regul to use if regularisation_method is 
                        "fixed"
    regul_nthreads      Number of threads to use for running simultaneous 
                        instances of ppxf if regularisation_method is "auto"
    delta_regul_min     Minimum spacing in regularisation before the program 
                        exits (make it a larger value to speed up execution)
                        if regularisation_method is "auto"
    regul_max           Maximum regularisation value before the program gives up 
                        trying to find the minimum in regul space (make it a 
                        smaller value to speed up execution) if 
                        regularisation_method is "auto"
    regul_start         Starting value for the regul parameter after the initial
                        regul = 0 run. The first array of regul values that 
                        will be run has start & endpoints
                            [regul_start, regul_start + regul_span]
    regul_span          Initial span of regul array.
    delta_delta_chi2_min  
                        Minimum value that Δχ (goal) - Δχ must reach to stop
                        execution if regularisation_method is "auto"

    plotit              Whether to show plots showing the evolution of the regul 
                        parameter & ppxf fits
    savefigs            If true, save figs to multi-page pdfs 
    fname_str           Filename prefix for figure pdfs: must include absolute 
                        path to directory.

    """
    np.seterr(divide = "ignore")

    assert regularisation_method in ["none", "fixed", "auto", "interactive"],\
        "regularisation_method must be one of 'none', 'auto' or 'interactive'!"

    if reddening is not None:
        assert mdegree == -1, "If the reddening keyword is used, mdegree must equal -1!"
    if mdegree != -1:
        assert reddening is None, "If mdegree != -1, then reddening must be None!"

    ##############################################################################
    # Set up plotting
    ##############################################################################
    if plotit and savefigs:
        print(f"WARNING: saving figures using prefix {fname_str}...")
        pdfpages_spec = PdfPages(f"{fname_str}_spec.pdf")
        pdfpages_regul = PdfPages(f"{fname_str}_regul.pdf")
        pdfpages_sfh = PdfPages(f"{fname_str}_sfh.pdf")

    ##############################################################################
    # Set up the input spectrum
    ##############################################################################
    lambda_start_A = lambda_vals_A[0]
    lambda_end_A = lambda_vals_A[-1]
    spec_linear = spec
    spec_err_linear = spec_err

    # Calculate the SNR
    SNR = np.nanmedian(spec_linear / spec_err_linear)
    # print("Median SNR = {:.4f}".format(SNR))

    # Rebin to a log scale
    spec_log, lambda_vals_log, velscale = util.log_rebin(
        np.array([lambda_start_A, lambda_end_A]), spec_linear)

    # Estimate the errors
    spec_err_log = log_rebin_errors(spec_linear, spec_err_linear, lambda_start_A, lambda_end_A)

    bad_px_mask = np.logical_or(spec_err_log <=0, np.isinf(spec_err_log))
    bad_px_mask = np.logical_or(bad_px_mask, spec_log < 0)
    bad_px_mask = np.logical_or(bad_px_mask, np.isnan(spec_err_log))

    # Mask out manually-defined negative values and problematic regions
    for r_A in bad_pixel_ranges_A:
        r1_A, r2_A = r_A
        r1_px = np.nanargmin(np.abs(np.exp(lambda_vals_log) - r1_A))
        r2_px = np.nanargmin(np.abs(np.exp(lambda_vals_log) - r2_A))
        bad_px_mask[r1_px:r2_px] = True
    good_px = np.squeeze(np.argwhere(~bad_px_mask))

    # For debugging: check the bad pixel ranges
    if plotit:
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(np.exp(lambda_vals_log), spec_log, color="k")
        for r_A in bad_pixel_ranges_A:
            r1_A, r2_A = r_A
            ax.axvspan(xmin=r1_A, xmax=r2_A, color="orange", alpha=0.3)
        ax.set_title("Bad pixel ranges")
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel(r"Flux ($F_\lambda$)")

    # Normalize spectrum to avoid numerical issues
    lambda_norm_idx = np.nanargmin(np.abs(np.exp(lambda_vals_log) / (1 + z) - lambda_norm_A))
    norm = spec_log[lambda_norm_idx]
    spec_err_log /= norm
    spec_log /= norm    
    spec_err_log[spec_err_log <= 0] = 99999
    spec_err_log[np.isnan(spec_err_log)] = 99999
    spec_err_log[np.isinf(spec_err_log)] = 99999

    vel = constants.c / 1e3 * np.log(1 + z) # Starting guess for systemic velocity (eq.(8) of Cappellari (2017))

    ##############################################################################
    # ppxf parameters
    ##############################################################################
    if isochrones == "Padova":
        if metals_to_use is None:
            metals_to_use = ['004', '008', '019']
        else:
            for m in metals_to_use:
                assert m in ['004', '008', '019'],\
                    f"Metallicity {m} for the {isochrones} isochrones not found!"

    elif isochrones == "Geneva":
        if metals_to_use is None:
            metals_to_use = ['001', '004', '008', '020', '040']
        else:
            for m in metals_to_use:
                assert m in ['001', '004', '008', '020', '040'],\
                    f"Metallicity {m} for the {isochrones} isochrones not found!"

    # pPXF parameters
    ncomponents = 1
    start_kin = [vel, 100.]
    nmoments = [2]
    if fit_gas:
        ncomponents += ngascomponents
        start_kin = [start_kin]
        start_kin += [[vel, 100]] * ngascomponents
        nmoments += [2] * ngascomponents
    limit_doublets = True

    ##############################################################################
    # SSP templates
    ##############################################################################
    stellar_templates_log, lambda_vals_ssp_log, metallicities, ages =\
        log_rebin_and_convolve_stellar_templates(isochrones, metals_to_use, 
                                                 FWHM_inst_A=FWHM_WIFES_INST_A, 
                                                 velscale=velscale)
    # Number of SSP templates
    n_ssp_templates = stellar_templates_log.shape[1]

    # Regularisation dimensions
    N_metallicities = len(metallicities)
    N_ages = len(ages)
    reg_dim = (N_metallicities, N_ages)

    # Normalise
    lambda_norm_idx = np.nanargmin(np.abs(np.exp(lambda_vals_ssp_log) - lambda_norm_A))
    stellar_template_norms = np.copy(stellar_templates_log)[lambda_norm_idx, :]
    stellar_templates_log /= stellar_template_norms

    # Reshape
    stellar_template_norms = np.reshape(stellar_template_norms, (N_metallicities, N_ages))

    # This line only works if velscale_ratio = 1
    dv = (lambda_vals_ssp_log[0] - lambda_vals_log[0]) * constants.c / 1e3  # km/s

    ##############################################################################
    # AGN templates
    ##############################################################################
    if fit_agn_cont:
        # Add power-law templates to replicate an AGN continuum
        n_agn_templates = len(alpha_nu_vals)
        agn_templates = np.zeros((len(lambda_vals_ssp_log), n_agn_templates))
        for aa, alpha_nu in enumerate(alpha_nu_vals):
            alpha_lambda = 2 - alpha_nu
            F_lambda_0 = 1 / (lambda_norm_A**(-alpha_lambda))
            F_lambda = F_lambda_0 * np.exp(lambda_vals_ssp_log)**(-alpha_lambda)
            agn_templates[:, aa] = F_lambda
        stellar_templates_log = np.column_stack([stellar_templates_log, agn_templates])
    else:
        n_agn_templates = 0

    n_ssp_and_agn_templates = stellar_templates_log.shape[1]

    ##############################################################################
    # Kinematic parameters, plus gas templates
    ##############################################################################
    cc = 0  # Component number
    kinematic_components = [cc] * n_ssp_and_agn_templates
    cc += 1

    if fit_gas:
        ##############################################################################
        # Gas templates
        ##############################################################################
        # Construct a set of Gaussian emission line templates
        # Estimate the wavelength fitted range in the rest frame.
        gas_templates, gas_names, eline_lambdas = util.emission_lines(
            logLam_temp=lambda_vals_ssp_log,
            lamRange_gal=np.array([lambda_start_A, lambda_end_A]) / (1 + z),
            FWHM_gal=FWHM_inst_A,
            tie_balmer=tie_balmer,
            limit_doublets=limit_doublets,
            vacuum=False
        )
        
        # forbidden lines contain "[*]"
        n_forbidden_lines = np.sum(["[" in a for a in gas_names])
        n_balmer_lines = len(gas_names) - n_forbidden_lines

        # Here, we lump together the Balmer + forbidden lines into a single kinematic component
        for ii in range(ngascomponents):
            kinematic_components += [cc] * len(gas_names)
            cc += 1

        # If the Balmer lines are tied one should allow for gas reddeining.
        # The gas_reddening can be different from the stellar one, if both are fitted.
        gas_reddening = 0 if tie_balmer else None

        # Combines the stellar and gaseous stellar_templates_log into a single array.
        # During the PPXF fit they will be assigned a different kinematic
        # COMPONENT value
        if ngascomponents >= 2:
            gas_templates = np.concatenate(tuple(gas_templates for ii in range(ngascomponents)), axis=1)
            gas_names = np.concatenate(tuple(gas_names for ii in range(ngascomponents)))

        gas_names_new = []
        for ii in range(len(gas_names)):
            gas_names_new.append(f"{gas_names[ii]} (component {kinematic_components[ii + n_ssp_and_agn_templates]})")
        gas_names = gas_names_new
        templates = np.column_stack([stellar_templates_log, gas_templates])

        # Determine which templates are gas ones
        gas_component = np.array(kinematic_components) > 0

    # Case: no gas templates
    else:
        gas_names = None 
        templates = stellar_templates_log
        gas_reddening = None
        gas_component = None 

    ##########################################################################
    # Mass-weighted weights
    ##########################################################################
    def compute_mass_weights(pp):
        # Reshape the normalisation factors into the same shape as the ppxf weights
        weights_light_weighted_normed = pp.weights

        if fit_agn_cont:
            weights_light_weighted_normed = np.reshape(
                weights_light_weighted_normed[~pp.gas_component][:-n_agn_templates], (N_metallicities, N_ages))
        else:
            weights_light_weighted_normed = np.reshape(
                weights_light_weighted_normed[~pp.gas_component], (N_metallicities, N_ages))

        # Convert the light-weighted ages into mass-weighted ages
        weights_mass_weighted = weights_light_weighted_normed * norm / stellar_template_norms

        return weights_mass_weighted

    ##########################################################################
    # light-weighted weights
    ##########################################################################
    def compute_light_weights(pp):
        # Reshape the normalisation factors into the same shape as the ppxf weights
        # Un-normalise the weights so that they are in units of Msun/(erg/s/Å) 
        # at the normalisation wavelength (by default 4020 Å)
        weights_light_weighted = pp.weights * norm

        if fit_agn_cont:
            weights_light_weighted = np.reshape(
                weights_light_weighted[~pp.gas_component][:-n_agn_templates], (N_metallicities, N_ages))
        else:
            weights_light_weighted = np.reshape(
                weights_light_weighted[~pp.gas_component], (N_metallicities, N_ages))

        return weights_light_weighted

    ##############################################################################
    # Wrapper for plotting
    ##############################################################################
    def plot_wrapper(pp, ages, metallicities):

        # Comptue the mass weights 
        N_ages = len(ages)
        N_metallicities = len(metallicities)
        weights_mass_weighted = compute_mass_weights(pp)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 6.5))
        ax_spec, ax_hist = axs
        bbox = ax_hist.get_position()
        cbarax = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.03, bbox.height])

        # Histogram
        cmap = matplotlib.cm.get_cmap("viridis_r").copy()
        cmap.set_bad("#DADADA")
        m = ax_hist.imshow(np.log10(weights_mass_weighted), cmap=cmap, origin="lower", aspect="auto", vmin=0, vmax=np.nanmax(np.log10(weights_mass_weighted)))
        fig.colorbar(m, cax=cbarax)
        ax_hist.set_yticks(range(len(metallicities)))
        ax_hist.set_yticklabels(["{:.3f}".format(met / 0.02)
                                 for met in metallicities])
        ax_hist.set_ylabel(r"Metallicity ($Z_\odot$)")
        cbarax.set_ylabel(r"Mass $\log_{10}(\rm M_\odot)$")
        ax_hist.set_xticks(range(len(ages)))
        ax_hist.set_xlabel("Age (Myr)")
        ax_hist.set_title("Best fit star formation history")
        ax_hist.set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")

        # Kinematic and age & metallicity fits
        ax_spec.clear()
        ppxf_plot(pp, ax_spec)
        ax_spec.set_title("ppxf fit")
        ax_spec.set_xticklabels([])
        ax_spec.set_xlabel("")
        ax_spec.autoscale(axis="x", enable=True, tight=True)

        fig.canvas.draw()
        if matplotlib.get_backend() != "agg":
            plt.show()

    ##########################################################################
    # Use pPXF to obtain the stellar age + metallicity, and fit emission lines
    ##########################################################################
    delta_chi2_ideal = np.sqrt(2 * len(good_px))
    if regularisation_method == "none":
        # If no regularisation is required, then just run once and exit.
        noise_scaling_factor = 1
        pp = ppxf(templates=templates,
                  galaxy=spec_log, noise=spec_err_log,
                  velscale=np.squeeze(velscale), start=start_kin,
                  goodpixels=good_px,
                  moments=nmoments, degree=adegree, mdegree=mdegree,
                  vsyst=dv,
                  lam=np.exp(lambda_vals_log),
                  regul=0,
                  clean=clean,
                  reddening=reddening, reddening_func=reddening_calzetti00,
                  reg_dim=reg_dim,
                  component=kinematic_components, gas_component=gas_component,
                  gas_names=gas_names, gas_reddening=gas_reddening, method="capfit",
                  quiet=True)
    else:
        # Otherwise, run ppxf the first time w/o regularisation. 
        ws = []  # List to store template weights from successive iterations 
        t = time()
        pp_age_met = ppxf(templates=templates,
                          galaxy=spec_log, noise=spec_err_log,
                          velscale=np.squeeze(velscale), start=start_kin,
                          goodpixels=good_px,
                          moments=nmoments, degree=adegree, mdegree=mdegree,
                          vsyst=dv,
                          lam=np.exp(lambda_vals_log),
                          regul=0,
                          clean=clean,
                          reddening=reddening, reddening_func=reddening_calzetti00,
                          reg_dim=reg_dim,
                          component=kinematic_components, gas_component=gas_component,
                          gas_names=gas_names, gas_reddening=gas_reddening, method="capfit",
                          quiet=True)
        delta_chi2 = (pp_age_met.chi2 - 1) * len(good_px)
        print("----------------------------------------------------")
        print(F"Iteration 0: Elapsed time in PPXF (single thread): {time() - t:.2f} s")

        # Get the template weights from this fit 
        w = compute_mass_weights(pp_age_met)
        ws.append(w)
        if w.shape[0] > 1:
            w = np.nansum(w, axis=0)
        else:
            w = w.squeeze()

        if plotit:
            # plt.close("all")
            plot_wrapper(pp_age_met, ages, metallicities)
            fig = plt.gcf()
            fig.suptitle(f"First ppxf iteration: regul = 0")
            if savefigs:
                pdfpages_spec.savefig(fig, bbox_inches="tight")

            if matplotlib.get_backend() != "agg" and interactive_mode:
                hit_key_to_continue()

        if regularisation_method == "fixed":
            # Run ppxf one more time with a fixed regul factor, after scaling
            # the noise vector appropriately
            print(F"Iteration 1: Scaling noise by {np.sqrt(pp_age_met.chi2):.4f}...")
            noise_scaling_factor = np.sqrt(pp_age_met.chi2)
            pp = ppxf(templates=templates,
                              galaxy=spec_log, noise=spec_err_log * noise_scaling_factor,
                              velscale=np.squeeze(velscale), start=start_kin,
                              goodpixels=good_px,
                              moments=nmoments, degree=adegree, mdegree=mdegree,
                              vsyst=dv,
                              lam=np.exp(lambda_vals_log),
                              regul=regul_fixed,
                              clean=clean,
                              reddening=reddening, reddening_func=reddening_calzetti00,
                              reg_dim=reg_dim,
                              component=kinematic_components, gas_component=gas_component,
                              gas_names=gas_names, gas_reddening=gas_reddening, method="capfit")
            delta_chi2 = (pp.chi2 - 1) * len(good_px)
            print("----------------------------------------------------")
            print(F"Desired Delta Chi^2: {delta_chi2_ideal:.4g}")
            print(F"Current Delta Chi^2: {delta_chi2:.4g}")
            print(F"Delta-Delta Chi^2: {np.abs(delta_chi2 - delta_chi2_ideal):.4g}")
            print("----------------------------------------------------")
            print(F"Elapsed time in PPXF: {time() - t:.2f} s")

        elif regularisation_method == "interactive":
            # Run again but with regularization.
            print(F"Iteration 0: Scaling noise by {np.sqrt(pp_age_met.chi2):.4f}...")
            noise_scaling_factor = np.sqrt(pp_age_met.chi2)
            cnt = 1

            # Manually select the regul parameter value.
            while True:
                key = input("Please enter a value for regul: ")
                if key.isdigit():
                    regul = float(key)
                    break

            while True:
                t = time()
                pp_age_met = ppxf(templates=templates,
                                  galaxy=spec_log, noise=spec_err_log * noise_scaling_factor,
                                  velscale=np.squeeze(velscale), start=start_kin,
                                  goodpixels=good_px,
                                  moments=nmoments, degree=adegree, mdegree=mdegree,
                                  vsyst=dv,
                                  lam=np.exp(lambda_vals_log),
                                  regul=regul,
                                  clean=clean,
                                  reddening=reddening, reddening_func=reddening_calzetti00,
                                  reg_dim=reg_dim,
                                  component=kinematic_components, gas_component=gas_component,
                                  gas_names=gas_names, gas_reddening=gas_reddening, method="capfit")
                delta_chi2 = (pp_age_met.chi2 - 1) * len(good_px)
                print("----------------------------------------------------")
                print(F"Desired Delta Chi^2: {delta_chi2_ideal:.4g}")
                print(F"Current Delta Chi^2: {delta_chi2:.4g}")
                print("----------------------------------------------------")
                print(F"Elapsed time in PPXF: {time() - t:.2f} s")

                # Compute the difference in the best fit between this & the last iteration
                w = compute_mass_weights(pp_age_met)
                ws.append(w)
                if w.shape[0] > 1:
                    w = np.nansum(w, axis=0)
                    dw = np.abs(w - np.nansum(ws[-2], axis=0))
                else:
                    w = w.squeeze()
                    dw = np.abs(w - ws[-2].squeeze())
                delta_m = np.nansum(dw)

                if plotit:
                    # plt.close("all")
                    plot_wrapper(pp_age_met, ages, metallicities)
                    fig = plt.gcf()
                    fig.suptitle(f"Manually determining regul parameter: regul = {regul:.2f} (iteration {cnt})")
                    if savefigs:
                        pdfpages_spec.savefig(fig, bbox_inches="tight")

                while True:
                    key = input("Enter a new regul value, otherwise press enter: ")
                    if key.isdigit() or key == "":
                        break
                if key == "":
                    break
                else:
                    regul = float(key)
                cnt += 1

        elif regularisation_method == "auto":
            # Run again but with regularization.
            cnt = 1
            print("----------------------------------------------------")
            print(f"Iteration {cnt}: Scaling noise by {np.sqrt(pp_age_met.chi2):.4f}...")
            noise_scaling_factor = np.sqrt(pp_age_met.chi2)

            # Run ppxf a number of times & find the value of regul that minimises 
            # the difference between the ideal delta-chi2 and the real delta-chi2.
            regul_vals = np.linspace(regul_start, regul_start + regul_span, regul_nthreads)
            delta_regul = np.diff(regul_vals)[0]
            print(f"Iteration {cnt}: Regularisation parameter range: {regul_vals[0]}-{regul_vals[-1]} (n = {len(regul_vals)})")

            obj_vals = []  # "objective" fn
            pps = []

            # Input arguments
            args_list = [
                [
                    templates, spec_log, spec_err_log, noise_scaling_factor,
                    velscale, start_kin, good_px, nmoments, adegree,
                    mdegree, dv, lambda_vals_log, regul, clean, reddening, reddening_calzetti00,
                    reg_dim, kinematic_components, gas_component, gas_names,
                    gas_reddening
                ] for regul in regul_vals
            ]

            # Run in parallel
            print(f"Iteration {cnt}: Running ppxf on {regul_nthreads} threads...")
            t = time()
            pool = multiprocessing.Pool(regul_nthreads)
            pps = list(pool.map(ppxf_helper, args_list))
            pool.close()
            pool.join()
            print(F"Iteration {cnt}: Elapsed time in PPXF (multithreaded): {time() - t:.2f} s")

            # Determine which is the optimal regul value
            # Quite certain this is correct - see here: https://pypi.org/project/ppxf/#how-to-set-regularization
            regul_vals = [p.regul for p in pps]  # Redefining as pool may not retain the order of the input list
            delta_chi2_vals = [(p.chi2 - 1) * len(good_px) for p in pps]
            obj_vals = [np.abs(delta_chi2 - delta_chi2_ideal) for delta_chi2 in delta_chi2_vals]
            opt_idx = np.nanargmin(obj_vals)

            # Compute the difference in the best fit between this & the last iteration
            w = compute_mass_weights(pps[opt_idx])
            ws.append(w)
            if w.shape[0] > 1:
                w = np.nansum(w, axis=0)
                dw = np.abs(w - np.nansum(ws[-2], axis=0))
            else:
                w = w.squeeze()
                dw = np.abs(w - ws[-2].squeeze())
            delta_m = np.nansum(dw)

            # Print info
            print(f"Iteration {cnt}: optimal regul = {regul_vals[opt_idx]:.2f}; Δm = {delta_m:g}; Δregul = {delta_regul:.2f} (Δregul_min = {delta_regul_min:.2f}); Δχ (goal) - Δχ = {obj_vals[np.nanargmin(obj_vals)]:.3f}")

            if plotit:
                # Plot the best fit
                plot_wrapper(pps[opt_idx], ages, metallicities)
                fig = plt.gcf()
                fig.suptitle(f"Automatically determining regul parameter: regul = {regul_vals[opt_idx]:.2f} (iteration {1})")
                if savefigs:
                    pdfpages_spec.savefig(fig, bbox_inches="tight")

                # Plot the regul values
                fig_regul, ax_regul = plt.subplots(nrows=1, ncols=1)
                ax_regul.plot(regul_vals, obj_vals, "bo")
                ax_regul.plot(regul_vals[np.nanargmin(obj_vals)], obj_vals[np.nanargmin(obj_vals)], "ro", label="Optimal fit")
                ax_regul.axhline(0, color="gray")
                ax_regul.set_title(f"Regularisation determination (iteration {1})")
                ax_regul.set_xlabel("Regularisation parameter")
                ax_regul.set_ylabel(r"$\Delta\chi_{\rm goal}^2 - \Delta\chi^2$")
                ax_regul.legend()
                fig_regul.canvas.draw()
                if savefigs:
                    pdfpages_regul.savefig(fig_regul, bbox_inches="tight")

                if matplotlib.get_backend() != "agg" and interactive_mode:
                    hit_key_to_continue()

            ###########
            while True:
                # Once the regul sampling reachs 1, quit 
                print("----------------------------------------------------")
                if delta_regul <= delta_regul_min:
                    print(f"STOPPING: Minimum spacing between regul values reached; using {regul_vals[opt_idx]:.2f} to produce the best fit")
                    break
                elif obj_vals[opt_idx] < delta_delta_chi2_min:
                    print(f"STOPPING: Convergence criterion reached; Δχ (goal) - Δχ = {obj_vals[opt_idx]}; using {regul_vals[opt_idx]:.2f} to produce the best fit")
                    break
                elif opt_idx == len(regul_vals) - 1 and regul_vals[opt_idx] >= regul_max:
                    print(f"STOPPING: Optimal regul value is >= {regul_max}; using {regul_vals[opt_idx]:.2f} to produce the best fit")
                    break
                
                # If the lowest regul value is "maxed out" then try again with an array starting at the highest regul value
                elif regul_vals[opt_idx] == np.nanmax(regul_vals):
                    regul_span = np.nanmax(regul_vals) - np.nanmin(regul_vals)
                    regul_0 = np.nanmax(regul_vals) - delta_regul
                    regul_end = np.nanmax(regul_vals) - delta_regul + regul_span
                    regul_vals = np.linspace(regul_0, regul_end, regul_nthreads)
                
                # If the lowest regul value is "minned out" then try again with an array ending with the lowest regul value
                elif regul_vals[opt_idx] == np.nanmin(regul_vals):
                    regul_span = np.nanmax(regul_vals) - np.nanmin(regul_vals)
                    regul_0 = np.nanmin(regul_vals) + delta_regul - regul_span
                    regul_end = np.nanmin(regul_vals) + delta_regul
                
                # Otherwise, sub-sample the previous array windowed around the optimal value.
                else:
                    delta_regul /= 5
                    regul_0 = regul_vals[opt_idx] - regul_nthreads / 2 * delta_regul
                    regul_end = regul_vals[opt_idx] + regul_nthreads / 2 * delta_regul

                # Reset the values
                regul_0 = 0 if regul_0 < 0 else regul_0  # If regul_0 < 0, set the starting point to 0, since there's no point running ppxf for -ve values.
                regil_end = regul_max if regul_end > regul_max else regul_end
                regul_vals = np.linspace(regul_0, regul_end, regul_nthreads)
                delta_regul = np.diff(regul_vals)[0]

                # Re-run ppxf
                args_list = [
                    [
                        templates, spec_log, spec_err_log, noise_scaling_factor,
                        velscale, start_kin, good_px, nmoments, adegree,
                        mdegree, dv, lambda_vals_log, regul, clean, reddening, reddening_calzetti00,
                        reg_dim, kinematic_components, gas_component, gas_names,
                        gas_reddening
                    ] for regul in regul_vals
                ]

                # Run in parallel
                cnt += 1
                print(f"Iteration {cnt}: Re-running ppxf on {regul_nthreads} threads (iteration {cnt})...")
                print(f"Iteration {cnt}: Regularisation parameter range: {regul_vals[0]}-{regul_vals[-1]} (n = {len(regul_vals)})")
                t = time()
                pool = multiprocessing.Pool(regul_nthreads)
                pps = list(pool.map(ppxf_helper, args_list))
                pool.close()
                pool.join()
                print(F"Iteration {cnt}: Elapsed time in PPXF (multithreaded): {time() - t:.2f} s")

                # Determine which is the optimal regul value
                regul_vals = [p.regul for p in pps]  # Redefining as pool may not retain the order of the input list
                delta_chi2_vals = [(p.chi2 - 1) * len(good_px) for p in pps]
                obj_vals = [np.abs(delta_chi2 - delta_chi2_ideal) for delta_chi2 in delta_chi2_vals]
                opt_idx = np.nanargmin(obj_vals)
                
                # Compute the difference in the best fit between this & the last iteration
                w = compute_mass_weights(pps[opt_idx])
                ws.append(w)
                if w.shape[0] > 1:
                    w = np.nansum(w, axis=0)
                    dw = np.abs(w - np.nansum(ws[-2], axis=0))
                else:
                    w = w.squeeze()
                    dw = np.abs(w - ws[-2].squeeze())
                delta_m = np.nansum(dw)

                # Print info
                print(f"Iteration {cnt}: optimal regul = {regul_vals[opt_idx]:.2f}; Δm = {delta_m:g}; Δregul = {delta_regul:.2f} (Δregul_min = {delta_regul_min:.2f}); Δχ (goal) - Δχ = {obj_vals[np.nanargmin(obj_vals)]:.3f}")

                # Plot the best fit
                if plotit:
                    plot_wrapper(pps[opt_idx], ages, metallicities)
                    fig = plt.gcf()
                    fig.suptitle(f"Automatically determining regul parameter: regul = {regul_vals[opt_idx]:.2f} (iteration {cnt})")
                    if savefigs:
                        pdfpages_spec.savefig(fig, bbox_inches="tight")

                    # Plot the regul values
                    fig_regul, ax_regul = plt.subplots(nrows=1, ncols=1)
                    ax_regul.plot(regul_vals, obj_vals, "bo")
                    ax_regul.plot(regul_vals[np.nanargmin(obj_vals)], obj_vals[np.nanargmin(obj_vals)], "ro", label="Optimal fit")
                    ax_regul.axhline(0, color="gray")
                    ax_regul.set_title(f"Regularisation determination (iteration {cnt})")
                    ax_regul.set_xlabel("Regularisation parameter")
                    ax_regul.set_ylabel(r"$\Delta\chi_{\rm goal}^2 - \Delta\chi^2$")
                    ax_regul.legend()

                    if savefigs:
                        pdfpages_regul.savefig(fig_regul, bbox_inches="tight")

                    if matplotlib.get_backend() != "agg" and interactive_mode:
                        hit_key_to_continue()

            pp = pps[opt_idx]

    ##########################################################################
    # Add some extra useful stuff to the ppxf instance
    ##########################################################################
    # Template ages and metallicities
    pp.isochrones = isochrones
    pp.ages = ages 
    pp.metallicities = metallicities

    pp.spec_norm = spec
    pp.spec_log_norm = spec_log

    # Convergence criteria 
    pp.noise_scaling_factor = noise_scaling_factor
    pp.regularisation_method = regularisation_method
    pp.delta_chi2_ideal = delta_chi2_ideal
    pp.delta_chi2 = (pp.chi2 - 1) * len(good_px)
    pp.delta_delta_chi2 = np.abs(pp.delta_chi2 - pp.delta_chi2_ideal)

    # Mass & light-weighted template weights 
    pp.weights_mass_weighted = compute_mass_weights(pp)  # Mass-weighted template weights
    pp.weights_light_weighted = compute_light_weights(pp)
    pp.norm = norm  # Normalisation factor for the logarithmically binned input spectrum
    pp.stellar_template_norms = stellar_template_norms  # Normalisation factors for the logarithmically binned stellar templates

    # Compute the mass- and light-weighted star formation history 
    pp.sfh_mw_1D = np.nansum(pp.weights_mass_weighted, axis=0)
    pp.sfh_lw_1D = np.nansum(pp.weights_light_weighted, axis=0)

    # Compute the mean SFR in each bin
    bin_edges, bin_widths = get_bin_edges_and_widths(isochrones)
    pp.sfr_mean = pp.sfh_mw_1D / bin_widths 

    # Gas parameters, if used
    if fit_gas:
        pp.fit_gas = True
        pp.ngascomponents = ngascomponents
        n_lines = n_balmer_lines + n_forbidden_lines
        # Sort the emission lines in order of narrowest ==> broadest
        sigma_vals = []
        v_vals = []
        F_vals = []
        for nn in range(ngascomponents):
            v, sigma = pp.sol[nn + 1]
            sigma_vals.append(sigma)
            v_vals.append(v)
            F_vals.append(pp.gas_flux[nn * n_lines:(nn + 1) * n_lines])
        sort_idx = np.argsort(sigma_vals)
        pp.v_gas = [v_vals[ii] for ii in sort_idx]
        pp.sigma_gas = [sigma_vals[ii] for ii in sort_idx]
        pp.F_gas = [F_vals[ii] for ii in sort_idx]
        pp.v_star = pp.sol[0][0]
        pp.sigma_star = pp.sol[0][1]
    else:
        pp.fit_gas = False
        pp.ngascomponents = 0
        pp.v_gas = None
        pp.sigma_gas = None
        pp.F_gas = None
        pp.v_star = pp.sol[0]
        pp.sigma_star = pp.sol[1]
    pp.tie_balmer = tie_balmer

    # AGN continuum parameters, if used 
    if fit_agn_cont:
        pp.fit_agn_cont = True
        pp.alpha_nu_vals = alpha_nu_vals
        pp.weights_agn = pp.weights[~pp.gas_component][-n_agn_templates:] 
        pp.weights_stellar = pp.weights[~pp.gas_component][:-n_agn_templates] 
    else:
        pp.fit_agn_cont = False
        pp.alpha_nu_vals = None 
        pp.weights_agn = None
        pp.agn_bestfit = None
        pp.weights_stellar = pp.weights[~pp.gas_component]

    ##########################################################################
    # Plotting the fit
    ##########################################################################
    if plotit:
        plot_wrapper(pp, ages, metallicities)
        fig = plt.gcf()
        if regularisation_method == "auto" or regularisation_method == "interactive": 
            regul_final = regul_vals[opt_idx]
        elif regularisation_method == "fixed":
            regul_final = regul_fixed
        elif regularisation_method == "none":
            regul_final = 0
        fig.suptitle(f"Best fit (regul = {regul_final:.2f})")
        if savefigs:
            pdfpages_spec.savefig(fig, bbox_inches="tight")
            pdfpages_spec.close()
            if regularisation_method != "none":
                pdfpages_regul.close()
            pdfpages_sfh.close()

    return pp

###############################################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###############################################################################
if __name__ == "__main__":
    from ppxftests.mockspec import create_mock_spectrum
    from ppxftests.ssputils import load_ssp_templates
    from ppxftests.sfhutils import plot_sfh_mass_weighted, load_sfh


    ###########################################################################
    # Mock spectra options
    ###########################################################################
    isochrones = "Padova"  # Set of isochrones to use 
    
    ###########################################################################
    # GALAXY PROPERTIES
    ###########################################################################
    z = 0.00               # Redshift 
    SNR = 100               # S/N ratio

    ###########################################################################
    # DEFINE THE SFH
    ###########################################################################
    # Idea 1: use a Gaussian kernel to smooth "delta-function"-like SFHs
    # Idea 2: are the templates logarithmically spaced in age? If so, could use e.g. every 2nd template 
    _, _, metallicities, ages = load_ssp_templates(isochrones)
    N_ages = len(ages)
    N_metallicities = len(metallicities)

    # sfh_input = np.zeros((N_metallicities, N_ages))
    # # sfh_input[1, 10] = 1e7
    # sfh_input[2, 4] = 1e10
    # sigma_star_kms = 200
    # plot_sfh_mass_weighted(sfh_input, ages, metallicities)

    # Realistic SFH
    gal = 10
    sfh_mw_input, sfh_lw_input, sfr_avg_input, sigma_star_kms = load_sfh(gal, plotit=True)

    ###########################################################################
    # CREATE THE MOCK SPECTRUM
    ###########################################################################
    spec, spec_err, lambda_vals_A = create_mock_spectrum(
        sfh_mass_weighted=sfh_mw_input,
        isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
        plotit=False)    

    # Add an AGN continuum
    lambda_norm_A = 4020
    lambda_norm_idx = np.nanargmin(np.abs(lambda_vals_A - lambda_norm_A))
    F_star_0 = spec[lambda_norm_idx]
    F_agn_0 = 0.5 * F_star_0
    alpha_lambda = 0.5
    F_lambda_0 = F_agn_0 / lambda_norm_A**(-alpha_lambda)
    F_agn = F_lambda_0 * lambda_vals_A**(-alpha_lambda)

    plt.figure()
    plt.plot(lambda_vals_A, spec)
    plt.plot(lambda_vals_A, F_agn)
    plt.plot(lambda_vals_A, spec + F_agn)
    plt.axvline(lambda_norm_A, color="k")

    ###########################################################################
    # RUN PPXF
    ###########################################################################
    pp_agn = run_ppxf(spec=spec + F_agn, spec_err=spec_err, lambda_vals_A=lambda_vals_A, 
                  z=z, ngascomponents=1,
                  regularisation_method="none",
                  fit_agn_cont=True,
                  plotit=True,
                  isochrones="Padova", tie_balmer=True)
    plt.gcf().suptitle("AGN templates included in fit")

    pp_noagn = run_ppxf(spec=spec + F_agn, spec_err=spec_err, lambda_vals_A=lambda_vals_A, 
                  z=z, ngascomponents=1,
                  regularisation_method="none",
                  fit_agn_cont=False,
                  plotit=True,
                  isochrones="Padova", tie_balmer=True)
    plt.gcf().suptitle("No AGN templates included in fit")

    # pp.sky stores the input sky templates
    # when sky is not None, the sky template weights are stored in
    #   pp.weights[pp.ntemp:]

    # sfh_fit = pp.weights_mass_weighted

    # ###########################################################################
    # # COMPARE THE INPUT AND OUTPUT
    # ###########################################################################
    # # side-by-side comparison of the SFHs, plus residual map
    # plot_sfh_mass_weighted(sfh_input, ages, metallicities)
    # plot_sfh_mass_weighted(sfh_fit, ages, metallicities)

    # # Plot the residual
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 3.5))
    # bbox = ax.get_position()
    # cax = fig.add_axes([bbox.x0 + bbox.width, bbox.x0, 0.025, bbox.height])
    
    # # Plot the SFH
    # delta_sfh = sfh_input - sfh_fit
    # m = ax.imshow(delta_sfh, cmap="coolwarm", 
    #               origin="lower", aspect="auto",
    #               vmin=-np.abs(np.nanmax(delta_sfh)), vmax=np.abs(np.nanmax(delta_sfh)))
    # fig.colorbar(m, cax=cax)
    
    # # Decorations
    # ax.set_yticks(range(len(metallicities)))
    # ax.set_yticklabels(["{:.3f}".format(met / 0.02) for met in metallicities])
    # ax.set_ylabel(r"Metallicity ($Z_\odot$)")
    # cax.set_ylabel(r"Residual (\rm M_\odot)$")
    # ax.set_xticks(range(len(ages)))
    # ax.set_xlabel("Age (Myr)")
    # ax.set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")
    # ax.autoscale(axis="x", enable=True, tight=True)
    # ax.grid()

