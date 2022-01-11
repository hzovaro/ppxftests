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

from scipy import constants, ndimage
import numpy as np
import extinction
from itertools import product
import multiprocessing
import pandas as pd

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util

from cosmocalc import get_dist
from log_rebin_errors import log_rebin_errors

from ppxftests.ssputils import load_ssp_templates
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
matplotlib.rc("font", **{"family": "serif"})
matplotlib.rc("image", interpolation="nearest")
matplotlib.rc("image", origin="lower")
plt.close("all")
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
# For using the FM07 reddening curve in ppxf
##############################################################################
def reddening_fm07(lam, ebv):
    # lam in Angstroms
    # Need to derive A(lambda) from E(B-V)
    # fm07 takes as input lambda and A_V, so we first need to convert E(B-V)
    # into A_V
    A_V = 3.1 * ebv
    A_lambda = extinction.fm07(lam, a_v=A_V, unit='aa')
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
# For running in parallel
##############################################################################
def ppxf_helper(args):
    # Parse arguments
    templates, spec_log, spec_err_log, noise_scaling_factor,\
        velscale, start_age_met, good_px, nmoments_age_met, adegree_age_met,\
        mdegree_age_met, dv, lambda_vals_log, regul, reddening_fm07,\
        reg_dim, kinematic_components, gas_component, gas_names,\
        gas_reddening = args

    # Run ppxf
    pp_age_met = ppxf(templates=templates,
          galaxy=spec_log, noise=spec_err_log * noise_scaling_factor,
          velscale=np.squeeze(velscale), start=start_age_met,
          goodpixels=good_px,
          moments=nmoments_age_met, degree=adegree_age_met, mdegree=mdegree_age_met,
          vsyst=dv,
          lam=np.exp(lambda_vals_log),
          regul=regul,
          reddening_func=reddening_fm07,
          reg_dim=reg_dim,
          component=kinematic_components, gas_component=gas_component,
          gas_names=gas_names, gas_reddening=gas_reddening, method="capfit",
          quiet=True)

    # Return
    return pp_age_met

##############################################################################
# START FUNCTION DEFINITION
##############################################################################
def run_ppxf(spec, spec_err, lambda_vals_A,
             z, ngascomponents,
             tie_balmer,
             isochrones, metals_to_use=None,
             fit_gas=True,
             FWHM_inst_A=FWHM_WIFES_INST_A,
             bad_pixel_ranges_A=[],
             regularisation_method="auto",
             auto_adjust_regul=False, regul_nthreads=20,
             delta_regul_min=5, regul_max=1e4,
             delta_delta_chi2_min=1,
             interactive_mode=False,
             plotit=True, savefigs=False, fname_str="ppxftests"):
    """
    Wrapper function for calling ppxf.

    Inputs:
    spec                Input spectrum to fit to, on a linear wavelength scale
    spec_err            Corresponding 1-sigma_diff_px errors 
    lambda_vals_A       Wavelength values, in Angstroms
    FWHM_inst_A         Instrumental resoltuion in Angstroms - defaults to WiFeS 
                        instrumental resolution for the B3000 grating
    z                   Galaxy redshift
    bad_pixel_ranges_A  Spectral regions to mask out (in Angstroms). 
                        format: [[lambda_1, lambda_2], ...]
    ngascomponents      Number of kinematic components to be fitted to the 
                        emission lines
    
    isochrones          Set of isochrones to use - must be Padova or Geneva
    tie_balmer          If true, use the Ha/Hb ratio to measure gas reddening.
    regularisation_method       
                        Method to use to determine the regularisation. 
                        Options: "none", "auto", "interactive"
    regul_nthreads      Number of threads to use for running simultaneous 
                        instances of ppxf if regularisation_method is "auto"
    delta_regul_min     Minimum spacing in regularisation before the program 
                        exits (make it a larger value to speed up execution)
                        if regularisation_method is "auto"
    regul_max           Maximum regularisation value before the program gives up 
                        trying to find the minimum in regul space (make it a 
                        smaller value to speed up execution) if 
                        regularisation_method is "auto"
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

    assert regularisation_method in ["none", "auto", "interactive"],\
        "regularisation_method must be one of 'none', 'auto' or 'interactive'!"

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

    # Mask out regions where the noise vector is zero or inifinite, and where 
    # the spectrum is negative
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

    # Normalize spectrum to avoid numerical issues
    norm = np.median(spec_log[good_px])
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

    # pPXF parameters for the age & metallicity + gas fit
    adegree_age_met = -1     # Should be zero for age + metallicity fitting
    mdegree_age_met = 4     # Should be zero for kinematic fitting
    ncomponents = ngascomponents + 1 if fit_gas else 1    # number of kinematic components. 2 = stars + gas; 3 = stars + 2 * gas
    nmoments_age_met = [2 for i in range(ncomponents)]
    start_age_met = [[vel, 100.] for i in range(ncomponents)] if fit_gas else [vel, 100.]
    fixed_age_met = [[0, 0] for i in range(ncomponents)] if fit_gas else [0, 0]
    # tie_balmer = True if grating == "COMB" else False
    limit_doublets = False

    # pPXF parameters for the stellar kinematics fit
    adegree_kin = 12   # Should be zero for age + metallicity fitting
    mdegree_kin = 0   # Should be zero for kinematic fitting
    nmoments_kin = 2    # 2: only fit radial velocity and velocity dispersion
    start_kin = [vel, 100.]
    fixed_kin = [0, 0]

    # SSP template parameters
    # Gonzalez-Delgado spectra_linear have a constant spectral sampling of 0.3 A.
    dlambda_A_ssp = 0.30
    # Assuming that sigma_diff_px = dlambda_A_ssp.
    FWHM_ssp_A = 2 * np.sqrt(2 * np.log(2)) * dlambda_A_ssp

    ##############################################################################
    # SSP templates
    ##############################################################################
    # Load the SSP templates
    stellar_templates_linear, lambda_vals_ssp_linear, metallicities, ages =\
        load_ssp_templates(isochrones, metals_to_use)
    N_metallicities = len(metallicities)
    N_ages = len(ages)
    N_lambda_ssp_linear = len(lambda_vals_ssp_linear)
    reg_dim = (N_metallicities, N_ages)

    # Reshape stellar templates so that its dimensions are (N_lambda, N_ages * N_metallicities)
    stellar_templates_linear =\
        stellar_templates_linear.reshape((N_lambda_ssp_linear, N_ages * N_metallicities))

    # Extract the wavelength range for the logarithmically rebinned templates
    _, lambda_vals_ssp_log, _ = util.log_rebin(np.array(
        [lambda_vals_ssp_linear[0], lambda_vals_ssp_linear[-1]]),
        stellar_templates_linear[:, 0], velscale=velscale)
    N_lambda_ssp_log = len(lambda_vals_ssp_log)

    # Create an array to store the logarithmically rebinned spectra in
    stellar_templates_log = np.zeros((N_lambda_ssp_log, N_ages * N_metallicities))

    # Convolve each SSP template to the instrumental resolution
    FWHM_diff_A = np.sqrt(FWHM_inst_A**2 - FWHM_ssp_A**2)
    sigma_diff_px = FWHM_diff_A / (2 * np.sqrt(2 * np.log(2))) / dlambda_A_ssp  # sigma_diff_px difference in pixels
    stars_templates_linear_conv = np.zeros(stellar_templates_linear.shape)
    for ii in range(N_ages * N_metallicities):
        # Convolve
        stars_templates_linear_conv[:, ii] =\
            ndimage.gaussian_filter1d(stellar_templates_linear[:, ii], sigma_diff_px)

        # Logarithmically rebin
        spec_ssp_log, lambda_vals_ssp_log, velscale_temp =\
            util.log_rebin(np.array(
                [lambda_vals_ssp_linear[0], lambda_vals_ssp_linear[-1]]),
                stars_templates_linear_conv[:, ii],
                velscale=velscale)
        stellar_templates_log[:, ii] = spec_ssp_log

    # Normalise
    stellar_template_norms = np.nanmedian(stellar_templates_log, axis=0)
    stellar_templates_log /= stellar_template_norms

    # Reshape
    stellar_template_norms = np.reshape(stellar_template_norms, (N_metallicities, N_ages))

    # This line only works if velscale_ratio = 1
    dv = (lambda_vals_ssp_log[0] - lambda_vals_log[0]) * constants.c / 1e3  # km/s

    ##############################################################################
    # Merge templates so they can be input to pPXF
    ##############################################################################
    if fit_gas:
        ##############################################################################
        # Gas templates
        ##############################################################################
        # Construct a set of Gaussian emission line stellar_templates_log.
        # Estimate the wavelength fitted range in the rest frame.
        gas_templates, gas_names, eline_lambdas = util.emission_lines(
            logLam_temp=lambda_vals_ssp_log,
            lamRange_gal=np.array([lambda_start_A, lambda_end_A]) / (1 + z),
            FWHM_gal=FWHM_inst_A,
            tie_balmer=tie_balmer,
            limit_doublets=limit_doublets,
            vacuum=False
        )
        # Combines the stellar and gaseous stars_templates into a single array.
        # During the PPXF fit they will be assigned a different kinematic
        # COMPONENT value
        n_ssp_templates = stellar_templates_log.shape[1]
        # forbidden lines contain "[*]"
        n_forbidden_lines = np.sum(["[" in a for a in gas_names])
        n_balmer_lines = len(gas_names) - n_forbidden_lines

        # Here, we lump together the Balmer + forbidden lines into a single kinematic component
        if ncomponents == 3:
            kinematic_components = [0] * n_ssp_templates + \
                [1] * len(gas_names) + [2] * len(gas_names)
        elif ncomponents == 4:
            kinematic_components = [0] * n_ssp_templates + \
                [1] * len(gas_names) + [2] * len(gas_names) + [3] * len(gas_names)
        elif ncomponents == 2:
            kinematic_components = [0] * n_ssp_templates + [1] * len(gas_names)

        # If the Balmer lines are tied one should allow for gas reddeining.
        # The gas_reddening can be different from the stellar one, if both are fitted.
        gas_reddening = 0 if tie_balmer else None

        # Combines the stellar and gaseous stellar_templates_log into a single array.
        # During the PPXF fit they will be assigned a different kinematic
        # COMPONENT value
        if ncomponents > 2:
            if ngascomponents == 2:
                gas_templates = np.concatenate((gas_templates, gas_templates), axis=1)
                gas_names = np.concatenate((gas_names, gas_names))
            if ngascomponents == 3:
                gas_templates = np.concatenate((gas_templates, gas_templates, gas_templates), axis=1)
                gas_names = np.concatenate((gas_names, gas_names, gas_names))
            eline_lambdas = np.concatenate((eline_lambdas, eline_lambdas))

        gas_names_new = []
        for ii in range(len(gas_names)):
            gas_names_new.append(f"{gas_names[ii]} (component {kinematic_components[ii + n_ssp_templates]})")
        gas_names = gas_names_new
        templates = np.column_stack([stellar_templates_log, gas_templates])

        # gas_component=True for gas templates
        gas_component = np.array(kinematic_components) > 0

    # Case: no gas templates
    else:
        n_ssp_templates = stellar_templates_log.shape[1]
        kinematic_components = [0] * n_ssp_templates
        gas_names = None 
        templates = stellar_templates_log
        gas_reddening = None
        gas_component = None

    ##########################################################################
    # Mass-weighted ages
    ##########################################################################
    def compute_mass_weights(pp):
        # Reshape the normalisation factors into the same shape as the ppxf weights
        weights_light_weighted = pp.weights
        weights_light_weighted = np.reshape(
            weights_light_weighted[~pp.gas_component], (N_metallicities, N_ages))

        # Convert the light-weighted ages into mass-weighted ages
        weights_mass_weighted = weights_light_weighted * norm / stellar_template_norms

        return weights_mass_weighted

    ##############################################################################
    # Wrapper for plotting
    ##############################################################################
    def plot_wrapper(pp):

        # Comptue the mass weights 
        N_ages = len(ages)
        N_metallicities = len(metallicities)
        weights_mass_weighted = compute_mass_weights(pp)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 6.5))
        ax_spec, ax_hist = axs
        bbox = ax_hist.get_position()
        cbarax = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.03, bbox.height])

        # Histogram
        m = ax_hist.imshow(np.log10(weights_mass_weighted), cmap="cubehelix_r", origin="lower", aspect="auto", vmin=0, vmax=np.nanmax(np.log10(weights_mass_weighted)))
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
    if regularisation_method == "none":
        # If no regularisation is required, then just run once and exit.
        regul = 0
        t = time()
        delta_chi2_ideal = np.sqrt(2 * len(good_px))
        pp = ppxf(templates=templates,
                  galaxy=spec_log, noise=spec_err_log,
                  velscale=np.squeeze(velscale), start=start_age_met,
                  goodpixels=good_px,
                  moments=nmoments_age_met, degree=adegree_age_met, mdegree=mdegree_age_met,
                  vsyst=dv,
                  lam=np.exp(lambda_vals_log),
                  regul=regul,
                  reddening_func=reddening_fm07,
                  reg_dim=reg_dim,
                  component=kinematic_components, gas_component=gas_component,
                  gas_names=gas_names, gas_reddening=gas_reddening, method="capfit",
                  quiet=True)
        print("----------------------------------------------------")
        print(F"Elapsed time in PPXF: {time() - t:.2f} s")

    else:
        # Otherwise, run ppxf the first time w/o regularisation. 
        ws = []  # List to store template weights from successive iterations 
        t = time()
        regul = 0
        delta_chi2_ideal = np.sqrt(2 * len(good_px))
        pp_age_met = ppxf(templates=templates,
                          galaxy=spec_log, noise=spec_err_log,
                          velscale=np.squeeze(velscale), start=start_age_met,
                          goodpixels=good_px,
                          moments=nmoments_age_met, degree=adegree_age_met, mdegree=mdegree_age_met,
                          vsyst=dv,
                          lam=np.exp(lambda_vals_log),
                          regul=regul,
                          reddening_func=reddening_fm07,
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
            plt.close("all")
            plot_wrapper(pp_age_met)
            fig = plt.gcf()
            fig.suptitle(f"First ppxf iteration: regul = 0")
            if savefigs:
                pdfpages_spec.savefig(fig, bbox_inches="tight")

            # SFH change 
            fig_sfh, axs_sfh = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
            fig_sfh.subplots_adjust(hspace=0)
            fig_sfh.suptitle(f"First ppxf iteration: regul = 0")
            axs_sfh[0].step(x=range(len(w)), y=w, where="mid", label=f"Iteration 0")
            axs_sfh[0].set_xlabel("Template age")
            axs_sfh[0].set_ylabel(r"$\log_{10} (M_* [\rm M_\odot])$")
            axs_sfh[0].set_yscale("log")
            axs_sfh[0].set_ylim([10^0, None])
            axs_sfh[0].autoscale(axis="x", enable=True, tight=True)
            axs_sfh[0].grid()
            axs_sfh[0].legend()

            axs_sfh[1].axhline(0, color="gray")
            axs_sfh[1].set_xticks(range(len(ages)))
            axs_sfh[1].set_xticklabels([f"{age / 1e6}" for age in ages], rotation="vertical")
            axs_sfh[1].autoscale(axis="x", enable=True, tight=True)
            axs_sfh[1].grid()
            fig_sfh.canvas.draw()
            if savefigs:
                pdfpages_sfh.savefig(fig_sfh, bbox_inches="tight")

            if matplotlib.get_backend() != "agg" and interactive_mode:
                hit_key_to_continue()

        if regularisation_method == "interactive":
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
                                  velscale=np.squeeze(velscale), start=start_age_met,
                                  goodpixels=good_px,
                                  moments=nmoments_age_met, degree=adegree_age_met, mdegree=mdegree_age_met,
                                  vsyst=dv,
                                  lam=np.exp(lambda_vals_log),
                                  regul=regul,
                                  reddening_func=reddening_fm07,
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
                    plt.close("all")
                    plot_wrapper(pp_age_met)
                    fig = plt.gcf()
                    fig.suptitle(f"Manually determining regul parameter: regul = {regul:.2f} (iteration {cnt})")
                    if savefigs:
                        pdfpages_spec.savefig(fig, bbox_inches="tight")

                    # Plot the updated SFH
                    fig_sfh.suptitle(f"Manually determining regul parameter: regul = {regul:.2f} (iteration {cnt})")
                    axs_sfh[0].step(x=range(len(w)), y=w, where="mid", label=f"Iteration {cnt}")
                    axs_sfh[0].set_ylim([10^0, None])
                    axs_sfh[0].autoscale(axis="x", enable=True, tight=True)
                    axs_sfh[0].grid()
                    axs_sfh[0].legend()

                    # Plot the difference in the SFH from the previous iteration
                    axs_sfh[1].clear()
                    axs_sfh[1].plot(range(len(w)), dw, "ko")
                    axs_sfh[1].set_yscale("log")
                    axs_sfh[1].set_ylabel("Percentage change from previous best fit")
                    axs_sfh[1].axhline(0, color="gray")
                    axs_sfh[1].set_xticks(range(len(ages)))
                    axs_sfh[1].set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")
                    axs_sfh[1].text(s=f"Absolute mass difference: {delta_m:.2g} $M_\odot$", x=0.1, y=0.9, transform=axs_sfh[1].transAxes)
                    axs_sfh[1].autoscale(axis="x", enable=True, tight=True)
                    axs_sfh[1].grid()
                    fig_sfh.canvas.draw()
                    if savefigs:
                        pdfpages_sfh.savefig(fig_sfh, bbox_inches="tight")

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
            print(F"Iteration {cnt}: Scaling noise by {np.sqrt(pp_age_met.chi2):.4f}...")
            noise_scaling_factor = np.sqrt(pp_age_met.chi2)

            # Run ppxf a number of times & find the value of regul that minimises 
            # the difference between the ideal delta-chi2 and the real delta-chi2.
            regul_vals = np.linspace(0, 1e4, regul_nthreads + 1)
            delta_regul = np.diff(regul_vals)[0]

            obj_vals = []  # "objective" fn
            pps = []

            # Input arguments
            args_list = [
                [
                    templates, spec_log, spec_err_log, noise_scaling_factor,
                    velscale, start_age_met, good_px, nmoments_age_met, adegree_age_met,
                    mdegree_age_met, dv, lambda_vals_log, regul, reddening_fm07,
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
                plot_wrapper(pps[opt_idx])
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

                # Plot the updated SFH
                fig_sfh.suptitle(f"SFH evolution: regul = {regul_vals[opt_idx]:.2f}, " + r"$\Delta \chi_{\rm goal} - \Delta \chi$ =" + f"{obj_vals[opt_idx]:.2f} (iteration {1})")
                axs_sfh[0].step(x=range(len(w)), y=w, where="mid", label=f"Iteration 1")
                axs_sfh[0].set_ylim([10^0, None])
                axs_sfh[0].autoscale(axis="x", enable=True, tight=True)
                axs_sfh[0].grid()
                axs_sfh[0].legend()

                # Plot the difference in the SFH from the previous iteration
                axs_sfh[1].clear()
                axs_sfh[1].plot(range(len(w)), dw, "ko")
                axs_sfh[1].set_yscale("log")
                axs_sfh[1].set_ylabel("Percentage change from previous best fit")
                axs_sfh[1].axhline(0, color="gray")
                axs_sfh[1].set_xticks(range(len(ages)))
                axs_sfh[1].set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")
                axs_sfh[1].text(s=f"Absolute mass difference: {delta_m:.2g} $M_\odot$", x=0.1, y=0.9, transform=axs_sfh[1].transAxes)
                axs_sfh[1].autoscale(axis="x", enable=True, tight=True)
                axs_sfh[1].grid()
                fig_sfh.canvas.draw()
                if savefigs:
                    pdfpages_sfh.savefig(fig_sfh, bbox_inches="tight")

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
                    regul_vals = np.linspace(np.nanmax(regul_vals),
                                             np.nanmax(regul_vals) + regul_span,
                                             regul_nthreads + 1)
                # Otherwise, sub-sample the previous array windowed around the optimal value.
                else:
                    delta_regul /= 5
                    regul_vals = np.linspace(regul_vals[opt_idx] - regul_nthreads / 2 * delta_regul, 
                                             regul_vals[opt_idx] + regul_nthreads / 2 * delta_regul,
                                             regul_nthreads + 1)

                # Reset the values
                delta_regul = np.diff(regul_vals)[0]

                # Re-run ppxf
                args_list = [
                    [
                        templates, spec_log, spec_err_log, noise_scaling_factor,
                        velscale, start_age_met, good_px, nmoments_age_met, adegree_age_met,
                        mdegree_age_met, dv, lambda_vals_log, regul, reddening_fm07,
                        reg_dim, kinematic_components, gas_component, gas_names,
                        gas_reddening
                    ] for regul in regul_vals
                ]

                # Run in parallel
                cnt += 1
                print(f"Iteration {cnt}: Re-running ppxf on {regul_nthreads} threads (iteration {cnt})...")
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
                    plot_wrapper(pps[opt_idx])
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

                    # Plot the updated SFH
                    axs_sfh[0].step(x=range(len(w)), y=w, where="mid", label=f"Iteration {cnt}")
                    axs_sfh[0].set_ylim([10^0, None])
                    axs_sfh[0].autoscale(axis="x", enable=True, tight=True)
                    axs_sfh[0].grid()
                    axs_sfh[0].legend()

                    # Plot the difference in the SFH from the previous iteration
                    fig_sfh.suptitle(f"SFH evolution: regul = {regul_vals[opt_idx]:.2f}, " + r"$\Delta \chi_{\rm goal} - \Delta \chi$ =" + f"{obj_vals[opt_idx]:.2f} (iteration {cnt})")
                    axs_sfh[1].clear()
                    axs_sfh[1].plot(range(len(w)), dw, "ko")
                    axs_sfh[1].set_yscale("log")
                    axs_sfh[1].set_ylabel("Percentage change from previous best fit")
                    axs_sfh[1].axhline(0, color="gray")
                    axs_sfh[1].set_xticks(range(len(ages)))
                    axs_sfh[1].set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")
                    axs_sfh[1].text(s=f"Absolute mass difference: {delta_m:.2g} $M_\odot$", x=0.1, y=0.9, transform=axs_sfh[1].transAxes)
                    axs_sfh[1].autoscale(axis="x", enable=True, tight=True)
                    axs_sfh[1].grid()
                    fig_sfh.canvas.draw()
                    if savefigs:
                        pdfpages_sfh.savefig(fig_sfh, bbox_inches="tight")

                    if savefigs:
                        pdfpages_regul.savefig(fig_regul, bbox_inches="tight")

                    if matplotlib.get_backend() != "agg" and interactive_mode:
                        hit_key_to_continue()

            pp = pps[opt_idx]

    ##########################################################################
    # Add some extra useful stuff to the ppxf instance
    ##########################################################################
    pp.weights_mass_weighted = compute_mass_weights(pp)  # Mass-weighted template weights
    pp.weights_light_weighted = np.reshape(pp.weights[~pp.gas_component], (N_metallicities, N_ages))
    pp.norm = norm  # Normalisation factor for the logarithmically binned input spectrum
    pp.stellar_template_norms = stellar_template_norms  # Normalisation factors for the logarithmically binned stellar templates

    ##########################################################################
    # Plotting the fit
    ##########################################################################
    if plotit:
        plot_wrapper(pp)
        fig = plt.gcf()
        if regularisation_method != "none":
            regul_final = regul_vals[opt_idx]
        else:
            regul_final = regul
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
    from mockspec import create_mock_spectrum, get_age_and_metallicity_values

    ###########################################################################
    # Mock spectra options
    ###########################################################################
    isochrones = "Padova"  # Set of isochrones to use 
    
    ###########################################################################
    # GALAXY PROPERTIES
    ###########################################################################
    sigma_star_kms = 350   # LOS velocity dispersion, km/s
    z = 0.05               # Redshift 
    SNR = 100               # S/N ratio

    ###########################################################################
    # DEFINE THE SFH
    ###########################################################################
    # Idea 1: use a Gaussian kernel to smooth "delta-function"-like SFHs
    # Idea 2: are the templates logarithmically spaced in age? If so, could use e.g. every 2nd template 
    ages, metallicities = get_age_and_metallicity_values(isochrones)
    N_ages = len(ages)
    N_metallicities = len(metallicities)

    sfh_input = np.zeros((N_metallicities, N_ages))
    # sfh_input[1, 10] = 1e7
    sfh_input[2, 4] = 1e10

    ###########################################################################
    # CREATE THE MOCK SPECTRUM
    ###########################################################################
    spec, spec_err, lambda_vals_A = create_mock_spectrum(
        sfh_mass_weighted=sfh_input,
        isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
        plotit=False)    

    ###########################################################################
    # RUN PPXF
    ###########################################################################
    pp, sfh_fit = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A, 
                           z=z, ngascomponents=1,
                           auto_adjust_regul=True,
                           plotit=True,
                           isochrones="Padova", tie_balmer=True)

    ###########################################################################
    # COMPARE THE INPUT AND OUTPUT
    ###########################################################################
    # side-by-side comparison of the SFHs, plus residual map
    plot_sfh_mass_weighted(sfh_input, ages, metallicities)
    plot_sfh_mass_weighted(sfh_fit, ages, metallicities)

    # Plot the residual
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 3.5))
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x0 + bbox.width, bbox.x0, 0.025, bbox.height])
    
    # Plot the SFH
    delta_sfh = sfh_input - sfh_fit
    m = ax.imshow(delta_sfh, cmap="coolwarm", 
                  origin="lower", aspect="auto",
                  vmin=-np.abs(np.nanmax(delta_sfh)), vmax=np.abs(np.nanmax(delta_sfh)))
    fig.colorbar(m, cax=cax)
    
    # Decorations
    ax.set_yticks(range(len(metallicities)))
    ax.set_yticklabels(["{:.3f}".format(met / 0.02) for met in metallicities])
    ax.set_ylabel(r"Metallicity ($Z_\odot$)")
    cax.set_ylabel(r"Residual (\rm M_\odot)$")
    ax.set_xticks(range(len(ages)))
    ax.set_xlabel("Age (Myr)")
    ax.set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")
    axs_sfh[1].autoscale(axis="x", enable=True, tight=True)
    axs_sfh[1].grid()
    fig_sfh.canvas.draw()

