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
    A_V = 3.1 * ebv
    A_lambda = extinction.fm07(wave=lam, a_v=A_V, unit='aa')
    fact = 10**(-0.4 * A_lambda)  # Need a minus sign here!
    return fact

def reddening_calzetti00(lam, ebv):
    # lam in Angstroms
    # Need to derive A(lambda) from E(B-V)
    # calzetti00 takes as input lambda and A_V, so we first need to convert E(B-V)
    # into A_V
    R_V = 3.1
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
        mdegree, dv, lambda_vals_log, regul, reddening, reddening_calzetti00,\
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
          reddening=reddening, reddening_func=reddening_calzetti00,
          reg_dim=reg_dim,
          component=kinematic_components, gas_component=gas_component,
          gas_names=gas_names, gas_reddening=gas_reddening, method="capfit",
          quiet=True)

    # Return
    return pp_age_met

##############################################################################
# START FUNCTION DEFINITION
##############################################################################
def run_ppxf(spec, spec_err, lambda_vals_A, z, 
             isochrones, metals_to_use=None,
             fit_agn_cont=False, alpha_nu_vals=[0.5, 1.0, 1.5, 2.0],
             fit_gas=True, ngascomponents=0, tie_balmer=True,
             mdegree=4, adegree=-1, reddening=None,
             FWHM_inst_A=FWHM_WIFES_INST_A,
             bad_pixel_ranges_A=[],
             lambda_norm_A=4020,
             regularisation_method="auto", regul_fixed=0, 
             auto_adjust_regul=False, regul_nthreads=20,
             regul_start=0, regul_span=1e4,
             delta_regul_min=1, regul_max=10e4, delta_delta_chi2_min=1,
             interactive_mode=False,
             plotit=False, savefigs=False, fname_str="ppxftests",
             reg_dim=None):
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
        cmap = matplotlib.cm.viridis_r
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
                    mdegree, dv, lambda_vals_log, regul, reddening, reddening_calzetti00,
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
                        mdegree, dv, lambda_vals_log, regul, reddening, reddening_calzetti00,
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

    # AGN continuum parameters, if used 
    if fit_agn_cont:
        pp.alpha_nu_vals = alpha_nu_vals
        pp.fit_agn_cont = True
        pp.weights_agn = pp.weights[~pp.gas_component][-n_agn_templates:] 
        # NOTE: agn_bestfit is on the wavelength grid of the logarithmically rebinned SSP templates - NOT the "galaxy" wavelength grid
        if n_agn_templates == 1:
            pp.agn_bestfit = np.squeeze(pp.templates[:, -n_agn_templates:]) * pp.weights[None, -n_agn_templates:]
        else:
            pp.agn_bestfit = np.sum(pp.templates[:, -n_agn_templates:] * pp.weights[None, -n_agn_templates:], axis=1)
    else:
        pp.alpha_nu_vals = None 
        pp.fit_agn_cont = False
        pp.weights_agn = None
        pp.agn_bestfit = None

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

