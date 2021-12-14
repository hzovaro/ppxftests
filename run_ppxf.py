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
from __future__ import print_function
import sys, os

import matplotlib
# matplotlib.use("Agg")

from time import time

from astropy.io import fits
from astroquery.ned import Ned
from astroquery.irsa_dust import IrsaDust
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
from ppxftests.ppxf_plot import ppxf_plot

from IPython.core.debugger import Tracer

##############################################################################
# Plotting settings
##############################################################################
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rc("font", size=14)
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
          gas_names=gas_names, gas_reddening=gas_reddening, method="capfit")

    # Return
    return pp_age_met

##########################################################################
# Mass-weighted ages
##########################################################################
def compute_mass_weights(pp)
    # Reshape the normalisation factors into the same shape as the ppxf weights
    weights_light_weighted = pp.weights
    weights_light_weighted = np.reshape(
        weights_light_weighted[~pp.gas_component], (nmetals, nages))

    # Convert the light-weighted ages into mass-weighted ages
    weights_mass_weighted = weights_light_weighted * norm / stellar_template_norms

    # Compute total mass in this bin
    mass_tot = np.nansum(weights_mass_weighted)

    # Sum in metallicity to get an age vector.
    weights_mass_weighted_metallicity_summed = np.nansum(weights_mass_weighted, axis=0)

##############################################################################
# Wrapper for plotting
##############################################################################
def plot_wrapper(pp, ages, metallicities):

    weights_mass_weighted = compute_mass_weights(pp)

    fig_spec = plt.figure(figsize=(20, 12))
    ax_hist = fig_spec.add_axes([0.1, 0.1, 0.8, 0.2])
    # ax_1dhist = fig_spec.add_axes([0.1, 0.3, 0.8, 0.2])
    ax_kin = fig_spec.add_axes([0.1, 0.55, 0.8, 0.2])
    ax_age_met = fig_spec.add_axes([0.1, 0.75, 0.8, 0.2])
    cbarax = fig_spec.add_axes([0.9, 0.1, 0.02, 0.2])

    # Open the pdf file
    # pp = PdfPages(fig_fname)

    # Figure for auto_adjust_regul
    # if auto_adjust_regul:
        # fig_regul, ax_regul = plt.subplots()
        # pp_regul = PdfPages(fig_regul_fname)

    # Star formation history
    # if np.any(weights_mass_weighted_metallicity_summed > 0): 
    #     ax_1dhist.semilogy(weights_mass_weighted_metallicity_summed)
    # ax_1dhist.set_ylabel(r"Mass ($\rm M_\odot$)")
    # ax_1dhist.text(x=0.5, y=0.9, s="Star formation history", transform=ax_1dhist.transAxes, horizontalalignment="center")
    # ax_1dhist.autoscale(axis="x", enable=True, tight=True)
    # ax_1dhist.set_xticklabels([])
    # ax_1dhist.axvline(idx_end_of_SB, color="gray")
    # ax_1dhist.axvline(idx_start_of_SB, color="gray")

    # Histogram
    m = ax_hist.imshow(np.log10(weights_mass_weighted), cmap="cubehelix_r", origin="lower", aspect="auto", vmin=0, vmax=np.nanmax(np.log10(weights_mass_weighted)))
    fig_spec.colorbar(m, cax=cbarax)
    ax_hist.set_yticks(range(len(metallicities)))
    ax_hist.set_yticklabels(["{:.3f}".format(met / 0.02)
                             for met in metallicities])
    ax_hist.set_ylabel(r"Metallicity ($Z_\odot$)")
    cbarax.set_ylabel(r"Mass $\log_{10}(\rm M_\odot)$")
    ax_hist.set_xticks(range(len(ages)))
    ax_hist.set_xlabel("Age (Myr)")
    ax_hist.set_title("Best fit stellar population")
    ax_hist.set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")

    # Kinematic and age & metallicity fits
    ax_age_met.clear()
    # ax_kin.clear()
    ppxf_plot(pp_age_met, ax_age_met)
    # ppxf_plot(pp_kin, ax_kin)
    ax_age_met.text(x=0.5, y=0.9, horizontalalignment="center", transform=ax_age_met.transAxes, s=r"ppxf fit (age and metallicity)")
    # ax_kin.text(x=0.5, y=0.9, horizontalalignment="center", transform=ax_kin.transAxes, s=r"ppxf fit (kinematics); $v_* - v_{\rm sys} = %.2f$ km s$^{-1}$, $\sigma_* = %.2f$ km s$^{-1}$" % (pp_kin.sol[0] - v_sys, pp_kin.sol[1]))
    ax_age_met.set_xticklabels([])
    ax_age_met.set_xlabel("")
    # ax_age_met.set_title(obj_name)
    ax_age_met.autoscale(axis="x", enable=True, tight=True)
    # ax_kin.autoscale(axis="x", enable=True, tight=True)

    fig_spec.canvas.draw()

    # Write to file
    # pp.savefig(fig_spec)

    # if auto_adjust_regul:
    #     ax_regul.clear()
    #     ax_regul.plot(regul_vals, obj_vals, "bo")
    #     ax_regul.plot(regul_vals[np.nanargmin(obj_vals)], obj_vals[np.nanargmin(obj_vals)], "ro", label="Optimal fit")
    #     ax_regul.axhline(0, color="gray")
    #     ax_regul.set_title("Integrated spectrum")
    #     ax_regul.set_xlabel("Regularisation parameter")
    #     ax_regul.set_ylabel(r"$\Delta\chi_{\rm goal}^2 - \Delta\chi^2$")
    #     ax_regul.legend()
    #     fig_regul.canvas.draw()
    #     pp_regul.savefig(fig_regul)

    plt.show()

##############################################################################
# START FUNCTION DEFINITION
##############################################################################
def run_ppxf(spec, spec_err, lambda_vals_A, FWHM_inst_A,
             z, ngascomponents,
             isochrones,
             tie_balmer,
             bad_pixel_ranges_A=[],
             auto_adjust_regul=False):
    """
    Wrapper function for calling ppxf.

    Inputs:
    spec                Input spectrum to fit to, on a linear wavelength scale
    spec_err            Corresponding 1-sigma errors 
    lambda_vals_A       Wavelength values, in Angstroms
    FWHM_inst_A         Instrumental resoltuion in Angstroms
    z                   Galaxy redshift
    bad_pixel_ranges_A  Spectral regions to mask out (in Angstroms). format: [[lambda_1, lambda_2], ...]
    ngascomponents      Number of kinematic components to be fitted to the emission lines
    
    isochrones          Set of isochrones to use - must be Padova or Geneva
    tie_balmer          If true, use the Ha/Hb ratio to measure gas reddening.
    auto_adjust_regul   Whether to automatically determine the ideal "regul" value

    """
    ##############################################################################
    # Set up the input spectrum
    ##############################################################################
    lambda_start_A = lambda_vals_A[0]
    lambda_end_A = lambda_vals_A[-1]
    spec_linear = spec
    spec_err_linear = spec_err

    # Calculate the SNR
    SNR = np.nanmedian(spec_linear / spec_err_linear)
    print("Median SNR = {:.4f}".format(SNR))

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
        metals_to_use = ['004', '008', '019']
    elif isochrones == "Geneva":
        metals_to_use = ['001', '004', '008', '020', '040']
    ssp_template_path = f"/home/u5708159/python/Modules/ppxftests/SSP_templates/SSP{isochrones}"

    # pPXF parameters for the age & metallicity + gas fit
    adegree_age_met = -1     # Should be zero for age + metallicity fitting
    mdegree_age_met = 4     # Should be zero for kinematic fitting
    ncomponents = ngascomponents + 1    # number of kinematic components. 2 = stars + gas; 3 = stars + 2 * gas
    nmoments_age_met = [2 for i in range(ncomponents)]
    start_age_met = [[vel, 100.] for i in range(ncomponents)]
    fixed_age_met = [[0, 0] for i in range(ncomponents)]
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
    # Assuming that sigma = dlambda_A_ssp.
    FWHM_ssp_A = 2 * np.sqrt(2 * np.log(2)) * dlambda_A_ssp

    ##############################################################################
    # SSP templates
    ##############################################################################
    # Load the .npz containing the stellar spectra
    ssp_template_fnames = [f"SSP{isochrones}.z{m}.npz" for m in metals_to_use]
    nmetals = len(ssp_template_fnames)

    # All stars_templates_log must have the same number of wavelength values &
    # number of age bins!
    stars_templates_log = []
    stars_templates_linear = []
    stellar_template_norms = []
    metallicities = []
    for ssp_template_fname in ssp_template_fnames:
        f = np.load(os.path.join(ssp_template_path, ssp_template_fname))
        metallicities.append(f["metallicity"].item())
        ages = f["ages"][:-2]
        spectra_ssp_linear = f["L_vals"][:, :-2]
        lambda_vals_ssp_linear = f["lambda_vals_A"]

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to a velocity scale 2x smaller than the stellar template spectra_linear, to
        # determine the size needed for the array which will contain the template
        # spectra_linear.
        spec_ssp_log, lambda_vals_ssp_log, velscale_temp = util.log_rebin(np.array(
            [lambda_vals_ssp_linear[0], lambda_vals_ssp_linear[-1]]),
            spectra_ssp_linear[:, 0], velscale=velscale)
        stars_templates_log.append(np.empty((spec_ssp_log.size, len(ages))))
        stars_templates_linear.append(np.empty((spectra_ssp_linear[:, 0].size,
                                                len(ages))))

        # Quadratic sigma difference in pixels Vazdekis --> SAURON
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        FWHM_diff_A = np.sqrt(FWHM_inst_A**2 - FWHM_ssp_A**2)
        sigma = FWHM_diff_A / (2 * np.sqrt(2 * np.log(2))) / \
            dlambda_A_ssp  # Sigma difference in pixels
        nages = spectra_ssp_linear.shape[1]
        for ii in range(nages):
            spec_ssp_linear = spectra_ssp_linear[:, ii]
            spec_ssp_linear = ndimage.gaussian_filter1d(spec_ssp_linear, sigma)
            spec_ssp_log, lambda_vals_ssp_log, velscale_temp =\
                util.log_rebin(np.array(
                    [lambda_vals_ssp_linear[0], lambda_vals_ssp_linear[-1]]),
                    spec_ssp_linear, velscale=velscale)
            # Normalise templates
            stars_templates_log[-1][:, ii] = spec_ssp_log / np.median(spec_ssp_log)
            stars_templates_linear[-1][:, ii] = spec_ssp_linear / np.median(spec_ssp_linear)

            stellar_template_norms.append(np.median(spec_ssp_log))
            
    # Reshape
    stellar_template_norms = np.reshape(stellar_template_norms, (nmetals, nages))

    # String for filename
    metal_string = ""
    for metal in metallicities:
        metal_string += str(metal).split("0.")[1]
        metal_string += "_"
    metal_string = metal_string[:-1]

    # Convert to array
    stars_templates_log = np.array(stars_templates_log)
    stars_templates_log = np.swapaxes(stars_templates_log, 0, 1)
    reg_dim = stars_templates_log.shape[1:]
    nmetals, nages = reg_dim
    stars_templates_log = np.reshape(
        stars_templates_log, (stars_templates_log.shape[0], -1))

    # Store the linear spectra_linear too
    stars_templates_linear = np.array(stars_templates_linear)
    stars_templates_linear = np.swapaxes(stars_templates_linear, 0, 1)
    stars_templates_linear = np.reshape(
        stars_templates_linear, (stars_templates_linear.shape[0], -1))

    # This line only works if velscale_ratio = 1
    dv = (lambda_vals_ssp_log[0] - lambda_vals_log[0]) * constants.c / 1e3  # km/s

    ##############################################################################
    # Gas templates
    ##############################################################################
    # Construct a set of Gaussian emission line stars_templates_log.
    # Estimate the wavelength fitted range in the rest frame.
    gas_templates, gas_names, eline_lambdas = util.emission_lines(
        logLam_temp=lambda_vals_ssp_log,
        lamRange_gal=np.array([lambda_start_A, lambda_end_A]) / (1 + z),
        FWHM_gal=FWHM_inst_A,
        tie_balmer=tie_balmer,
        limit_doublets=limit_doublets,
        vacuum=False
    )

    ##############################################################################
    # Merge templates so they can be input to pPXF
    ##############################################################################
    # Combines the stellar and gaseous stars_templates_log into a single array.
    # During the PPXF fit they will be assigned a different kinematic
    # COMPONENT value
    n_ssp_templates = stars_templates_log.shape[1]
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
    # gas_component=True for gas templates
    gas_component = np.array(kinematic_components) > 0

    # If the Balmer lines are tied one should allow for gas reddeining.
    # The gas_reddening can be different from the stellar one, if both are fitted.
    gas_reddening = 0 if tie_balmer else None

    # Combines the stellar and gaseous stars_templates_log into a single array.
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
    templates = np.column_stack([stars_templates_log, gas_templates])

    ##########################################################################
    # Use pPXF to obtain the stellar age + metallicity, and fit emission lines
    ##########################################################################
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
                      gas_names=gas_names, gas_reddening=gas_reddening, method="capfit")
    delta_chi2 = (pp_age_met.chi2 - 1) * len(good_px)
    print("----------------------------------------------------")
    print(F"Desired Delta Chi^2: {delta_chi2_ideal:.4g}")
    print(F"Current Delta Chi^2: {delta_chi2:.4g}")
    print("----------------------------------------------------")
    print(F"Elapsed time in PPXF: {time() - t:.2f} s")

    plt.close("all")
    plot_wrapper(pp_age_met, ages, metallicities)

    if not auto_adjust_regul:
        # Run again but with regularization.
        print(F"Scaling noise by {np.sqrt(pp_age_met.chi2):.4f}...")
        noise_scaling_factor = np.sqrt(pp_age_met.chi2)

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

            plt.close("all")
            plot_wrapper(pp_age_met, ages, metallicities)

            while True:
                key = input("Enter a new regul value, otherwise press enter: ")
                if key.isdigit() or key == "":
                    break
            if key == "":
                break
            else:
                regul = float(key)

    else:
        # Run again but with regularization.
        print(F"Scaling noise by {np.sqrt(pp_age_met.chi2):.4f}...")
        noise_scaling_factor = np.sqrt(pp_age_met.chi2)

        # Run ppxf a number of times & find the value of regul that minimises 
        # the difference between the ideal delta-chi2 and the real delta-chi2.
        regul_vals = np.linspace(0, 1000, 11)
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
        nthreads = 20 # min(multiprocessing.cpu_count(), len(args_list))
        print(f"Running ppxf on {nthreads} threads...")
        pool = multiprocessing.Pool(nthreads)
        pps = list(pool.map(ppxf_helper, args_list))
        pool.close()
        pool.join()

        # Determine which is the optimal regul value
        # Quite certain this is correct - see here: https://pypi.org/project/ppxf/#how-to-set-regularization
        regul_vals = [p.regul for p in pps]  # Redefining as pool may not retain the order of the input list
        delta_chi2_vals = [(p.chi2 - 1) * len(good_px) for p in pps]
        obj_vals = [np.abs(delta_chi2 - delta_chi2_ideal) for delta_chi2 in delta_chi2_vals]
        opt_idx = np.nanargmin(obj_vals)

        # If opt_idx is the largest value, then re-run this bin with larger regul values.
        cnt = 2
        while regul_vals[opt_idx] == np.nanmax(regul_vals) and np.nanmax(regul_vals) < 5e3:
            # Input arguments
            regul_vals = np.linspace(np.nanmax(regul_vals), np.nanmax(regul_vals) + 1000, 11)
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
            print(f"Re-running ppxf on {nthreads} threads (iteration {cnt})...")
            pool = multiprocessing.Pool(nthreads)
            pps = list(pool.map(ppxf_helper, args_list))
            pool.close()
            pool.join()

            # Determine which is the optimal regul value
            regul_vals = [p.regul for p in pps]  # Redefining as pool may not retain the order of the input list
            delta_chi2_vals = [(p.chi2 - 1) * len(good_px) for p in pps]
            obj_vals = [np.abs(delta_chi2 - delta_chi2_ideal) for delta_chi2 in delta_chi2_vals]
            opt_idx = np.nanargmin(obj_vals)
            cnt += 1

        pp_age_met = pps[opt_idx]

    ##########################################################################
    # Use pPXF to fit the stellar kinematics
    ##########################################################################
    pp_kin = ppxf(templates=stars_templates_log,
                  galaxy=spec_log - pp_age_met.gas_bestfit, noise=spec_err_log * noise_scaling_factor,
                  velscale=np.squeeze(velscale), start=start_kin,
                  goodpixels=good_px,
                  moments=nmoments_kin, degree=adegree_kin, mdegree=mdegree_kin,
                  vsyst=dv,
                  lam=np.exp(lambda_vals_log),
                  method="capfit")
    print("Formal errors:")
    print("     dV    dsigma   dh3      dh4")
    print("".join("%8.2g" % f for f in pp_kin.error * np.sqrt(pp_kin.chi2)))
    print("Elapsed time in pPXF: %.2f s" % (time() - t))

    ##########################################################################
    # Reddening
    ##########################################################################
    # Calculate the A_V
    if not tie_balmer and grating == "COMB":
        intrinsic_ratios = {
            "Halpha/Hbeta": 2.85,
            "Hgamma/Hbeta": 0.468,
            "Hdelta/Hbeta": 0.259,
        }
        balmer_line_waves = {
            "Hdelta": 4101.734,
            "Hgamma": 4340.464,
            "Hbeta": 4861.325,
            "Halpha": 6562.800,
        }

        for line_1, line_2 in [["Hgamma", "Hbeta"], ["Hdelta", "Hbeta"]]:
            # From p. 384 of D&S
            intrinsic_ratio = intrinsic_ratios[line_1 + "/" + line_2]

            lfmap_1 = pp_age_met.gas_flux[list(gas_names).index(line_1)] * norm
            lfmap_1_err = pp_age_met.gas_flux_error[list(
                gas_names).index(line_1)] * norm

            lfmap_2 = pp_age_met.gas_flux[list(gas_names).index(line_2)] * norm
            lfmap_2_err = pp_age_met.gas_flux_error[list(
                gas_names).index(line_2)] * norm

            ratio = lfmap_1 / lfmap_2
            ratio_err = ratio * ((lfmap_1_err / lfmap_1) **
                                 2 + (lfmap_2_err / lfmap_2)**2)**(0.5)
            ratio_SNR = ratio / ratio_err

            E_ba = 2.5 * (np.log10(ratio)) - 2.5 * np.log10(intrinsic_ratio)
            E_ba_err = 2.5 / np.log(10) * ratio_err / ratio

            # Calculate ( A(Ha) - A(Hb) ) / E(B-V) from extinction curve
            R_V = 3.1
            wave_1_A = np.array([balmer_line_waves[line_1]])
            wave_2_A = np.array([balmer_line_waves[line_2]])

            # A_V is a multiplicative scale factor for the extinction curve.
            # So the below calculation is the same regardless of A_V because
            # we normalise by it.
            E_ba_over_E_BV = float(extinction.fm07(wave_2_A, a_v=1.0) -
                                   extinction.fm07(wave_1_A, a_v=1.0)) /\
                1.0 * R_V

            # Calculate E(B-V)
            E_BV = 1 / E_ba_over_E_BV * E_ba
            E_BV_err = 1 / E_ba_over_E_BV * E_ba_err

            # Calculate A(V)
            A_V = R_V * E_BV
            A_V_err = R_V * E_BV_err

            print(
                "-----------------------------------------------------------------------")
            print("Estimated mean A_V for integrated spectrum using ratio " +
                  line_1 + "/" + line_2 + " (pPXF):")
            print(f"A_V = {A_V:6.4f} +/- {A_V_err:6.4f}")
            print(
                "-----------------------------------------------------------------------")
    elif tie_balmer and grating == "COMB":
        print("-----------------------------------------------------------------------")
        print("Estimated mean A_V for integrated spectrum using all Balmer lines (calculated by pPXF):")
        print(f"A_V = {pp_age_met.gas_reddening * 3.1:6.4f}")
        print("-----------------------------------------------------------------------")
    else:
        print("------------------------------------------------------------------------------")
        print("Reddening not calculated due to insufficient Balmer lines in wavelength range")
        print("------------------------------------------------------------------------------")


    ##########################################################################
    # Print emission line fluxes
    ##########################################################################
    print("pPXF emission line fluxes")
    print("-----------------------------------------------------------------------")
    # NOTE: since the input spectrum is in units of erg/s/cm2/A, these fluxes 
    # need to be multilpied by the spectral pixel width in Angstroms to get the 
    # flux in units of erg/s/cm2.
    for name, flux, flux_error in zip(pp_age_met.gas_names,
                                      pp_age_met.gas_flux,
                                      pp_age_met.gas_flux_error):
        try:
            print(f"{name} \t & {sci_notation(flux * norm, flux_error * norm)} \\\\")
        except:
            pass

    ##########################################################################
    # Template weights
    ##########################################################################
    weights_age_met = pp_age_met.weights
    weights_age_met = np.reshape(
        weights_age_met[~gas_component], (nmetals, nages))
    weights_age_met /= np.nansum(weights_age_met)

    weights_kin = pp_kin.weights
    weights_kin = np.reshape(weights_kin, (nmetals, nages))
    weights_kin /= np.nansum(weights_kin)
    # np.save(os.path.join(data_dir, f"{obj_name}_mass_weights.npy"), weights_mass_weighted_metallicity_summed)
    # np.save(os.path.join(data_dir, f"ppxf_ages.npy"), ages)
    # When you load this, you must use the pickle option:
    #    np.load(os.path.join(data_dir, f"{obj_name}_mass_weights.npy"), allow_pickle=True)

    # ##########################################################################
    # # Insert your own code here to do stuff with the SFH 
    # # (e.g. calculate the mass-weighted mean age etc.)
    # ##########################################################################
    # if np.all(weights_mass_weighted_metallicity_summed <= 0):
    #     print(f"ERROR processing {obj_name}: all weights are <= 0!")
    #     t_start_of_SB = np.nan
    #     t_end_of_SB = np.nan
    #     idx_end_of_SB = np.nan
    #     idx_start_of_SB = np.nan
    # else:
    #     # Case: star formation is still ongoing
    #     if weights_mass_weighted_metallicity_summed[0] > 0:
    #         idx_end_of_SB = 0
    #         t_end_of_SB = 0
    #         if np.any(weights_mass_weighted_metallicity_summed == 0):
    #             idx_start_of_SB = np.argwhere(weights_mass_weighted_metallicity_summed == 0)[0][0]
    #             t_start_of_SB = ages[weights_mass_weighted_metallicity_summed == 0][0] / 1e6  # in Myr
    #         else:
    #             idx_start_of_SB = len(ages - 1)
    #             t_start_of_SB = ages[-1]
    #     else:
    #         # Other case: star formation has ceased
    #         idx_end_of_SB = np.argwhere(weights_mass_weighted_metallicity_summed == 0)[0][0]
    #         t_end_of_SB = ages[weights_mass_weighted_metallicity_summed == 0][0] / 1e6  # in Myr
    #         if len(np.argwhere(weights_mass_weighted_metallicity_summed == 0)) >= 2:
    #             idx_start_of_SB = np.argwhere(weights_mass_weighted_metallicity_summed == 0)[1][0]        
    #             t_start_of_SB = ages[weights_mass_weighted_metallicity_summed == 0][1] / 1e6  # in Myr
    #         else:
    #             idx_start_of_SB = len(ages - 1)
    #             t_start_of_SB = ages[-1]            

    # # mass-weighted mean stellar age in Myr
    # mass_average_age = (np.nansum(weights_mass_weighted_metallicity_summed * ages) / np.nansum(weights_mass_weighted)) / 1e6
    
    # mask = np.zeros(ages.shape, dtype="bool")
    # mask[ages > 1.0e8] = True # mask out all the values that have ages larger than 100 Myr or 1e8 yrs 
    # mass_array_below_100Myr = np.ma.masked_array(weights_mass_weighted_metallicity_summed, mask=mask)
    # total_mass_below_100Myr = np.nansum(mass_array_below_100Myr)
    # total_mass_fraction_below_100Myr = total_mass_below_100Myr / mass_tot

    # mask = np.zeros(ages.shape, dtype="bool")
    # mask[ages > 2.5e8] = True # mask out all the values that have ages larger than 100 Myr or 1e8 yrs 
    # mass_array_below_250Myr = np.ma.masked_array(weights_mass_weighted_metallicity_summed, mask=mask)
    # total_mass_below_250Myr = np.nansum(mass_array_below_250Myr)
    # total_mass_fraction_below_250Myr = total_mass_below_250Myr / mass_tot

    # mask = np.zeros(ages.shape, dtype="bool")
    # mask[ages > 1e9] = True # mask out all the values that have ages larger than 100 Myr or 1e8 yrs 
    # mass_array_below_1Gyr = np.ma.masked_array(weights_mass_weighted_metallicity_summed, mask=mask)
    # total_mass_below_1Gyr = np.nansum(mass_array_below_1Gyr)
    # total_mass_fraction_below_1Gyr = total_mass_below_1Gyr / mass_tot

    ##########################################################################
    # Plotting the fit
    ##########################################################################
    if plotit:
        plot_wrapper(pp)

    # Close the PDF files
    # if plotit:
    #     pp.close()
    #     if auto_adjust_regul:
    #         pp_regul.close()

    # ##########################################################################
    # # Save to FITS file
    # ##########################################################################
    # hdulist = []
    # hdulist.append(fits.PrimaryHDU())
    # hdulist[0].header['NAXIS'] = 1
    # hdulist[0].header['OBJECT'] = obj_name
    # hdulist[0].header['FNAME'] = input_fits_fname
    # hdulist[0].header['ISOCHRN'] = isochrones
    # hdulist[0].header['REGUL'] = regul
    # hdulist[0].header["NRE"] = r

    # # Wavelength information
    # # Because the FITS standard only allows linear axis values, we store the 
    # # log of the rebinned wavelength values since these will be evenly spaced.
    # hdulist[0].header['NAXIS1'] = len(lambda_vals_log)
    # hdulist[0].header['CRPIX1'] = 1
    # hdulist[0].header['CDELT1'] = lambda_vals_log[1] - lambda_vals_log[0]
    # hdulist[0].header['CUNIT1'] = 'log Angstroms'
    # hdulist[0].header['CTYPE1'] = 'Wavelength'
    # hdulist[0].header['CRVAL1'] = lambda_vals_log[0]

    # # Storing other information
    # hdulist.append(fits.ImageHDU(data=spec_log * norm,  #binned_spec_cube_log,
    #                              name="Integrated spectrum (log)"))
    # hdulist.append(fits.ImageHDU(data=spec_err_log * norm,
    #                              name="Integrated spectrum (log) errors"))
    # hdulist.append(fits.ImageHDU(data=(pp_age_met.bestfit - pp_age_met.gas_bestfit) * norm,
    #                              name="Best fit spectrum (log)"))

    # # Other stuff we need to keep
    # hdulist[0].header["SNR"] = SNR
    # hdulist[0].header["R_V"] = 3.1
    # hdulist[0].header["A_V"] = pp_age_met.gas_reddening * 3.1 if pp_age_met.gas_reddening is not None else None
    # hdulist[0].header["V"] = pp_kin.sol[0]
    # hdulist[0].header["V_ERR"] = pp_kin.error[0]  # NOTE: unreliable error estimate (see ppxf documentation)
    # hdulist[0].header["VD"] = pp_kin.sol[1]
    # hdulist[0].header["VD_ERR"] = pp_kin.error[1]  # NOTE: unreliable error estimate (see ppxf documentation)

    # # Gas kinematics
    # for n in range(1, ncomponents):
    #     hdulist[0].header["VG{}".format(n)] = pp_age_met.sol[n][0]
    #     hdulist[0].header["VG{}_ERR".format(n)] = pp_age_met.error[n][0]
    #     hdulist[0].header["VGD{}".format(n)] = pp_age_met.sol[n][1]
    #     hdulist[0].header["VGD{}ERR".format(n)] = pp_age_met.error[n][1]

    # hdulist = fits.HDUList(hdulist)

    # # Save to file
    # hdulist.writeto(output_fits_fname, overwrite=True)

    # ##########################################################################
    # # Save to DataFrame in CSV file
    # ##########################################################################
    # df_output.loc[obj_name, "ppxf_success"] = True if np.any(weights_mass_weighted_metallicity_summed > 0) else False
    # df_output.loc[obj_name, "t_start_of_SB (Myr)"] = t_start_of_SB
    # df_output.loc[obj_name, "t_end_of_SB (Myr)"] = t_end_of_SB
    # df_output.loc[obj_name, "mass_average_age (Myr)"] = mass_average_age
    # df_output.loc[obj_name, "total_mass_below_100Myr (M_sun)"] = total_mass_below_100Myr
    # df_output.loc[obj_name, "total_mass_fraction_below_100Myr"] = total_mass_fraction_below_100Myr
    # df_output.loc[obj_name, "total_mass_below_250Myr (M_sun)"] = total_mass_below_250Myr
    # df_output.loc[obj_name, "total_mass_fraction_below_250Myr"] = total_mass_fraction_below_250Myr
    # df_output.loc[obj_name, "mass_tot (M_sun)"] = mass_tot

    # for aa, age in enumerate(ages):
    #     df_output.loc[obj_name, f"mass (M_sun) in template t = {age / 1e6:.2e} yr"] = weights_mass_weighted_metallicity_summed[aa]

    # df_output.loc[obj_name, "SNR"] = SNR
    # df_output.loc[obj_name, "R_V"] = 3.1
    # df_output.loc[obj_name, "A_V"] = pp_age_met.gas_reddening * 3.1 if pp_age_met.gas_reddening is not None else None
    # df_output.loc[obj_name, "stellar_velocity (km/s)"] = pp_kin.sol[0]
    # df_output.loc[obj_name, "stellar_velocity_error (km/s)"] = pp_kin.error[0]  # NOTE: unreliable error estimate (see ppxf documentation)
    # df_output.loc[obj_name, "stellar_velocity_dispersion (km/s)"] = pp_kin.sol[1]
    # df_output.loc[obj_name, "stellar_velocity_dispersion_error (km/s)"] = pp_kin.error[1]  # NOTE: unreliable error estimate (see ppxf documentation)

    # # Gas kinematics
    # for n in range(1, ncomponents):
    #     df_output.loc[obj_name, "gas_velocity_component_{} (km/s)".format(n)] = pp_age_met.sol[n][0]
    #     df_output.loc[obj_name, "gas_velocity_component_{}_error (km/s)".format(n)] = pp_age_met.error[n][0]
    #     df_output.loc[obj_name, "gas_velocity_dispersion_component_{} (km/s)".format(n)] = pp_age_met.sol[n][1]
    #     df_output.loc[obj_name, "gas_velocity_dispersion_component_{}_error (km/s)".format(n)] = pp_age_met.error[n][1]

    # df_output.to_csv(os.path.join(data_dir, "S7_ppxf_data.csv"))


