###############################################################################
#
#   File:       plot_ppxf_summary.py
#   Author:     Henry Zovaro
#   Email:      henry.zovaro@anu.edu.au
#
#   Description:
#   Make nice plots from a ppxf instance.
#
#   Copyright (C) 2021 Henry Zovaro
#
###############################################################################
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.debugger import Tracer

from ppxftests.ssputils import load_ssp_templates

###################################################################
def plot_ppxf_summary(sfh_lw_input, sfh_mw_input,
                      pp_regul, pp_mc_list,
                      isochrones):
    """
    Given 
        - an input SFH
        - an instance from a regularised ppxf run
        - a list of instances from ppxf runs (i.e. MC simulations)
    make a series of plots:
        - the light- and mass-weighted SFHs 
        - light- and mass-weighted mean ages as a function of cutoff age 
        - cumulative light- and mass- fractions
    
    """
    # Get the age & metallicity dimensions
    _, _, metallicities, ages = load_ssp_templates(isochrones)
    N_ages = len(ages)
    N_metallicities = len(metallicities)

    # Get the measured SFHs
    sfh_lw_1D_input = np.nansum(sfh_lw_input, axis=0)
    sfh_mw_1D_input = np.nansum(sfh_mw_input, axis=0)
    sfh_lw_1D_MC = compute_mean_1D_sfh(pp_mc_list, isochrones, weighttype="lw")
    sfh_mw_1D_MC = compute_mean_1D_sfh(pp_mc_list, isochrones, weighttype="mw")
    sfh_lw_1D_regul = pp_regul.sfh_lw_1D
    sfh_mw_1D_regul = pp_regul.sfh_mw_1D

    # Compute weighted ages
    ages_lw_input = [10**compute_lw_age(sfh_lw_1D_input, isochrones=isochrones, age_thresh_lower=None, age_thresh_upper=age_thresh_upper)[0] for age_thresh_upper in ages[1:]]
    ages_lw_MC = [10**compute_lw_age(sfh_lw_1D_MC, isochrones=isochrones, age_thresh_lower=None, age_thresh_upper=age_thresh_upper)[0] for age_thresh_upper in ages[1:]]
    ages_lw_regul = [10**compute_lw_age(sfh_lw_1D_regul, isochrones=isochrones, age_thresh_lower=None, age_thresh_upper=age_thresh_upper)[0] for age_thresh_upper in ages[1:]]
    
    ages_mw_input = [10**compute_mw_age(sfh_mw_1D_input, isochrones=isochrones, age_thresh_lower=None, age_thresh_upper=age_thresh_upper)[0] for age_thresh_upper in ages[1:]]
    ages_mw_MC = [10**compute_mw_age(sfh_mw_1D_MC, isochrones=isochrones, age_thresh_lower=None, age_thresh_upper=age_thresh_upper)[0] for age_thresh_upper in ages[1:]]
    ages_mw_regul = [10**compute_mw_age(sfh_mw_1D_regul, isochrones=isochrones, age_thresh_lower=None, age_thresh_upper=age_thresh_upper)[0] for age_thresh_upper in ages[1:]]

    # Compute weighted ages
    masses_input = [compute_mass(sfh_mw_1D_input, isochrones=isochrones, age_thresh_lower=None, age_thresh_upper=age_thresh_upper) for age_thresh_upper in ages[1:]]
    masses_MC = [compute_mass(sfh_mw_1D_MC, isochrones=isochrones, age_thresh_lower=None, age_thresh_upper=age_thresh_upper) for age_thresh_upper in ages[1:]]
    masses_regul = [compute_mass(sfh_mw_1D_regul, isochrones=isochrones, age_thresh_lower=None, age_thresh_upper=age_thresh_upper) for age_thresh_upper in ages[1:]]

    ###################################################################
    # Plot the SFH
    ###################################################################
    M_tot = np.nansum(sfh_mw_input)
    # plot_sfh_mass_weighted(sfh_mw, ages, metallicities)

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))

    axs[0].step(x=ages, y=sfh_mw_1D_input / M_tot, where="mid", color="blue", label="SFH (input)", linewidth=2.5)
    axs[0].step(x=ages, y=sfh_mw_1D_regul / M_tot, where="mid", color="indigo", label="SFH (regularised)", linewidth=1.0)
    axs[0].step(x=ages, y=sfh_mw_1D_MC / M_tot, where="mid", color="cornflowerblue", label="SFH (MC)", linewidth=1.0)
    for pp in pp_mc_list:
        axs[0].step(x=ages, y=pp.sfh_mw_1D / M_tot, where="mid", color="lightblue", alpha=0.1, linewidth=0.25)
    axs[0].axhline(1e-4, color="k", ls="--", linewidth=1)
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].legend(loc="upper left", fontsize="x-small")
    axs[0].grid()
    axs[0].set_ylabel("Mass fraction")
    axs[0].set_xlabel("Age (yr)")
    axs[0].autoscale(axis="x", tight=True, enable=True)

    axs[1].step(x=ages, y=sfh_lw_1D_input, where="mid", color="blue", label="SFH (input)", linewidth=2.5)
    axs[1].step(x=ages, y=sfh_lw_1D_regul, where="mid", color="indigo", label="SFH (regularised)", linewidth=1.0)
    axs[1].step(x=ages, y=sfh_lw_1D_MC, where="mid", color="cornflowerblue", label="SFH (MC)", linewidth=1.0)
    for pp in pp_mc_list:
        axs[1].step(x=ages, y=pp.sfh_lw_1D, where="mid", color="lightblue", alpha=0.1, linewidth=0.25)
    axs[1].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].legend(loc="upper left", fontsize="x-small")
    axs[1].grid()
    axs[1].set_ylabel("Light weight (erg/s)")
    axs[1].set_xlabel("Age (yr)")
    axs[1].autoscale(axis="x", tight=True, enable=True)

    ###################################################################
    # Plot the mean mass- and light-weighted age vs. age threshold
    ###################################################################
    # Plot the "true" values
    fig, ax = plt.subplots()
    ax.step(x=ages[1:], y=ages_mw_input, label=f"Mass-weighted age", where="mid", linewidth=2.5, color="green")
    ax.step(x=ages[1:], y=ages_lw_input, label=f"Light-weighted age", where="mid", linewidth=2.5, color="red")

    # Plot the values measured from the ppxf runs
    ax.step(x=ages[1:], y=ages_lw_regul, label=f"Measured light-weighted age (regularised)", where="mid", color="maroon")
    ax.step(x=ages[1:], y=ages_lw_MC, label=f"Measured light-weighted age (MC)", where="mid", color="orange")
    ax.step(x=ages[1:], y=ages_mw_regul, label=f"Measured mass-weighted age (regularised)", where="mid", color="darkgreen")
    ax.step(x=ages[1:], y=ages_mw_MC, label=f"Measured mass-weighted age (MC)", where="mid", color="lightgreen")

    # Decorations    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize="x-small")
    ax.set_xlabel("Age threshold (yr)")
    ax.set_ylabel("Weighted mean age below threshold (yr)")
    ax.set_xlim([ages[0], ages[-1]])
    ax.grid()

    ###################################################################
    # Also plot cumulative mass expressed as a % so we can see what the threshold is at this S/N 
    ###################################################################
    # Plot the "true" values
    fig, ax = plt.subplots()
    ax.step(x=ages[1:], y=masses_input / M_tot, label=f"Cumulative mass", where="mid", linewidth=2.5, color="blue")

    # Plot 
    ax.step(x=ages[1:], y=masses_regul / M_tot, label=f"Cumulative mass (regularised)", where="mid", color="indigo")
    ax.step(x=ages[1:], y=masses_MC / M_tot, label=f"Cumulative mass (MC)", where="mid", color="cornflowerblue")

    # Decorations    
    ax.axhline(1e-4, color="k", ls="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize="x-small")
    ax.set_xlabel("Age (yr)")
    ax.set_ylabel("Cumulative mass fraction")
    ax.set_xlim([ages[0], ages[-1]])
    ax.grid()

    return