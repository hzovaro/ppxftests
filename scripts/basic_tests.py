"""
Run ppxf on Phil's simulated galaxies to test how well we can recover 
parameters such as the mass-weighted age, etc. in the best-case scenario - 
i.e., high S/N, no emission lines, etc. 

Save the output figures to 
/priv/meggs3/u5708159/ppxftests/figs/basic_tests/ga<galaxy no.>.pdf

Save the DataFrame to 
/priv/meggs3/u5708159/ppxftests/figs/basic_tests/summary.csv

"""
import os
import numpy as np
from numpy.random import RandomState
from time import time 
from tqdm.notebook import tqdm
import multiprocessing
import pandas as pd

from astropy.io import fits

from ppxftests.run_ppxf import run_ppxf
from ppxftests.ssputils import load_ssp_templates, get_bin_edges_and_widths
from ppxftests.mockspec import create_mock_spectrum
from ppxftests.sfhutils import load_sfh, convert_mass_weights_to_light_weights
from ppxftests.sfhutils import compute_mw_age, compute_lw_age, compute_sfr_thresh_age, compute_sb_zero_age, compute_mass
from ppxftests.ppxf_plot import plot_sfh_mass_weighted, plot_sfh_light_weighted

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

fig_path = "/priv/meggs3/u5708159/ppxftests/figs/"

###########################################################################
# Settings
###########################################################################
isochrones = "Padova"
sigma_star_kms = 300
SNR = 200
z = 0.01

niters = 100
nthreads = 20

# For computing mean ages, etc.
age_thresh_vals = [None, 1e7, 1e8, 1e9, None]
sfr_thresh = 1

# List of galaxies 
gals = list(range(0, 20))

# DataFrame for storing results 
df = pd.DataFrame(index=gals)
df.index.name = "ID"

# Load the stellar templates so we can get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates(isochrones)
N_ages = len(ages)
N_metallicities = len(metallicities)

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

###########################################################################
# Convenience functions for computing mean quantities from a list of ppxf instances 
###########################################################################
def compute_mean_1D_sfh(pp_list, weighttype):
    """
    Convenience function for computing the mean SFH given a list of ppxf
    instances.
    """
    assert weighttype == "lw" or weighttype == "mw",\
        "weighttype must be 'lw' or 'mw'!"

    if weighttype == "lw":
        sfh_list = [pp.sfh_lw_1D for pp in pp_list]
    elif weighttype == "mw":
        sfh_list = [pp.sfh_mw_1D for pp in pp_list]
    sfh_1D_mean = np.nansum(np.array(sfh_list), axis=0) / len(sfh_list)

    return sfh_1D_mean

def compute_mean_mass(pp_list, age_thresh_lower, age_thresh_upper):
    """
    Convenience function for computing the mean & std. dev. of the total mass
    in the range [age_thresh_lower, age_thresh_upper] given a list of ppxf instances.
    """
    sfh_list = [pp.weights_mass_weighted for pp in pp_list]
    mass_list = [compute_mass(sfh_mw, isochrones, age_thresh_lower, age_thresh_upper) for sfh_mw in sfh_list]
    mass_mean = np.nanmean(mass_list)
    mass_std = np.nanstd(mass_list)
    return mass_mean, mass_std


def compute_mean_sfr(pp_list):
    """
    Convenience function for computing the mean SFR given a list of ppxf
    instances.
    """
    sfr_list = [pp.sfr_mean for pp in pp_list]
    sfr_mean = np.nansum(np.array(sfr_list), axis=0) / len(sfr_list)
    return sfr_mean

def compute_mean_age(pp_list, weighttype, age_thresh_lower, age_thresh_upper):
    """
    Convenience function for computing the mean & std. dev. of the mass-
    weighted age in the range [age_thresh_lower, age_thresh_upper] given 
    a list of ppxf instances.
    """
    assert weighttype == "lw" or weighttype == "mw",\
        "weighttype must be 'lw' or 'mw'!"
    
    if weighttype == "mw":
        sfh_list = [pp.weights_mass_weighted for pp in pp_list]
        age_list = [10**compute_mw_age(sfh, isochrones, age_thresh_lower, age_thresh_upper)[0] for sfh in sfh_list]
        age_mean = np.nanmean(age_list)
        age_std = np.nanstd(age_list)
        
    elif weighttype == "lw":
        sfh_list = [pp.weights_light_weighted for pp in pp_list]
        age_list = [10**compute_lw_age(sfh, isochrones, age_thresh_lower, age_thresh_upper)[0] for sfh in sfh_list]
        age_mean = np.nanmean(age_list)
        age_std = np.nanstd(age_list)
    
    return age_mean, age_std

def compute_mean_sfr_thresh_age(pp_list, sfr_thresh):
    """
    Convenience function for computing the mean and std. dev. in the
    SFR threshold age from a list of ppxf instances.
    """
    sfr_age_list = [10**compute_sfr_thresh_age(pp.weights_mass_weighted, sfr_thresh, isochrones)[0] for pp in pp_list]
    sfr_age_mean = np.nanmean(sfr_age_list)
    sfr_age_std = np.nanstd(sfr_age_list)

    return sfr_age_mean, sfr_age_std

###########################################################################
# Define the SFH
###########################################################################
for gal in gals[:1]:

    sfh_mw_input, sfh_lw_input, sfr_avg_input = load_sfh(gal=gal, plotit=True)
    sfh_mw_1D_input = np.nansum(sfh_mw_input, axis=0)
    sfh_lw_1D_input = np.nansum(sfh_lw_input, axis=0)

    # Compute truth values 
    age_sfr_input = 10**compute_sfr_thresh_age(sfh_mw_input, sfr_thresh, isochrones=isochrones)[0]  # SFR threshold
    age_sb_input = 10**compute_sb_zero_age(sfh_mw_input, isochrones=isochrones)[0]  # "Starburst" age
    df.loc[gal, f"SFR age (>{sfr_thresh} Msun yr^-1) (truth)"] = age_sfr_input
    df.loc[gal, f"SB age (truth)"] = age_sb_input

    for aa in range(len(age_thresh_vals)- 1):
        age_thresh_lower = age_thresh_vals[aa]
        age_thresh_upper = age_thresh_vals[aa + 1]

        # Determine age boundaries
        if age_thresh_lower is None:
            age_thresh_lower = ages[0]
        if age_thresh_upper is None:
            age_thresh_upper = ages[-1]
        
        # Compute mass- and light-weighted mean ages, and the total mass too
        age_mw_input = 10**compute_mw_age(sfh_mw_input, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)[0]
        age_lw_input = 10**compute_lw_age(sfh_lw_input, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)[0]
        mass = compute_mass(sfh_mw_input, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)
        
        # Store in DataFrame
        df.loc[gal, f"MW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (truth)"] = age_mw_input
        df.loc[gal, f"LW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (truth)"] = age_lw_input
        df.loc[gal, f"Mass {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (truth)"] = mass

    ###########################################################################
    # Create spectrum
    ###########################################################################
    spec, spec_err, lambda_vals_A = create_mock_spectrum(
        sfh_mass_weighted=sfh_mw_input,
        agn_continuum=False,
        isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
        plotit=True)

    ###########################################################################
    # Run ppxf WITHOUT regularisation, using a MC approach
    ###########################################################################
    # Input arguments
    seeds = list(np.random.randint(low=0, high=100 * niters, size=niters))
    args_list = [[s, spec, spec_err, lambda_vals_A] for s in seeds]

    # Run in parallel
    print(f"Running ppxf on {nthreads} threads...")
    t = time()
    with multiprocessing.Pool(nthreads) as pool:
        pp_list = list(tqdm(pool.imap(ppxf_helper, args_list), total=niters))
    print(f"Elapsed time in ppxf: {time() - t:.2f} s")

    ###########################################################################
    # Compute average quantities from the MC simulations
    ###########################################################################
    # Compute the mean SFH and SFR from the lists of MC runs
    sfh_MC_lw_1D_mean = compute_mean_1D_sfh(pp_list, "lw")
    sfh_MC_mw_1D_mean = compute_mean_1D_sfh(pp_list, "mw")
    sfr_avg_MC = compute_mean_sfr(pp_list)

    # Compute the "SFR age"
    age_sfr_mean, age_sfr_std = compute_mean_sfr_thresh_age(pp_list, sfr_thresh)
    df.loc[gal, f"SFR age (>{sfr_thresh} Msun yr^-1) (MC) mean"] = age_sfr_mean
    df.loc[gal, f"SFR age (>{sfr_thresh} Msun yr^-1) (MC) std. dev."] = age_sfr_std

    # Compute the mean mass- and light-weighted ages plus the total mass in a series of age ranges
    for aa in range(len(age_thresh_vals) - 1):
        age_thresh_lower = age_thresh_vals[aa]
        age_thresh_upper = age_thresh_vals[aa + 1]
        
        if age_thresh_lower is None:
            age_thresh_lower = ages[0]
        if age_thresh_upper is None:
            age_thresh_upper = ages[-1]
            
        # Compute the mean mass- and light-weighted ages plus the total mass in this age range
        age_lw_mean, age_lw_std = compute_mean_age(pp_list, "lw", age_thresh_lower, age_thresh_upper)
        age_mw_mean, age_mw_std = compute_mean_age(pp_list, "mw", age_thresh_lower, age_thresh_upper)
        mass_mean, mass_std = compute_mean_mass(pp_list, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)

        # Put in DataFrame
        df.loc[gal, f"MW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (MC) mean"] = age_mw_mean
        df.loc[gal, f"LW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (MC) mean"] = age_lw_mean
        df.loc[gal, f"Mass {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (MC) mean"] = mass_mean
        df.loc[gal, f"MW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (MC) std. dev."] = age_mw_std
        df.loc[gal, f"LW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (MC) std. dev."] = age_lw_std
        df.loc[gal, f"Mass {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (MC) std. dev."] = mass_std

    ###########################################################################
    # Run ppxf with regularisation
    ###########################################################################
    t = time()
    pp_regul = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                  z=z, ngascomponents=1,
                  regularisation_method="auto",
                  isochrones=isochrones,
                  fit_gas=False, tie_balmer=True,
                  delta_regul_min=1, regul_max=5e4, delta_delta_chi2_min=1,
                  plotit=False, savefigs=False, interactive_mode=False)
    print(f"Total time in run_ppxf: {time() - t:.2f} seconds")

    ###########################################################################
    # Compute quantities from the regularised fit
    ###########################################################################
    # Get the SFH and SFR
    sfh_regul_mw_1D = pp_regul.sfh_mw_1D
    sfh_regul_lw_1D = pp_regul.sfh_lw_1D
    sfr_avg_regul = pp_regul.sfr_mean

    age_sfr_regul = 10**compute_sfr_thresh_age(sfh_regul_lw_1D, sfr_thresh, isochrones=isochrones)[0]  # SFR threshold
    age_sb_regul = 10**compute_sb_zero_age(sfh_regul_lw_1D, isochrones=isochrones)[0]  # "Starburst" age
    df.loc[gal, f"SFR age (>{sfr_thresh} Msun yr^-1) (regul)"] = age_sfr_regul
    df.loc[gal, f"SB age (regul)"] = age_sb_regul

    for aa in range(len(age_thresh_vals)- 1):
        age_thresh_lower = age_thresh_vals[aa]
        age_thresh_upper = age_thresh_vals[aa + 1]

        # Determine age boundaries
        if age_thresh_lower is None:
            age_thresh_lower = ages[0]
        if age_thresh_upper is None:
            age_thresh_upper = ages[-1]
        
        # Compute mass- and light-weighted mean ages, and the total mass too
        age_mw_regul = 10**compute_mw_age(sfh_regul_lw_1D, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)[0]
        age_lw_regul = 10**compute_lw_age(sfh_regul_mw_1D, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)[0]
        mass_regul = compute_mass(sfh_regul_mw_1D, isochrones=isochrones, age_thresh_lower=age_thresh_lower, age_thresh_upper=age_thresh_upper)
        
        # Store in DataFrame
        df.loc[gal, f"MW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (regul)"] = age_mw_regul
        df.loc[gal, f"LW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (regul)"] = age_lw_regul
        df.loc[gal, f"Mass {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (regul)"] = mass_regul

    ###########################################################################
    # Plot the input mass- and light-weighted SFHs
    ###########################################################################
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 15))
    fig.subplots_adjust(hspace=0.35)
    log_scale = True
    info_str = r"$S/N = %d, z = %.3f, \sigma_* = %d\rm\, km\,s^{-1}$" % (SNR, z, sigma_star_kms)
    for ax, weighttype in zip(axs, ["mw", "lw", "sfr"]):
        # Plot the SFHs from each ppxf run, plus the "truth" SFH
        if weighttype == "mw":
            ax.set_title(f"Galaxy {gal:004}: mass-weighted template weights ({info_str})")
            ax.fill_between(ages, sfh_mw_1D_input, step="mid", alpha=0.5, color="lightblue", label="Input SFH")
            ax.step(ages, sfh_MC_mw_1D_mean, color="red", where="mid", label="Mean ppxf fit (MC simulations)", alpha=0.5)
            ax.step(ages, sfh_regul_mw_1D, color="green", where="mid", label="Mean ppxf fit (regularised fit)", alpha=0.5)
            ax.set_ylim([1e3, None])
            ax.set_ylabel(r"MW template weight ($\rm M_\odot$)")
        elif weighttype == "lw":
            ax.set_title(f"Galaxy {gal:004}: light-weighted template weights ({info_str})")
            ax.fill_between(ages, sfh_lw_1D_input, step="mid", alpha=0.5, color="lightblue", label="Input SFH")
            ax.step(ages, sfh_MC_lw_1D_mean, color="red", where="mid", label="Mean ppxf fit (MC simulations)", alpha=0.5)
            ax.step(ages, sfh_regul_lw_1D, color="green", where="mid", label="Mean ppxf fit (regularised fit)", alpha=0.5)
            ax.set_ylim([1e35, None])
            ax.set_ylabel(r"LW template weight ($\rm erg\,s^{-1}\,Å^{-1}$)")
        elif weighttype == "sfr":
            ax.set_title(f"Galaxy {gal:004}: mean SFR ({info_str})")
            ax.fill_between(ages, sfr_avg_input, step="mid", alpha=0.5, color="lightblue", label="Input SFH")
            ax.step(ages, sfr_avg_MC, color="red", where="mid", label="Mean ppxf fit (MC simulations)", alpha=0.5)
            ax.step(ages, sfr_avg_regul, color="green", where="mid", label="Mean ppxf fit (regularised fit)", alpha=0.5)
            ax.set_ylim([1e-2, None])
            ax.set_ylabel(r"Mean SFR ($\rm M_\odot\,yr^{-1}$)")

        # Plot horizontal error bars indicating the SFR threshold age from the MC simulations
        y1, y2 = ax.get_ylim()
        y = 10**(0.9 * (np.log10(y2) - np.log10(y1)) + np.log10(y1)) if log_scale else 0.9 * y2
        ax.errorbar(x=df.loc[gal, f"SFR age (>{sfr_thresh} Msun yr^-1) (MC) mean"],
                    xerr=df.loc[gal, f"SFR age (>{sfr_thresh} Msun yr^-1) (MC) std. dev."],
                    y=y, 
                    marker="*", mfc="orange", mec="orange", ecolor="orange", linestyle="none", capsize=10, markersize=10,
                    label="SFR age (mean, MC simulations)")
        ax.errorbar(x=df.loc[gal, f"SFR age (>{sfr_thresh} Msun yr^-1) (truth)"], xerr=0, markersize=10,
                    y=y, 
                    marker="*", mfc="lightblue", mec="blue", ecolor="lightblue", linestyle="none",
                    label="SFR age (input)")
        ax.errorbar(x=df.loc[gal, f"SFR age (>{sfr_thresh} Msun yr^-1) (regul)"], xerr=0, markersize=10,
                    y=y, 
                    marker="*", mfc="lightgreen", mec="green", ecolor="green", linestyle="none",
                    label="SFR age (regularised fit)")

        # Plot horizontal error bars indicating the mean mass- and light-weighted ages ages from the MC simulations
        for aa in range(len(age_thresh_vals) - 1):
            age_thresh_lower = age_thresh_vals[aa]
            age_thresh_upper = age_thresh_vals[aa + 1]

            if age_thresh_lower is None:
                age_thresh_lower = ages[0]
            if age_thresh_upper is None:
                age_thresh_upper = ages[-1]
            
            # mass-weighted age
            y = 10**(0.8 * (np.log10(y2) - np.log10(y1)) + np.log10(y1)) if log_scale else 0.8 * y2
            ax.errorbar(x=df.loc[gal, f"MW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (MC) mean"],
                        xerr=df.loc[gal, f"MW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (MC) std. dev."],
                        y=y,
                        marker="D", mfc="red", mec="red", ecolor="red", linestyle="none", capsize=10,
                        label="Mean MW age in range (MC simulations)" if aa == 0 else None)
            ax.errorbar(x=df.loc[gal, f"MW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (regul)"], xerr=0,
                        y=y,
                        marker="D", mfc="lightgreen", mec="green", ecolor="green", linestyle="none",
                        label="Mean MW age in range (regularised fit)" if aa == 0 else None)
            ax.errorbar(x=df.loc[gal, f"MW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (truth)"], xerr=0,
                        y=y,
                        marker="D", mfc="lightblue", mec="blue", ecolor="lightblue", linestyle="none",
                        label="Mean MW age in range (input)" if aa == 0 else None)

            # light-weighted age
            y = 10**(0.7 * (np.log10(y2) - np.log10(y1)) + np.log10(y1)) if log_scale else 0.7 * y2
            ax.errorbar(x=df.loc[gal, f"LW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (MC) mean"],
                        xerr=df.loc[gal, f"LW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (MC) std. dev."],
                        y=y,
                        marker="X", mfc="red", mec="red", ecolor="red", linestyle="none", capsize=10,
                        label="Mean LW age in range (MC simulations)" if aa == 0 else None)
            ax.errorbar(x=df.loc[gal, f"LW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (regul)"], xerr=0,
                        y=y,
                        marker="X", mfc="lightgreen", mec="green", ecolor="green", linestyle="none",
                        label="Mean LW age in range (regularised fit)" if aa == 0 else None)
            ax.errorbar(x=df.loc[gal, f"LW age {np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f} (truth)"], xerr=0,
                        y=y,
                        marker="X", mfc="lightblue", mec="blue", ecolor="lightblue", linestyle="none",
                        label="Mean LW age in range (input)" if aa == 0 else None)

            ax.axvline(age_thresh_lower, color="black", linestyle="--", label="Age range" if aa == 0 else None)
            ax.axvline(age_thresh_upper, color="black", linestyle="--")

        # Decorations 
        ax.autoscale(axis="x", enable=True, tight=True)
        ax.set_xlabel("Age (Myr)")
        ax.legend(fontsize="small", loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax.set_yscale("log") if log_scale else None
        ax.set_xscale("log")
        ax.grid()
    
    # Save figure    
    fig.savefig(os.path.join(fig_path, "basic_tests", f"ga{gal:004}.pdf"), format="pdf", bbox_inches="tight")

    # Save DataFrame to file 
    # Do this every iteration in case it breaks! 
    df.to_csv(os.path.join(fig_path, "basic_tests", "summary.csv"))
