"""

Analyse the DataFrame output by basic_tests.py:
which age measure is the most accurate?

"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from ppxftests.ssputils import load_ssp_templates
from ppxftests.sfhutils import compute_mw_age, compute_lw_age, compute_mass

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.size"] = 10
plt.ion()
plt.close("all")

# from IPython.core.debugger import Tracer

df_path = "/priv/meggs3/u5708159/ppxftests/"

###############################################################################
# Load DataFrame 
###############################################################################
df_fname = os.path.join(df_path, "ppxf_output_tmp.hd5")
df = pd.read_hdf(df_fname)

###############################################################################
# Compute the mass- and light-weighted mean age for each galaxy,
# and store in the DataFrame
###############################################################################
# Load the stellar templates so we can get the age & metallicity dimensions
isochrones = "Padova"
_, _, metallicities, ages = load_ssp_templates(isochrones)
N_ages = len(ages)
N_metallicities = len(metallicities)

gals = df["ID"].unique()
df = df.set_index("ID")

age_thresh = 1e9
age_thresh_vals = [None, 1e9, None]
weighttypes = ["light", "mass"]

for gal in tqdm(gals):
    for aa in range(len(age_thresh_vals)- 1):
        # Compute light- and mass-weighted ages in each range
        for weighttype in weighttypes:
            age_thresh_lower = age_thresh_vals[aa]
            age_thresh_upper = age_thresh_vals[aa + 1]

            # Determine age boundaries
            if age_thresh_lower is None:
                age_thresh_lower = ages[0]
            if age_thresh_upper is None:
                age_thresh_upper = ages[-1]

            age_str = f"{np.log10(age_thresh_lower):.2f} < log t < {np.log10(age_thresh_upper):.2f}"

            # Compute the mass-weighted mean age from each SFH
            sfh_input = np.nansum(df.loc[gal, f"SFH - {weighttype} weighted (input)"], axis=0)
            sfh_MC = np.nansum(df.loc[gal, f"SFH - {weighttype} weighted (MC, mean)"], axis=0)
            sfh_MC_std = np.nansum(df.loc[gal, f"SFH - {weighttype} weighted (MC, std. dev.)"], axis=0)
            sfh_regul = np.nansum(df.loc[gal, f"SFH - {weighttype} weighted (regularised)"], axis=0)

            # Compute the weighted mean age < 1 Gyr
            if weighttype == "mass":
                age_fn = compute_mw_age
            elif weighttype == "light":
                age_fn = compute_lw_age

            log_age_input = age_fn(sfh_input, "Padova", age_thresh_lower, age_thresh_upper)[0]
            log_age_MC = age_fn(sfh_MC, "Padova", age_thresh_lower, age_thresh_upper)[0]
            log_age_regul = age_fn(sfh_regul, "Padova", age_thresh_lower, age_thresh_upper)[0]

            # Compute errors on the MC age 
            log_ages = []
            for ii in range(100):
                sfh = sfh_MC + np.random.normal(loc=0, scale=sfh_MC_std)
                log_age = age_fn(sfh, "Padova", age_thresh_lower, age_thresh_upper)[0]
                log_ages.append(log_age)
            log_age_MC_err = np.nanstd(log_ages)
            log_age_MC_mean = np.nanmean(log_ages)

            # Store in DataFrame
            df.loc[gal, f"log {weighttype} weighted age {age_str} (input)"] = log_age_input
            df.loc[gal, f"log {weighttype} weighted age {age_str} (regularised)"] = log_age_regul
            df.loc[gal, f"log {weighttype} weighted age {age_str} (MC) mean"] = log_age_MC_mean
            df.loc[gal, f"log {weighttype} weighted age {age_str} (MC) std. dev."] = log_age_MC_err

        ###############################################################################
        # Compute masses in each age range 
        ###############################################################################
        sfh_mw_input = np.nansum(df.loc[gal, f"SFH - {weighttype} weighted (input)"], axis=0)
        sfh_mw_MC = np.nansum(df.loc[gal, f"SFH - {weighttype} weighted (MC, mean)"], axis=0)
        sfh_mw_MC_std = np.nansum(df.loc[gal, f"SFH - {weighttype} weighted (MC, std. dev.)"], axis=0)
        sfh_mw_regul = np.nansum(df.loc[gal, f"SFH - {weighttype} weighted (regularised)"], axis=0)

        log_mass_input = np.log10(compute_mass(sfh_mw_input, "Padova", age_thresh_lower, age_thresh_upper))
        log_mass_MC = np.log10(compute_mass(sfh_mw_MC, "Padova", age_thresh_lower, age_thresh_upper))
        log_mass_regul = np.log10(compute_mass(sfh_mw_regul, "Padova", age_thresh_lower, age_thresh_upper))

        # Compute errors on the MC age 
        log_masses = []
        for ii in range(100):
            sfh = sfh_mw_MC + np.random.normal(loc=0, scale=sfh_mw_MC_std)
            log_mass = np.log10(compute_mass(sfh, "Padova", age_thresh_lower, age_thresh_upper))
            log_masses.append(log_mass)
        log_mass_MC_err = np.nanstd(log_masses)
        log_mass_MC_mean = np.nanmean(log_masses)

        # Store in DataFrame
        df.loc[gal, f"log mass {age_str} (input)"] = log_mass_input
        df.loc[gal, f"log mass {age_str} (regularised)"] = log_mass_regul
        df.loc[gal, f"log mass {age_str} (MC) mean"] = log_mass_MC_mean
        df.loc[gal, f"log mass {age_str} (MC) std. dev."] = log_mass_MC_err

###############################################################################
# Plot: difference between input & output in SB age, MW age, LW age, SFR age as a function of galaxy number
n_weighttypes = len(weighttypes)
weighttype = weighttypes[0]
n_age_ranges = len([c for c in df.columns if c.startswith(f"log {weighttype}") and c.endswith("(input)")])
age_range_strs = [s.split(f"log {weighttype} weighted age ")[1].split(" (input)")[0] for s in [c for c in df.columns if c.startswith(f"log {weighttype}") and c.endswith("(input)")]]

# Plot the input & estimated mass-, light- and SFR-weighted ages as a function of galaxy ID
for weighttype, marker in zip(weighttypes, ["x", "D", "o"]):
    # fig, axs = plt.subplots(nrows=n_age_ranges, ncols=2, figsize=(12, 3 * n_age_ranges))
    # fig.subplots_adjust(hspace=0, wspace=0.3)
    # fig.suptitle(f"{weighttype} weighted age")

    for aa in range(n_age_ranges):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
        fig.subplots_adjust(hspace=0)
        fig.suptitle(f"{weighttype} weighted age ({age_range_strs[aa]})")

        # Axis 1: absolute values 
        axs[0].errorbar(x=df.index, 
                            y=df[f"log {weighttype} weighted age {age_range_strs[aa]} (input)"],
                            marker=marker, mfc="lightblue", mec="blue", ecolor="lightblue", linestyle="none",
                            label="Input")
        axs[0].errorbar(x=df.index, 
                            y=df[f"log {weighttype} weighted age {age_range_strs[aa]} (regularised)"],
                            marker=marker, mfc="lightgreen", mec="green", ecolor="green", linestyle="none", markersize=3.5,
                            label="Regularised fit")
        axs[0].errorbar(x=df.index, 
                            y=df[f"log {weighttype} weighted age {age_range_strs[aa]} (MC) mean"],
                            yerr=df[f"log {weighttype} weighted age {age_range_strs[aa]} (MC) std. dev."],
                            marker=marker, mfc="red", mec="red", ecolor="red", alpha=0.5, linewidth=0.5, linestyle="none", markersize=3.5,
                            label="MC simulations")

        # Axis 2: relative values 
        axs[1].axhline(0, color="black")
        axs[1].errorbar(x=df.index, 
                            y=df[f"log {weighttype} weighted age {age_range_strs[aa]} (regularised)"] - df[f"log {weighttype} weighted age {age_range_strs[aa]} (input)"],
                            marker=marker, mfc="lightgreen", mec="green", ecolor="green", linestyle="none", markersize=3.5,
                            label="Regularised fit")
        axs[1].errorbar(x=df.index, 
                            y=df[f"log {weighttype} weighted age {age_range_strs[aa]} (MC) mean"] - df[f"log {weighttype} weighted age {age_range_strs[aa]} (input)"],
                            yerr=df[f"log {weighttype} weighted age {age_range_strs[aa]} (MC) std. dev."],
                            marker=marker, mfc="red", mec="red", ecolor="red", alpha=0.5, linewidth=0.5, linestyle="none", markersize=3.5,
                            label="MC simulations")
        # Decorations
        # axs[0].text(s=age_range_strs[aa], x=0.05, y=0.95, transform=axs[0].transAxes, horizontalalignment="left", verticalalignment="top")
        axs[0].set_ylabel(f"{weighttype} weighted age (log Myr)")
        axs[0].set_ylim([6, 10.5])
        axs[1].set_ylim([-1, 1])
        axs[1].set_ylabel(f"{weighttype} weighted age error (log Myr)")

        # Decorations
        for ax in axs.flat:
            ax.grid()
            ax.autoscale(axis="x", tight=True, enable=True)
        axs[0].legend(fontsize="small")
        axs[1].set_xlabel("Galaxy ID")

"""
# Repeat, but plot the estimated mass in each range
fig, axs = plt.subplots(nrows=n_age_ranges, ncols=2, figsize=(12, 3 * n_age_ranges))
fig.subplots_adjust(hspace=0)
fig.suptitle(f"Total mass")
for aa in range(n_age_ranges):
    # Axis 1: absolute values 
    axs[aa][0].errorbar(x=df.index, 
                        y=df[f"log mass {age_range_strs[aa]} (input)"],
                        marker=marker, mfc="lightblue", mec="blue", ecolor="lightblue", linestyle="none",
                        label="input")
    axs[aa][0].errorbar(x=df.index, 
                        y=df[f"log mass {age_range_strs[aa]} (regularised)"],
                        marker=marker, mfc="lightgreen", mec="green", ecolor="green", linestyle="none", markersize=3.5,
                        label="Regularised fit")
    axs[aa][0].errorbar(x=df.index, 
                        y=df[f"log mass {age_range_strs[aa]} (MC) mean"],
                        yerr=df[f"log mass {age_range_strs[aa]} (MC) std. dev."],
                        marker=marker, mfc="red", mec="red", ecolor="red", linestyle="none", markersize=3.5,
                        label="MC simulations")

    # Axis 2: relative values 
    axs[aa][1].axhline(0, color="black")
    axs[aa][1].errorbar(x=df.index, 
                        y=df[f"log mass {age_range_strs[aa]} (regularised)"] - df[f"log mass {age_range_strs[aa]} (input)"],
                        marker=marker, mfc="lightgreen", mec="green", ecolor="green", linestyle="none", markersize=3.5,
                        label="Regularised fit")
    axs[aa][1].errorbar(x=df.index, 
                        y=df[f"log mass {age_range_strs[aa]} (MC) mean"] - df[f"log mass {age_range_strs[aa]} (input)"],
                        yerr=df[f"log mass {age_range_strs[aa]} (MC) std. dev."],
                        marker=marker, mfc="red", mec="red", ecolor="red", linestyle="none", markersize=3.5,
                        label="MC simulations")
    # Decorations
    axs[aa][0].text(s=age_range_strs[aa], x=0.05, y=0.95, transform=axs[aa][0].transAxes, horizontalalignment="left", verticalalignment="top")
    axs[aa][0].set_ylabel(r"Total mass ($\log_{10}\rm M_\odot$)")
    axs[aa][1].set_ylabel(r"Total mass error ($\log_{10}\rm M_\odot$)")

# Decorations
for ax in axs.flat:
    # ax.set_yscale("log")
    ax.grid()
axs[-1][0].legend(fontsize="x-small")
axs[-1][0].set_xlabel("Galaxy ID")
axs[-1][1].set_xlabel("Galaxy ID")
"""

