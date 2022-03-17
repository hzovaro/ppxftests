"""
For each galaxy, plot the measured/input LW/MW age as a function of galaxy
number
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from ppxftests.ssputils import load_ssp_templates

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
rcParams["font.size"] = 14
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

data_path = "/priv/meggs3/u5708159/ppxftests/best_case/"
fig_path = "/priv/meggs3/u5708159/ppxftests/figs/best_case/"

debug = False
savefigs = True

###############################################################################
# Compute the mass- and light-weighted mean age for each galaxy,
# and store in the DataFrame
###############################################################################
# Load the stellar templates so we can get the age & metallicity dimensions
isochrones = "Padova"
_, _, metallicities, ages = load_ssp_templates(isochrones)
N_ages = len(ages)
N_metallicities = len(metallicities)

###############################################################################
# Load the DataFrame from each galaxy & store the relevant values in a single 
# big DataFrame
###############################################################################
if debug:
    df_fnames = [f for f in os.listdir(data_path) if f.endswith("_DEBUG.hd5")]
    gals = [int(f.split("_DEBUG.hd5")[0].split("ga")[1]) for f in df_fnames]
else:
    df_fnames = [f for f in os.listdir(data_path) if f.endswith("_bestcase.hd5")]
    gals = [int(f.split("_bestcase.hd5")[0].split("ga")[1]) for f in df_fnames]

df_ages = pd.DataFrame(index=gals)
df_ages.index.name = "Galaxy"

for gal, df_fname in tqdm(zip(gals, df_fnames), total=len(gals)):
    # Open the DataFrame
    df_gal = pd.read_hdf(os.path.join(data_path, df_fname))

    # Extract the age index at the appropriate time 
    for age in ages[1:]:
        age_idx = np.nanargmin(np.abs(ages - age))
        for col in ["input", "MC mean", "MC error", "regularised"]:
            df_ages.loc[gal, f"MW age (<{age/1e6:.2f} Myr) ({col})"] =\
                df_gal[f"Mass-weighted age vs. age cutoff ({col})"].values.item()[0][age_idx - 1]
            df_ages.loc[gal, f"LW age (<{age/1e6:.2f} Myr) ({col})"] =\
                df_gal[f"Light-weighted age vs. age cutoff ({col})"].values.item()[0][age_idx - 1]
            # Also compute the cumulative light fraction
            df_ages.loc[gal, f"Cumulative light (<{age/1e6:.2f} Myr) ({col})"] =\
                df_gal[f"Cumulative light vs. age cutoff ({col})"].values.item()[0][age_idx - 1] - df_gal[f"Cumulative light vs. age cutoff ({col})"].values.item()[0][-1]

###############################################################################
# Plot
###############################################################################
for weighttype, marker in zip(["LW", "MW"], ["x", "D"]):
    for age in [1e8, 1e9]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5))
        ax.set_title(f"{weighttype} weighted age " + r"($\tau_{\rm cutoff} = %.0f \,\rm Myr$)" % (age / 1e6))

        #//////////////////////////////////////////////////////////////////////
        # Axis 1: absolute values 
        #//////////////////////////////////////////////////////////////////////
        cond_good_input = df_ages[f"Cumulative light (<{age/1e6:.2f} Myr) (input)"] > -2.5
        cond_good_regul = df_ages[f"Cumulative light (<{age/1e6:.2f} Myr) (regularised)"] > -2.5
        cond_good_MC = df_ages[f"Cumulative light (<{age/1e6:.2f} Myr) (MC mean)"] > -2.5
        
        # Reliable
        ax.errorbar(x=df_ages.loc[cond_good_input].index, 
                            y=df_ages.loc[cond_good_input, f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"],
                            marker=marker, mfc="lightblue", mec="blue", ecolor="lightblue", linestyle="none",
                            label="Input")
        ax.errorbar(x=df_ages.loc[cond_good_regul].index, 
                            y=df_ages.loc[cond_good_regul, f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"],
                            marker=marker, mfc="lightgreen", mec="green", ecolor="green", linestyle="none", markersize=3.5,
                            label="Regularised fit")
        ax.errorbar(x=df_ages.loc[cond_good_MC].index, 
                            y=df_ages.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"],
                            yerr=df_ages.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC error)"],
                            marker=marker, mfc="red", mec="red", ecolor="red", alpha=0.5, linewidth=0.5, linestyle="none", markersize=3.5,
                            label="MC simulations")
        # Unreliable
        ax.errorbar(x=df_ages.loc[~cond_good_input].index, 
                            y=df_ages.loc[~cond_good_input, f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"],
                            marker=marker, mfc="lightgrey", mec="grey", ecolor="grey", linestyle="none",
                            label=r"Input (cumulative light fraction $< -2.5$)")
        ax.errorbar(x=df_ages.loc[~cond_good_regul].index, 
                            y=df_ages.loc[~cond_good_regul, f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"],
                            marker=marker, mfc="lightgrey", mec="grey", ecolor="grey", linestyle="none", markersize=3.5,
                            label="Regularised fit (unreliable)")
        ax.errorbar(x=df_ages.loc[~cond_good_MC].index, 
                            y=df_ages.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"],
                            yerr=df_ages.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC error)"],
                            marker=marker, mfc="lightgrey", mec="grey", ecolor="grey", alpha=0.5, linewidth=0.5, linestyle="none", markersize=3.5,
                            label="MC simulations (unreliable)")

        # Decorations
        ax.set_ylabel(f"{weighttype} weighted age (log Myr)")
        ax.set_ylim([6, 10.5])
        ax.grid()
        ax.autoscale(axis="x", tight=True, enable=True)
        ax.legend(fontsize="small")
        ax.set_xlabel("Galaxy ID")

        if savefigs:
            fig.savefig(os.path.join(fig_path, f"{weighttype}_ages_bestcase_tau={age:.0e}.pdf"), bbox_inches="tight", format="pdf")

        #//////////////////////////////////////////////////////////////////////
        # Axis 2: relative values 
        #//////////////////////////////////////////////////////////////////////
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5))
        ax.set_title(f"{weighttype} weighted age error " + r"($\tau_{\rm cutoff} = %.0f \,\rm Myr$)" % (age / 1e6))

        ax.axhline(0, color="black")
        ax.errorbar(x=df_ages.loc[cond_good_regul].index, 
                            y=df_ages.loc[cond_good_regul, f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"] - df_ages.loc[cond_good_regul, f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"],
                            marker=marker, mfc="lightgreen", mec="green", ecolor="green", linestyle="none", markersize=3.5,
                            label="Regularised fit")
        ax.errorbar(x=df_ages.loc[cond_good_MC].index, 
                            y=df_ages.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"] - df_ages.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"],
                            yerr=df_ages.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC error)"],
                            marker=marker, mfc="red", mec="red", ecolor="red", alpha=0.5, linewidth=0.5, linestyle="none", markersize=3.5,
                            label="MC simulations")

        ax.errorbar(x=df_ages.loc[~cond_good_regul].index, 
                            y=df_ages.loc[~cond_good_regul, f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"] - df_ages.loc[~cond_good_regul, f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"],
                            marker=marker, mfc="grey", mec="grey", ecolor="grey", linestyle="none", markersize=3.5,
                            label="Regularised fit (unreliable)")
        ax.errorbar(x=df_ages.loc[~cond_good_MC].index, 
                            y=df_ages.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"] - df_ages.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"],
                            yerr=df_ages.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC error)"],
                            marker=marker, mfc="grey", mec="grey", ecolor="grey", alpha=0.5, linewidth=0.5, linestyle="none", markersize=3.5,
                            label="MC simulations (unreliable)")

        # Decorations
        # ax.text(s=age_range_strs[aa], x=0.05, y=0.95, transform=ax.transAxes, horizontalalignment="left", verticalalignment="top")
        ax.set_ylim([-1, 1])
        ax.set_ylabel(f"{weighttype} weighted age error (log Myr)")
        ax.axhline(-0.1, ls="--", lw=0.5, color="black")
        ax.axhline(+0.1, ls="--", lw=0.5, color="black")
        ax.grid()
        ax.autoscale(axis="x", tight=True, enable=True)
        ax.legend(fontsize="small")
        ax.set_xlabel("Galaxy ID")

        if savefigs:
            fig.savefig(os.path.join(fig_path, f"{weighttype}_ages_error_bestcase_tau={age:.0e}.pdf"), bbox_inches="tight", format="pdf")

        #//////////////////////////////////////////////////////////////////////
        # Histograms
        #//////////////////////////////////////////////////////////////////////
        # Absolute values
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5))
        xmin = np.nanmin(df_ages[[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)",
                                  f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)",
                                  f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"]].values)
        xmax = np.nanmax(df_ages[[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)",
                                  f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)",
                                  f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"]].values)
        ax.hist(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"], 
                range=(xmin, xmax), bins=20, histtype="stepfilled", color="lightblue", edgecolor="blue", alpha=0.4,
                label="Input")
        ax.hist(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"], 
                range=(xmin, xmax), bins=20, histtype="stepfilled", color="green", edgecolor="green", alpha=0.4,
                label="Regularised fit")
        ax.hist(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"], 
                range=(xmin, xmax), bins=20, histtype="stepfilled", color="red", edgecolor="red", alpha=0.4,
                label="MC simulations")
        ax.grid()
        ax.legend(fontsize="small")
        ax.set_ylabel(r"$N$")
        ax.set_xlabel(f"{weighttype} weighted age (log Myr)")
        ax.set_title(f"{weighttype} weighted age " + r"($\tau_{\rm cutoff} = %.0f \,\rm Myr$)" % (age / 1e6))

        if savefigs:
            fig.savefig(os.path.join(fig_path, f"{weighttype}_ages_bestcase_hist_tau={age:.0e}.pdf"), bbox_inches="tight", format="pdf")

        # Relative values
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5))
        xmin_tmp = np.nanmin([
                    np.nanmin(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"]),
                    np.nanmin(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"])
                    ])
        xmax_tmp = np.nanmax([
                    np.nanmax(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"]),
                    np.nanmax(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"])
                    ])
        xmax = max(np.abs(xmin_tmp), np.abs(xmax_tmp))
        xmin = -xmax
        ax.hist(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"], 
                range=(xmin, xmax), bins=20, histtype="stepfilled", color="green", edgecolor="green", alpha=0.4,
                label="Regularised fit")
        ax.hist(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"], 
                range=(xmin, xmax), bins=20, histtype="stepfilled", color="red", edgecolor="red", alpha=0.4,
                label="MC simulations")
        ax.grid()
        ax.legend(fontsize="small")
        ax.axvline(0, color="k")
        ax.axvline(-0.1, ls="--", lw=0.5, color="black")
        ax.axvline(+0.1, ls="--", lw=0.5, color="black")
        ax.set_ylabel(r"$N$")
        ax.set_xlabel(f"{weighttype} weighted age error (log Myr)")
        ax.set_title(f"{weighttype} weighted age error " + r"($\tau_{\rm cutoff} = %.0f \,\rm Myr$)" % (age / 1e6))

        if savefigs:
            fig.savefig(os.path.join(fig_path, f"{weighttype}_ages_bestcase_hist_err_tau={age:.0e}.pdf"), bbox_inches="tight", format="pdf")


for weighttype, marker in zip(["LW", "MW"], ["x", "D"]):
    for age in [1e8, 1e9]:

        # Print statistics
        err_mean_abs_regul = np.nanmean(np.abs(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"]))
        err_mean_regul = np.nanmean(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"])
        err_std_regul = np.nanstd(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"])
        err_mean_abs_MC = np.nanmean(np.abs(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"]))
        err_mean_MC = np.nanmean(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"])
        err_std_MC = np.nanstd(df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"])

        print("--------------------------------------------------------------")
        print(f"Weight type = {weighttype}, t_cutoff = {age/1e6:.2f} Myr:")
        print(f"Regularised fit:")
        print(f"Mean error = {err_mean_regul:.4f} dex ")
        print(f"Mean absolute error = {err_mean_abs_regul:.4f} dex ")
        print(f"Std. dev. of errors = {err_std_regul:.4f} dex ")
        print(f"MC fit:")
        print(f"Mean error = {err_mean_MC:.4f} dex ")
        print(f"Mean absolute error = {err_mean_abs_MC:.4f} dex ")
        print(f"Std. dev. of errors = {err_std_MC:.4f} dex ")
        print("--------------------------------------------------------------")

