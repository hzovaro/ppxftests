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
rcParams["font.size"] = 10
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


###############################################################################
# Plot
###############################################################################
if savefigs:
    pp = PdfPages(os.path.join(fig_path, "mw_lw_ages.pdf"))
for weighttype, marker in zip(["LW", "MW"], ["x", "D"]):

    for age in [1e8, 1e9]:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
        fig.subplots_adjust(hspace=0)
        fig.suptitle(f"{weighttype} weighted age (<{age/1e6:.2f} Myr)")

        # Axis 1: absolute values 
        axs[0].errorbar(x=df_ages.index, 
                            y=df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"],
                            marker=marker, mfc="lightblue", mec="blue", ecolor="lightblue", linestyle="none",
                            label="Input")
        axs[0].errorbar(x=df_ages.index, 
                            y=df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"],
                            marker=marker, mfc="lightgreen", mec="green", ecolor="green", linestyle="none", markersize=3.5,
                            label="Regularised fit")
        axs[0].errorbar(x=df_ages.index, 
                            y=df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"],
                            yerr=df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (MC error)"],
                            marker=marker, mfc="red", mec="red", ecolor="red", alpha=0.5, linewidth=0.5, linestyle="none", markersize=3.5,
                            label="MC simulations")

        # Axis 2: relative values 
        axs[1].axhline(0, color="black")
        axs[1].errorbar(x=df_ages.index, 
                            y=df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"],
                            marker=marker, mfc="lightgreen", mec="green", ecolor="green", linestyle="none", markersize=3.5,
                            label="Regularised fit")
        axs[1].errorbar(x=df_ages.index, 
                            y=df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"] - df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (input)"],
                            yerr=df_ages[f"{weighttype} age (<{age/1e6:.2f} Myr) (MC error)"],
                            marker=marker, mfc="red", mec="red", ecolor="red", alpha=0.5, linewidth=0.5, linestyle="none", markersize=3.5,
                            label="MC simulations")
        # Decorations
        # axs[0].text(s=age_range_strs[aa], x=0.05, y=0.95, transform=axs[0].transAxes, horizontalalignment="left", verticalalignment="top")
        axs[0].set_ylabel(f"{weighttype} weighted age (log Myr)")
        axs[0].set_ylim([6, 10.5])
        axs[1].set_ylim([-1, 1])
        axs[1].set_ylabel(f"{weighttype} weighted age error (log Myr)")
        axs[1].axhline(-0.1, ls="--", lw=0.5, color="black")
        axs[1].axhline(+0.1, ls="--", lw=0.5, color="black")

        # Decorations
        for ax in axs.flat:
            ax.grid()
            ax.autoscale(axis="x", tight=True, enable=True)
        axs[0].legend(fontsize="small")
        axs[1].set_xlabel("Galaxy ID")

        if savefigs:
            pp.savefig(fig, bbox_inches="tight")

if savefigs:
    pp.close()




