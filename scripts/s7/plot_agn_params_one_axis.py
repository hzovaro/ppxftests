import os
import numpy as np
import pandas as pd

from ppxftests.ssputils import load_ssp_templates

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

"""
For each galaxy in our sample, plot the best-fit x_AGN and A_V.
"""

s7_data_path = "/priv/meggs3/u5708159/S7/mar23/ppxf"
fig_path = "/priv/meggs3/u5708159/S7/mar23/ppxf/figs/"

###############################################################################
# Settings
###############################################################################
savefigs = False

marker_dict = {
    "MC": "d", 
    "regul": "o"
}

colour_dict = {
    "FOURAS" : "blue", 
    "ONEKPC" : "red", 
    "RE1" : "green",
}

###############################################################################
# Create the figure in which to plot ALL aperture measurements together
###############################################################################
fig_AGN, ax_AGN = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))
fig_AV, ax_AV = plt.subplots(nrows=1, ncols=1, figsize=(20, 5))

dx = -0.5
for ap in ["FOURAS", "ONEKPC", "RE1"]:
    dx += 0.25

    ###############################################################################
    # Load the DataFrame
    ###############################################################################
    # Load the DataFrame containing all S7 galaxies
    df_fname = f"s7_ppxf_{ap}.hd5"
    df_all = pd.read_hdf(os.path.join(s7_data_path, df_fname), key="s7")

    gals = df_all.index.unique()
    gals = gals.sort_values()

    # Get the age & metallicity dimensions
    _, _, metallicities, ages = load_ssp_templates("Padova")

    ###############################################################################
    # Load the DataFrame from each galaxy & store the relevant values in a single 
    # big DataFrame
    ###############################################################################
    df_ages = pd.DataFrame(index=gals)
    df_ages.index.name = "Galaxy"

    for gg, gal in enumerate(gals):
        # Open the DataFrame
        df_gal = df_all[df_all.index == gal]

        # x_AGN and A_V measurements
        df_ages.loc[gal, "x_AGN (total, regularised)"] = df_gal["x_AGN (total, regularised)"].values[0]
        df_ages.loc[gal, "x_AGN (total, MC mean)"] = df_gal["x_AGN (total, MC mean)"].values[0]
        df_ages.loc[gal, "x_AGN (total, MC error)"] = df_gal["x_AGN (total, MC error)"].values[0]
        df_ages.loc[gal, "x_AGN (total, MC 16th percentile)"] = df_gal["x_AGN (total, MC 16th percentile)"].values[0]
        df_ages.loc[gal, "x_AGN (total, MC 50th percentile)"] = df_gal["x_AGN (total, MC 50th percentile)"].values[0]
        df_ages.loc[gal, "x_AGN (total, MC 84th percentile)"] = df_gal["x_AGN (total, MC 84th percentile)"].values[0]
        df_ages.loc[gal, "A_V (regularised)"] = df_gal["A_V (regularised)"].values[0]
        df_ages.loc[gal, "A_V (MC mean)"] = df_gal["A_V (MC mean)"].values[0]
        df_ages.loc[gal, "A_V (MC error)"] = df_gal["A_V (MC error)"].values[0]
        df_ages.loc[gal, "A_V (MC 16th percentile)"] = df_gal["A_V (MC 16th percentile)"].values[0]
        df_ages.loc[gal, "A_V (MC 50th percentile)"] = df_gal["A_V (MC 50th percentile)"].values[0]
        df_ages.loc[gal, "A_V (MC 84th percentile)"] = df_gal["A_V (MC 84th percentile)"].values[0]
        df_ages.loc[gal, "Number"] = gg

    ###############################################################################
    # Plot AGN
    ###############################################################################
    ax_AGN.errorbar(x=df_ages["Number"].values + dx, 
                y=df_ages[f"x_AGN (total, regularised)"],
                marker=marker_dict["regul"], mfc=colour_dict[ap], mec=colour_dict[ap], ecolor=colour_dict[ap], linestyle="none", markersize=6,
                label="Regularised fit")
    ax_AGN.errorbar(x=df_ages["Number"].values + dx, 
                y=df_ages[f"x_AGN (total, MC mean)"],
                yerr=[
                            df_ages[f"x_AGN (total, MC 50th percentile)"] - df_ages[f"x_AGN (total, MC 16th percentile)"],
                            df_ages[f"x_AGN (total, MC 84th percentile)"] - df_ages[f"x_AGN (total, MC 50th percentile)"],
                        ],
                marker=marker_dict["MC"], mfc=colour_dict[ap], mec=colour_dict[ap], ecolor=colour_dict[ap], alpha=0.5, linewidth=1.2, linestyle="none", markersize=6,
                label="MC simulations")

    ###############################################################################
    # Plot A_V
    ###############################################################################
    ax_AV.errorbar(x=df_ages["Number"].values + dx, 
                y=df_ages[f"A_V (regularised)"],
                marker=marker_dict["regul"], mfc=colour_dict[ap], mec=colour_dict[ap], ecolor=colour_dict[ap], linestyle="none", markersize=6,
                label="Regularised fit")
    ax_AV.errorbar(x=df_ages["Number"].values + dx, 
                y=df_ages[f"A_V (MC mean)"],
                yerr=[
                            df_ages[f"A_V (MC 50th percentile)"] - df_ages[f"A_V (MC 16th percentile)"],
                            df_ages[f"A_V (MC 84th percentile)"] - df_ages[f"A_V (MC 50th percentile)"],
                        ],
                marker=marker_dict["MC"], mfc=colour_dict[ap], mec=colour_dict[ap], ecolor=colour_dict[ap], alpha=0.5, linewidth=1.2, linestyle="none", markersize=6,
                label="MC simulations")

# Decorations
ax_AV.set_ylabel(r"$A_V$ (mag)")
ax_AGN.set_ylabel(r"$x_{\rm AGN}$")
for ax in [ax_AV, ax_AGN]:
    ax.set_xticks(range(len(gals)))
    ax.set_xticks(np.array(range(len(gals))) + 0.5, minor=True)
    ax.set_xticklabels(gals, rotation="vertical", fontsize="small")
    ax.grid(b=True, which="minor", axis="x")
    ax.grid(b=True, which="major", axis="y")
    ax.autoscale(axis="x", tight=True, enable=True)
    ax.set_xlabel("Galaxy")
    ax.set_xlim([-1, len(gals)])
    # Legend
    legend_elements = [
        Patch(facecolor=colour_dict["RE1"], label=r"$1R_e$"),
        Patch(facecolor=colour_dict["ONEKPC"], label=r"1 kpc"),
        Patch(facecolor=colour_dict["FOURAS"], label='4"'),
        Line2D([0], [0], markerfacecolor="k", markeredgecolor="none", color="none", marker=marker_dict["MC"], label="MC simulations"),
        Line2D([0], [0], markerfacecolor="k", markeredgecolor="none", color="none", marker=marker_dict["regul"], label="Regularised fit"),
        Line2D([0], [0], markerfacecolor="grey", alpha=0.5, markeredgecolor="none", color="none", marker=marker_dict["MC"], label="MC simulations (unreliable)"),
        Line2D([0], [0], markerfacecolor="grey", alpha=0.5, markeredgecolor="none", color="none", marker=marker_dict["regul"], label="Regularised fit (unreliable)"),
    ]
    ax.legend(handles=legend_elements, fontsize="small", loc="center left", bbox_to_anchor=[1.02, 0.5])

if savefigs:
    fig_AGN.savefig(os.path.join(fig_path, f"x_AGN.pdf"), bbox_inches="tight", format="pdf")
    fig_AV.savefig(os.path.join(fig_path, f"A_V.pdf"), bbox_inches="tight", format="pdf")
