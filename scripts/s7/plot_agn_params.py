import os, sys 
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

from ppxftests.ssputils import load_ssp_templates

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
savefigs = True
aps = [ap.upper() for ap in sys.argv[1:]]
for ap in aps:
    assert ap in ["FOURAS", "ONEKPC", "RE1"], "ap must be one of 'FOURAS', 'ONEKPC', 'RE1'!"

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
        df_ages.loc[gal, "x_AGN (total, regularised)"] = df_gal["x_AGN (total, regularised)"].values[0]
        df_ages.loc[gal, "x_AGN (total, MC mean)"] = df_gal["x_AGN (total, MC mean)"].values[0]
        df_ages.loc[gal, "x_AGN (total, MC error)"] = df_gal["x_AGN (total, MC error)"].values[0]

        df_ages.loc[gal, "A_V (regularised)"] = df_gal["A_V (regularised)"].values[0]
        df_ages.loc[gal, "A_V (MC mean)"] = df_gal["A_V (MC mean)"].values[0]
        df_ages.loc[gal, "A_V (MC error)"] = df_gal["A_V (MC error)"].values[0]
        df_ages.loc[gal, "Number"] = gg

    ###############################################################################
    # Plot x_AGN
    ###############################################################################
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    fig.subplots_adjust(bottom=0.3)
    ax.set_title(r"AGN continuum strength")
    ax.errorbar(x=df_ages["Number"].values, 
                y=df_ages[f"x_AGN (total, regularised)"],
                marker="o", mfc="lightgreen", mec="green", ecolor="green", linestyle="none", markersize=8,
                label="Regularised fit")
    ax.errorbar(x=df_ages["Number"].values, 
                y=df_ages[f"x_AGN (total, MC mean)"],
                yerr=df_ages[f"x_AGN (total, MC error)"],
                marker="o", mfc="red", mec="red", ecolor="red", alpha=0.5, linewidth=1.2, linestyle="none", markersize=8,
                label="MC simulations")

    # Decorations
    ax.set_xticks(range(len(gals)))
    ax.set_xticklabels(gals, rotation="vertical", fontsize="small")
    ax.set_ylabel(r"$x_{\rm AGN}$")
    ax.grid()
    ax.autoscale(axis="x", tight=True, enable=True)
    ax.legend(fontsize="small")
    ax.set_xlabel("Galaxy")
    ax.set_xlim([-1, len(gals)])
    ax.set_ylim([-0.25, 4])
    ax.axhline(0, color="k", ls="--")

    if savefigs:
        fig.savefig(os.path.join(fig_path, f"x_AGN_{ap}.pdf"), bbox_inches="tight", format="pdf")

    ###############################################################################
    # Plot A_V
    ###############################################################################
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    fig.subplots_adjust(bottom=0.3)
    ax.set_title(r"Extinction")
    ax.errorbar(x=df_ages["Number"].values, 
                y=df_ages[f"A_V (regularised)"],
                marker="o", mfc="lightgreen", mec="green", ecolor="green", linestyle="none", markersize=8,
                label="Regularised fit")
    ax.errorbar(x=df_ages["Number"].values, 
                y=df_ages[f"A_V (MC mean)"],
                yerr=df_ages[f"A_V (MC error)"],
                marker="o", mfc="red", mec="red", ecolor="red", alpha=0.5, linewidth=1.2, linestyle="none", markersize=8,
                label="MC simulations")

    # Decorations
    ax.set_xticks(range(len(gals)))
    ax.set_xticklabels(gals, rotation="vertical", fontsize="small")
    ax.set_ylabel(r"$A_V$ (mag)")
    ax.grid()
    ax.autoscale(axis="x", tight=True, enable=True)
    ax.legend(fontsize="small")
    ax.set_xlabel("Galaxy")
    ax.set_xlim([-1, len(gals)])
    ax.set_ylim([-0.25, 5])
    ax.axhline(0, color="k", ls="--")

    if savefigs:
        fig.savefig(os.path.join(fig_path, f"A_V_{ap}.pdf"), bbox_inches="tight", format="pdf")


