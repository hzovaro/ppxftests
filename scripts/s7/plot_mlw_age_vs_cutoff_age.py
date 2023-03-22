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
For each galaxy in our sample, plot the MW/LW ages as a function of cutoff age.
"""

s7_data_path = "/priv/meggs3/u5708159/S7/ppxf"
fig_path = "/priv/meggs3/u5708159/S7/ppxf/figs/"

###############################################################################
# Settings
###############################################################################
savefigs = True

###############################################################################
# Load the DataFrame
###############################################################################
# Load the DataFrame containing all S7 galaxies
df_fname = f"s7_ppxf.hd5"
df_all = pd.read_hdf(os.path.join(s7_data_path, df_fname), key="s7")
gals = df_all["Galaxy"].unique()

# Get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates("Padova")

###############################################################################
# Summary plot
###############################################################################
for gal in tqdm(gals):
    # Extract the row in the DataFrame corresponding to this galaxy
    df_gal = df_all[df_all["Galaxy"] == gal]

    #//////////////////////////////////////////////////////////////////////////////
    # Plot the mean mass- and light-weighted age vs. age threshold
    # expressed as an ERROR
    #//////////////////////////////////////////////////////////////////////////////
    # weighttypes = ["Mass-weighted age", "Light-weighted age", "Cumulative mass", "Cumulative light"]
    weighttypes = ["Mass-weighted age", "Light-weighted age"]
    fig, axs = plt.subplots(nrows=1, ncols=len(weighttypes), figsize=(5 * len(weighttypes), 4))
    # fig.subplots_adjust(left=0.025, right=1 - 0.025, top=1 - 0.05, bottom=0.025)
    fig.suptitle(gal)

    colour_lists = [
        ("red", "maroon", "pink"),
        ("green", "darkgreen", "lightgreen"),
        ("blue", "indigo", "cornflowerblue"),
        ("gold", "brown", "orange")
    ]

    # PLOT: 
    for cc, weighttype in enumerate(weighttypes):
        colname = f"{weighttype} vs. age cutoff"
        colour_input, colour_regul, colour_mc = colour_lists[cc]

        #//////////////////////////////////////////////////////////////////////
        # Plot the measured values vs. age
        axs[cc].step(x=ages[1:], y=df_gal[f"{colname} (regularised)"].values.item(), label=f"{weighttype} (regularised)", where="mid", color=colour_regul)
        y_meas = df_gal[f"{colname} (MC mean)"].values.item()
        y_err = df_gal[f"{colname} (MC error)"].values.item()
        axs[cc].step(x=ages[1:], y=y_meas, where="mid", color=colour_mc, label=f"{weighttype} (MC)")
        axs[cc].fill_between(x=ages[1:], y1=y_meas - y_err, y2=y_meas + y_err, step="mid", alpha=0.2, color=colour_mc)

        # Add shaded region to indicate where the age estimate becomes unreliable
        unreliable_idxs_regul = np.log10(df_gal["Cumulative light fraction vs. age cutoff (regularised)"].item()) < -2.5    
        try:
            first_reliable_age_regul = ages[np.argwhere(~unreliable_idxs_regul)[0]][0]
            first_reliable_age_regul_plot = 10**(np.log10(first_reliable_age_regul) - 0.025)
            axs[cc].axvspan(xmin=ages[0], xmax=first_reliable_age_regul_plot, alpha=0.1, color="grey")
            axs[cc].axvline(first_reliable_age_regul_plot, lw=1, ls="--", color="k", label="Unreliable range (regularised fit)")
        except IndexError as e:
            print(f"ERROR: unable to plot unreliable range for {gal} (regularised fit). Skipping...")
        
        unreliable_idxs_mc = np.log10(df_gal["Cumulative light fraction vs. age cutoff (MC mean)"].item()) < -2.5    
        try:
            first_reliable_age_mc = ages[np.argwhere(~unreliable_idxs_mc)[0]][0]
            first_reliable_age_mc_plot = 10**(np.log10(first_reliable_age_mc) - 0.025)
            axs[cc].axvspan(xmin=ages[0], xmax=first_reliable_age_mc_plot, alpha=0.1, color="grey")
            axs[cc].axvline(first_reliable_age_mc_plot, lw=1, ls=":", color="k", label="Unreliable range (MC fit)")
        except IndexError as e:
            print(f"ERROR: unable to plot unreliable range for {gal} (MC fit). Skipping...")

        # Decorations  
        # axs[cc].axvspan(ages[1], ages[np.argwhere(np.isfinite(df_gal[f"{colname} (input)"].values.item()))[0][0]], color="grey", alpha=0.2)  
        axs[cc].set_xscale("log")
        axs[cc].legend(loc="upper left", fontsize="small")
        axs[cc].set_xlabel("Age threshold (yr)")
        axs[cc].set_ylabel(f"{weighttype} (log yr)" if weighttype.endswith("age") else f"{weighttype} " + r"($M_\odot$)")
        axs[cc].set_xlim([ages[0], ages[-1]])
        axs[cc].grid()

    if savefigs:
        fig.savefig(os.path.join(fig_path, f"{gal}_mlw_age_vs_cutoff_age.pdf"), bbox_inches="tight", format="pdf")   

    plt.close("all")
