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

if len(sys.argv) > 1:
    gals = sys.argv[1:]
else:
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
    # Plot the SFHs (both mass- and light-weighted) from the MC and regularised fits
    #//////////////////////////////////////////////////////////////////////////////
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    for weighttype, ax in zip(["Mass", "Light"], axs):
        ax.step(x=ages, y=df_gal[f"SFH {weighttype[0]}W 1D (regularised)"].values.item(), where="mid", color="purple", label="SFH (regularised)", linewidth=1.5)
        ax.step(x=ages, y=df_gal[f"SFH {weighttype[0]}W 1D (MC mean)"].values.item(), where="mid", color="cornflowerblue", label="SFH (MC)", linewidth=1.0)
        ax.errorbar(x=ages, 
                    y=df_gal[f"SFH {weighttype[0]}W 1D (MC mean)"].values.item(),
                    yerr=df_gal[f"SFH {weighttype[0]}W 1D (MC error)"].values.item(),
                    linestyle="none", color="cornflowerblue")
        # ax.axhline(1e-4, color="k", ls="--", linewidth=1)

        # Add shaded region to indicate where the age estimate becomes unreliable
        unreliable_idxs_regul = np.log10(df_gal["Cumulative light fraction vs. age cutoff (regularised)"].item()) < -2.5    
        try:
            first_reliable_age_regul = ages[np.argwhere(~unreliable_idxs_regul)[0]][0]
            first_reliable_age_regul_plot = 10**(np.log10(first_reliable_age_regul) - 0.025)
            ax.axvspan(xmin=ages[0], xmax=first_reliable_age_regul_plot, alpha=0.1, color="grey")
            ax.axvline(first_reliable_age_regul_plot, lw=1, ls="--", color="k", label="Unreliable range (regularised fit)")
        except IndexError as e:
            print(f"ERROR: unable to plot unreliable range for {gal} (regularised fit). Skipping...")
        
        unreliable_idxs_mc = np.log10(df_gal["Cumulative light fraction vs. age cutoff (MC mean)"].item()) < -2.5    
        try:
            first_reliable_age_mc = ages[np.argwhere(~unreliable_idxs_mc)[0]][0]
            first_reliable_age_mc_plot = 10**(np.log10(first_reliable_age_mc) - 0.025)
            ax.axvspan(xmin=ages[0], xmax=first_reliable_age_mc_plot, alpha=0.1, color="grey")
            ax.axvline(first_reliable_age_mc_plot, lw=1, ls=":", color="k", label="Unreliable range (MC fit)")
        except IndexError as e:
            print(f"ERROR: unable to plot unreliable range for {gal} (MC fit). Skipping...")

        # Decorations
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid()
        if weighttype == "Mass":
            ax.set_ylabel(r"Template mass ($\rm M_\odot$)")
        else:
            ax.set_ylabel(r"Template luminosity at 4020$\,\rm \AA$ ($\rm erg\,s^{-1}$)")
        ax.set_xlabel("Age (yr)")
        ax.autoscale(axis="x", tight=True, enable=True)
        axs[0].legend(loc="upper left", fontsize="small")
        axs[0].set_title(gal)

        if savefigs:
            fig.savefig(os.path.join(fig_path, f"{gal}_sfh.pdf"), bbox_inches="tight", format="pdf")   

    # plt.close("all")