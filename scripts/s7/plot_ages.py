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

        # Extract the age index at the appropriate time 
        for age in ages[1:]:
            age_idx = np.nanargmin(np.abs(ages - age))
            for col in ["MC mean", "MC error", "regularised"]:
                df_ages.loc[gal, f"MW age (<{age/1e6:.2f} Myr) ({col})"] =\
                    df_gal[f"Mass-weighted age vs. age cutoff ({col})"].values.item()[age_idx - 1]
                df_ages.loc[gal, f"LW age (<{age/1e6:.2f} Myr) ({col})"] =\
                    df_gal[f"Light-weighted age vs. age cutoff ({col})"].values.item()[age_idx - 1]
                # Also compute the cumulative light fraction
                df_ages.loc[gal, f"Cumulative light fraction (<{age/1e6:.2f} Myr) ({col})"] =\
                    df_gal[f"Cumulative light fraction vs. age cutoff ({col})"].values.item()[age_idx - 1]
                df_ages.loc[gal, "Number"] = gg

    ###############################################################################
    # Plot
    ###############################################################################
    for weighttype, marker in zip(["LW", "MW"], ["x", "D"]):
        for age in [1e8, 1e9]:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
            fig.subplots_adjust(bottom=0.3)
            ax.set_title(f"{weighttype} weighted age " + r"($\tau_{\rm cutoff} = %.0f \,\rm Myr$)" % (age / 1e6))

            #//////////////////////////////////////////////////////////////////////
            # Axis 1: absolute values 
            #//////////////////////////////////////////////////////////////////////
            cond_good_regul = df_ages[f"Cumulative light fraction (<{age/1e6:.2f} Myr) (regularised)"] > 10**(-2.5)
            cond_good_MC = df_ages[f"Cumulative light fraction (<{age/1e6:.2f} Myr) (MC mean)"] > 10**(-2.5)
            
            # Reliable
            ax.errorbar(x=df_ages.loc[cond_good_regul, "Number"].values, 
                        y=df_ages.loc[cond_good_regul, f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"],
                        marker=marker, mfc="lightgreen", mec="green", ecolor="green", linestyle="none", markersize=8,
                        label="Regularised fit")
            ax.errorbar(x=df_ages.loc[cond_good_MC, "Number"].values, 
                        y=df_ages.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"],
                        yerr=df_ages.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC error)"],
                        marker=marker, mfc="red", mec="red", ecolor="red", alpha=0.5, linewidth=0.5, linestyle="none", markersize=8,
                        label="MC simulations")
            # Unreliable
            ax.errorbar(x=df_ages.loc[~cond_good_regul, "Number"].values, 
                        y=df_ages.loc[~cond_good_regul, f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"],
                        marker=marker, mfc="lightgrey", mec="grey", ecolor="grey", linestyle="none", markersize=8,
                        label="Regularised fit (unreliable)")
            ax.errorbar(x=df_ages.loc[~cond_good_MC, "Number"].values, 
                        y=df_ages.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"],
                        yerr=df_ages.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC error)"],
                        marker=marker, mfc="lightgrey", mec="grey", ecolor="grey", alpha=0.5, linewidth=0.5, linestyle="none", markersize=8,
                        label="MC simulations (unreliable)")

            # Decorations
            ax.set_xticks(range(len(gals)))
            ax.set_xticklabels(gals, rotation="vertical", fontsize="small")
            ax.set_ylabel(f"{weighttype} weighted age (log Myr)")
            ax.set_ylim([6, 10.5])
            ax.grid()
            ax.autoscale(axis="x", tight=True, enable=True)
            ax.legend(fontsize="small")
            ax.set_xlabel("Galaxy")
            ax.set_xlim([-1, len(gals)])

            if savefigs:
                fig.savefig(os.path.join(fig_path, f"{weighttype}_ages_tau={age:.0e}_{ap}.pdf"), bbox_inches="tight", format="pdf")


