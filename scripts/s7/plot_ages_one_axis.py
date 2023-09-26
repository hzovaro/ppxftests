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
savefigs = True
age_cutoff_strs = ["1e8", "1e9"]
weighttypes = ["LW", "MW"]

marker_dict = {
    "MC": "d", 
    "regul": "o"
}

colour_dict = {
    "FOURAS" : "blue", 
    "ONEKPC" : "red", 
    "RE1" : "green",
}

"""
On each axis: plot ALL apertures for ONE cutoff age and ONE weight type, w/ MC AND regul fits shown
Differentt symbol for MC and regul; same colour for each aperture type
"""

###############################################################################
# Create the figure in which to plot ALL aperture measurements together
###############################################################################
fig_dict = {
    "LW": {
        "1e8": plt.subplots(nrows=1, ncols=1, figsize=(20, 5)),
        "1e9": plt.subplots(nrows=1, ncols=1, figsize=(20, 5)),
    },
    "MW": {
        "1e8": plt.subplots(nrows=1, ncols=1, figsize=(20, 5)),
        "1e9": plt.subplots(nrows=1, ncols=1, figsize=(20, 5)),
    },
}

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

        # Extract the age index at the appropriate time 
        for age in ages[1:]:
            age_idx = np.nanargmin(np.abs(ages - age))
            for col in ["MC mean", "MC error", "MC 50th percentile", "MC 16th percentile", "MC 84th percentile", "regularised"]:
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
    for weighttype in weighttypes:
        for age_str in age_cutoff_strs:
            age = float(age_str)
            
            #//////////////////////////////////////////////////////////////////////
            # Axis 1: absolute values 
            #//////////////////////////////////////////////////////////////////////
            cond_good_regul = df_ages[f"Cumulative light fraction (<{age/1e6:.2f} Myr) (regularised)"] > 10**(-2.5)
            cond_good_MC = df_ages[f"Cumulative light fraction (<{age/1e6:.2f} Myr) (MC mean)"] > 10**(-2.5)
            
            #//////////////////////////////////////////////////////////////////////
            # Plot on the shared axes
            #//////////////////////////////////////////////////////////////////////
            # Reliable
            fig_all, ax_all = fig_dict[weighttype][age_str]
            ax_all.errorbar(x=df_ages.loc[cond_good_regul, "Number"].values + dx, 
                        y=df_ages.loc[cond_good_regul, f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"],
                        marker=marker_dict["regul"], mfc=colour_dict[ap], mec=colour_dict[ap], ecolor=colour_dict[ap], linestyle="none", markersize=5,
                        label="Regularised fit")
            ax_all.errorbar(x=df_ages.loc[cond_good_MC, "Number"].values + dx, 
                        y=df_ages.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"],
                        yerr=[
                            df_ages.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 50th percentile)"] - df_ages.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 16th percentile)"],
                            df_ages.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 84th percentile)"] - df_ages.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 50th percentile)"],
                        ],
                        marker=marker_dict["MC"], mfc=colour_dict[ap], mec=colour_dict[ap], ecolor=colour_dict[ap], alpha=0.5, linewidth=0.5, linestyle="none", markersize=5,
                        label="MC simulations")
            # Unreliable
            ax_all.errorbar(x=df_ages.loc[~cond_good_regul, "Number"].values + dx, 
                        y=df_ages.loc[~cond_good_regul, f"{weighttype} age (<{age/1e6:.2f} Myr) (regularised)"],
                        marker=marker_dict["regul"], mfc="lightgrey", mec="grey", ecolor="grey", linestyle="none", markersize=5,
                        label="Regularised fit (unreliable)")
            ax_all.errorbar(x=df_ages.loc[~cond_good_MC, "Number"].values + dx, 
                        y=df_ages.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"],
                        yerr=[
                            df_ages.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 50th percentile)"] - df_ages.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 16th percentile)"],
                            df_ages.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 84th percentile)"] - df_ages.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 50th percentile)"],
                        ],
                        marker=marker_dict["MC"], mfc="lightgrey", mec="grey", ecolor="grey", alpha=0.5, linewidth=0.5, linestyle="none", markersize=5,
                        label="MC simulations (unreliable)")
            
    ###############################################################################
    # Save the "ages" DataFrame to file
    ###############################################################################
    # Only save a select number of age values to file
    cols = [c for c in df_ages if ("1000.0" in c or "100.0" in c) and "age" in c]
    cols += [c for c in df_ages if "17783.0" in c and "age" in c]
    df_ages_save = df_ages[cols]
    df_ages_save = df_ages_save.rename(columns=dict(zip(cols, [c.replace(" (<17783.00 Myr)", "") + " (log yr)" for c in cols])))
    df_ages_save.to_csv(os.path.join(s7_data_path, f"s7_ppxf_{ap}_ages.csv"))

# Decorations
for weighttype in weighttypes:
    for age_str in age_cutoff_strs:
        fig_all, ax_all = fig_dict[weighttype][age_str]
        age = float(age_str)
        ax_all.set_title(f"{'Light' if weighttype == 'LW' else 'Mass'}-weighted age " + r"($\tau_{\rm cutoff} = %.0f \,\rm Myr$)" % (age / 1e6))
        ax_all.set_xticks(range(len(gals)))
        ax_all.set_xticks(np.array(range(len(gals))) + 0.5, minor=True)
        ax_all.set_xticklabels(gals, rotation="vertical", fontsize="small")
        ax_all.grid(b=True, which="minor", axis="x")
        ax_all.grid(b=True, which="major", axis="y")
        ax_all.set_ylabel(f"{'Light' if weighttype == 'LW' else 'Mass'}-weighted age (log Myr)")
        ax_all.set_ylim([6, 10.5])
        ax_all.autoscale(axis="x", tight=True, enable=True)
        ax_all.set_xlabel("Galaxy")
        ax_all.set_xlim([-1, len(gals)])
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
        ax_all.legend(handles=legend_elements, fontsize="small", loc="center left", bbox_to_anchor=[1.02, 0.5])

        if savefigs:
            fig_all.savefig(os.path.join(fig_path, f"{weighttype}_ages_tau={age:.0e}.pdf"), bbox_inches="tight", format="pdf")
