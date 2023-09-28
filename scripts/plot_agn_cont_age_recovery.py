import os, sys 
import numpy as np
import pandas as pd
from itertools import product

from ppxftests.ssputils import load_ssp_templates

import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

# from IPython.core.debugger import Tracer

"""
How well do we recover the AGN continuum/A_V parameters when an AGN 
continuum is present?
"""

data_path = "/priv/meggs3/u5708159/ppxftests/ext_and_agn/"
fig_path = "/priv/meggs3/u5708159/ppxftests/figs/ext_and_agn/"

###############################################################################
# Settings
###############################################################################
debug = False
savefigs = True

isochrones = "Padova"

###############################################################################
# Load the DataFrame
###############################################################################
gals = [int(g) for g in sys.argv[1:]]
for gal in gals: 

    # True == plot the results from the fits w/ the 4th-order polynomial; 
    # False == plot the results from the fit w/ the extinction curve 
    mpoly = False  

    if not debug:
        df_fname = f"ga{gal:004d}_{'mpoly' if mpoly else 'ext'}.hd5"
    else:
        df_fname = f"ga{gal:004d}_DEBUG.hd5"
    df = pd.read_hdf(os.path.join(data_path, df_fname), key="ext_and_agn")

    # Drop rows with alpha_nu = 0.3
    cond = df["alpha_nu (input)"] == 0.3 
    df = df[~cond]

    x_AGN_vals = df["x_AGN (input)"].unique()
    alpha_nu_vals = df["alpha_nu (input)"].unique()
    A_V_vals = df["A_V (input)"].unique()
    ppxf_alpha_nu_vals = df.iloc[0]["ppxf alpha_nu_vals"][0]

    #//////////////////////////////////////////////////////////////////////////////
    # Accuracy of age measurements 
    # Stick with a 1 Gyr cutoff for now 
    #//////////////////////////////////////////////////////////////////////////////
    # Get the age & metallicity dimensions
    _, _, metallicities, ages = load_ssp_templates(isochrones)

    age_thresh = 1e8
    age_thresh_idx = np.nanargmin(np.abs(ages - age_thresh))

    # For each combination of alpha_nu, x_AGN, make this plot
    df_ages = pd.DataFrame()
    for A_V, alpha_nu, x_AGN in product(A_V_vals, alpha_nu_vals, x_AGN_vals):
        # IF both are NaN, then do NOT add an AGN continuum
        if np.isnan(x_AGN) and np.isnan(alpha_nu):
            cond = (df["A_V (input)"] == A_V) & np.isnan(df["x_AGN (input)"]) & np.isnan(df["alpha_nu (input)"])
        elif np.isnan(x_AGN) or np.isnan(alpha_nu):
            continue
        else:
            cond = (df["A_V (input)"] == A_V) & (df["x_AGN (input)"] == x_AGN) & (df["alpha_nu (input)"] == alpha_nu)
        
        if not np.any(cond):
            print("Row missing from DataFrame! Skipping...")
            continue

        thisrow = {}
        thisrow["A_V"] = A_V
        thisrow["x_AGN"] = x_AGN
        thisrow["alpha_nu"] = alpha_nu
        for weighttype in ["Mass-weighted age", "Light-weighted age", "Cumulative mass", "Cumulative light"]:
            thisrow[f"{weighttype} vs. age cutoff (input)"] = df.loc[cond, f"{weighttype} vs. age cutoff (input)"].values.item()[age_thresh_idx]
            for meastype in ["regularised", "MC mean", "MC error"]:
                thisrow[f"{weighttype} vs. age cutoff ({meastype})"] = df.loc[cond, f"{weighttype} vs. age cutoff ({meastype})"].values.item()[0][age_thresh_idx]

        # Also compute total light fraction < this age 
        if weighttype.startswith("Cumulative"):
            thisrow[f"{weighttype} fraction (input)"] = df.loc[cond, f"{weighttype} vs. age cutoff (input)"].values.item()[age_thresh_idx] - df.loc[cond, f"{weighttype} vs. age cutoff (input)"].values.item()[-1]
            thisrow[f"{weighttype} fraction (regularised)"] = df.loc[cond, f"{weighttype} vs. age cutoff (regularised)"].values.item()[0][age_thresh_idx] - df.loc[cond, f"{weighttype} vs. age cutoff (regularised)"].values.item()[0][-1]
            thisrow[f"{weighttype} fraction (MC mean)"] = df.loc[cond, f"{weighttype} vs. age cutoff (MC mean)"].values.item()[0][age_thresh_idx] - df.loc[cond, f"{weighttype} vs. age cutoff (MC mean)"].values.item()[0][-1]
            thisrow[f"{weighttype} fraction < -2.5? (regularised)"] = thisrow[f"{weighttype} fraction (regularised)"] < -2.5
            thisrow[f"{weighttype} fraction < -2.5? (MC mean)"] = thisrow[f"{weighttype} fraction (MC mean)"] < -2.5

        df_ages = df_ages.append(thisrow, ignore_index=True)

    # Plot
    colour_lists = [
        ("red", "maroon", "pink"),
        ("green", "darkgreen", "lightgreen"),
        ("blue", "indigo", "cornflowerblue"),
        ("gold", "brown", "orange")
    ]
    cmap_alpha_nu = plt.cm.get_cmap("cool", len(alpha_nu_vals) - 1)

    #//////////////////////////////////////////////////////////////////////////////
    # Plot age estimates, etc. as a function of x_AGN for varying A_V, alpha_nu
    #//////////////////////////////////////////////////////////////////////////////
    for A_V in A_V_vals:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        fig.suptitle(f"ga{gal:004}" + r" ($A_V = %.1f, \tau_{\rm cutoff} = 10^{%.1f}\,\rm yr$)" % (A_V, np.log10(age_thresh)))
        for cc, weighttype in enumerate(["Mass-weighted age", "Light-weighted age"]):
            # colour_input, colour_regul, colour_mc = colour_lists[cc]
            colour_input, colour_regul, colour_mc = ("black", "grey", "black")
            colname = f"{weighttype} vs. age cutoff"
            if weighttype.endswith("age"):
                units = r"$\log \,\rm yr$"
            elif weighttype.endswith("mass"):
                units = r"$\log \,\rm M_\odot$"
            elif weighttype.endswith("light"):
                units = r"$\log \,\rm erg\, s^{-1}$"

            cond_noagn = (df_ages["A_V"] == A_V) & np.isnan(df_ages["x_AGN"]) & np.isnan(df_ages["alpha_nu"])
            cond_agn = (df_ages["A_V"] == A_V) & ~np.isnan(df_ages["x_AGN"]) & ~np.isnan(df_ages["alpha_nu"])

            #//////////////////////////////////////////////////////////////////
            # Plot the age estimates with no AGN continuum added
            axs[cc].axhline(df_ages.loc[cond_noagn, f"{colname} (input)"].values, color=colour_input, ls="--", lw=2, label=f"Input (no AGN)")
            axs[cc].axhline(df_ages.loc[cond_noagn, f"{colname} (regularised)"].values, color=colour_regul, label=f"Regularised fit (no AGN)")
            axs[cc].axhline(df_ages.loc[cond_noagn, f"{colname} (MC mean)"].values, color=colour_mc, ls=":", label=f"MC fit (no AGN)")
            axs[cc].axhspan(ymin=df_ages.loc[cond_noagn, f"{colname} (MC mean)"].values - df_ages.loc[cond_noagn, f"{colname} (MC error)"].values, 
                       ymax=df_ages.loc[cond_noagn, f"{colname} (MC mean)"].values + df_ages.loc[cond_noagn, f"{colname} (MC error)"].values, 
                       color=colour_mc, alpha=0.05)
              
            #//////////////////////////////////////////////////////////////////
            # Reliable age estimates  
            for aa, alpha_nu in enumerate([a for a in alpha_nu_vals if ~np.isnan(a)]):
                cond_alpha = cond_agn & (df_ages["alpha_nu"] == alpha_nu)
                # MC
                cond_isreliable = df_ages[f"Cumulative light fraction < -2.5? (MC mean)"] == False
                axs[cc].errorbar(x=df_ages.loc[cond_alpha & cond_isreliable, "x_AGN"].values, y=df_ages.loc[cond_alpha & cond_isreliable, f"{colname} (MC mean)"], yerr=df_ages.loc[cond_alpha & cond_isreliable, f"{colname} (MC error)"], 
                            marker="D", linestyle="none", markersize=5, markeredgecolor="black", ecolor="black", color=cmap_alpha_nu(aa))
                # Regul
                cond_isreliable = df_ages[f"Cumulative light fraction < -2.5? (regularised)"] == False
                axs[cc].scatter(x=df_ages.loc[cond_alpha & cond_isreliable, "x_AGN"].values, y=df_ages.loc[cond_alpha & cond_isreliable, f"{colname} (regularised)"].values,
                           marker="o", edgecolors="grey", color=cmap_alpha_nu(aa), zorder=9999)

            #//////////////////////////////////////////////////////////////////
            # UNreliable age estimates  
            for aa, alpha_nu in enumerate([a for a in alpha_nu_vals if ~np.isnan(a)]):
                cond_alpha = cond_agn & (df_ages["alpha_nu"] == alpha_nu)
                # MC
                cond_isreliable = df_ages[f"Cumulative light fraction < -2.5? (MC mean)"] == True
                axs[cc].errorbar(x=df_ages.loc[cond_alpha & cond_isreliable, "x_AGN"].values, y=df_ages.loc[cond_alpha & cond_isreliable, f"{colname} (MC mean)"], yerr=df_ages.loc[cond_alpha & cond_isreliable, f"{colname} (MC error)"], 
                            marker="D", linestyle="none", markersize=5, markeredgecolor="black", ecolor="black", color="lightgrey")
                # Regul
                cond_isreliable = df_ages[f"Cumulative light fraction < -2.5? (regularised)"] == True
                axs[cc].scatter(x=df_ages.loc[cond_alpha & cond_isreliable, "x_AGN"].values, y=df_ages.loc[cond_alpha & cond_isreliable, f"{colname} (regularised)"].values,
                           marker="o", edgecolors="grey", color="lightgrey", zorder=9999)

            #//////////////////////////////////////////////////////////////////
            # Decorations
            handles, _ = axs[cc].get_legend_handles_labels()
            handles += [Patch(facecolor=colour_mc, alpha=0.05, label=r"MC fit (no AGN) $\pm1\sigma$")] +\
                    [Line2D([0], [0], marker="o", color="grey", label="Regularised fit"), Line2D([0], [0], marker="D", color="black", ls=":", label="MC fit")] +\
                    [Patch(facecolor=cmap_alpha_nu(aa), label=r"$\alpha_\nu = %.1f$" % alpha_nu_vals[aa + 1]) for aa in range(len(alpha_nu_vals) - 1)] +\
                    [Patch(facecolor="lightgrey", alpha=1, label=r"Unreliable")]
            axs[cc].grid()
            axs[cc].set_xlabel(r"$x_{\rm AGN}$")
            axs[cc].set_ylabel(f"{weighttype} ({units})")
            # axs[cc].set_ylim([8.0, 9.0])
            # axs[1].tick_params(axis="y", reset=True)
            # axs[1].grid(True)

        axs[-1].legend(handles=handles, loc="center left", bbox_to_anchor=[1.05, 0.5], fontsize="small")
        fig.savefig(os.path.join(fig_path, f"ga{gal:004}_mw_lw_age_agn_av={A_V:.1f}_tcutoff={age_thresh:.1g}.pdf"), bbox_inches="tight", format="pdf")

