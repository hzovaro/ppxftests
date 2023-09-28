import os, sys 
import numpy as np
import pandas as pd
from itertools import product

from ppxftests.ssputils import load_ssp_templates

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.ion()
plt.close("all")

# from IPython.core.debugger import Tracer

"""
How well do we recover mean age measures and other properties when an AGN 
continuum is present?

In this script we will plot light/mass-weighted ages as a function of 
alpha_nu and x_AGN.
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
gal = int(sys.argv[1]) 

# True == plot the results from the fits w/ the 4th-order polynomial; 
# False == plot the results from the fit w/ the extinction curve 
mpoly = False  

if not debug:
    df_fname = f"ga{gal:004d}_{'mpoly' if mpoly else 'ext'}.hd5"
else:
    df_fname = f"ga{gal:004d}_DEBUG.hd5"
df = pd.read_hdf(os.path.join(data_path, df_fname), key="ext_and_agn")

x_AGN_vals = df["x_AGN (input)"].unique()
alpha_nu_vals = df["alpha_nu (input)"].unique()
A_V_vals = df["A_V (input)"].unique()

#//////////////////////////////////////////////////////////////////////////////
# Accuracy of age measurements 
# Stick with a 1 Gyr cutoff for now 
#//////////////////////////////////////////////////////////////////////////////
# Get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates(isochrones)

age_thresh = 1e9
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

    df_ages = df_ages.append(thisrow, ignore_index=True)

# Plot
colour_lists = [
    ("red", "maroon", "pink"),
    ("green", "darkgreen", "lightgreen"),
    ("blue", "indigo", "cornflowerblue"),
    ("gold", "brown", "orange")
]

###############################################################################
# Plotting
###############################################################################
if savefigs:
    pp = PdfPages(os.path.join(fig_path, f"ga{gal:004d}_av_and_agn_params.pdf"))

#//////////////////////////////////////////////////////////////////////////////
# Plot age estimates, etc. as a function of x_AGN for varying A_V, alpha_nu
#//////////////////////////////////////////////////////////////////////////////
for A_V in A_V_vals:
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
    fig.suptitle(f"A_V = {A_V:.1f}")
    axs[0][1].set_visible(False)
    axs[1][1].set_visible(False)
    axs[0][3].set_visible(False)
    axs[1][3].set_visible(False)
    axs = [axs[0][0], axs[1][0], axs[0][2], axs[1][2]]
    for cc, weighttype in enumerate(["Mass-weighted age", "Light-weighted age", "Cumulative mass", "Cumulative light"]):
        colour_input, colour_regul, colour_mc = colour_lists[cc]
        colname = f"{weighttype} vs. age cutoff"
        if weighttype.endswith("age"):
            units = r"$\log \,\rm yr$"
        elif weighttype.endswith("mass"):
            units = r"$\log \,\rm M_\odot$"
        elif weighttype.endswith("light"):
            units = r"$\log \,\rm erg\, s^{-1}$"

        cond_noagn = (df_ages["A_V"] == A_V) & np.isnan(df_ages["x_AGN"]) & np.isnan(df_ages["alpha_nu"])
        cond_agn = (df_ages["A_V"] == A_V) & ~np.isnan(df_ages["x_AGN"]) & ~np.isnan(df_ages["alpha_nu"])

        axs[cc].axhline(df_ages.loc[cond_noagn, f"{colname} (input)"].values, color=colour_input, ls="--", label=f"{weighttype} (input)")
        axs[cc].axhline(df_ages.loc[cond_noagn, f"{colname} (regularised)"].values, color=colour_regul, label=f"{weighttype} (regularised, no AGN)")
        axs[cc].axhline(df_ages.loc[cond_noagn, f"{colname} (MC mean)"].values, color=colour_mc, label=f"{weighttype} (MC, no AGN)")
        axs[cc].axhspan(ymin=df_ages.loc[cond_noagn, f"{colname} (MC mean)"].values - df_ages.loc[cond_noagn, f"{colname} (MC error)"].values, 
                   ymax=df_ages.loc[cond_noagn, f"{colname} (MC mean)"].values + df_ages.loc[cond_noagn, f"{colname} (MC error)"].values, 
                   color=colour_mc, alpha=0.2)
            
        alpha = 1. / len(alpha_nu_vals)
        for alpha_nu in [a for a in alpha_nu_vals if ~np.isnan(a)]:
            cond_alpha = cond_agn & (df_ages["alpha_nu"] == alpha_nu)
            axs[cc].plot(df_ages.loc[cond_alpha, "x_AGN"].values, df_ages.loc[cond_alpha, f"{colname} (regularised)"].values, "o", alpha=alpha, color=colour_regul, label=f"{weighttype} (regul) " + r"($\alpha_\nu = %.1f$)" % alpha_nu)
            axs[cc].errorbar(x=df_ages.loc[cond_alpha, "x_AGN"].values, 
                        y=df_ages.loc[cond_alpha, f"{colname} (MC mean)"], 
                        yerr=df_ages.loc[cond_alpha, f"{colname} (MC error)"], 
                        linestyle="none", marker="D", alpha=alpha, color=colour_mc, label=f"{weighttype} (MC) " + r"($\alpha_\nu = %.1f$)" % alpha_nu)
            alpha += 1. / len(alpha_nu_vals)

        # Decorations 
        axs[cc].legend(loc="center left", bbox_to_anchor=[1.05, 0.5], fontsize="x-small")
        axs[cc].grid()
        axs[cc].set_xlabel(r"$x_{\rm AGN}$")
        axs[cc].set_ylabel(f"{weighttype} ({units})")
    if savefigs:
        pp.savefig(fig, bbox_inches="tight")

#//////////////////////////////////////////////////////////////////////////////
# Plot age estimates, etc. as a function of alpha_nu for varying A_V, x_AGN
#//////////////////////////////////////////////////////////////////////////////
for A_V in A_V_vals:
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
    fig.suptitle(f"A_V = {A_V:.1f}")
    axs[0][1].set_visible(False)
    axs[1][1].set_visible(False)
    axs[0][3].set_visible(False)
    axs[1][3].set_visible(False)
    axs = [axs[0][0], axs[1][0], axs[0][2], axs[1][2]]
    for cc, weighttype in enumerate(["Mass-weighted age", "Light-weighted age", "Cumulative mass", "Cumulative light"]):
        colour_input, colour_regul, colour_mc = colour_lists[cc]
        colname = f"{weighttype} vs. age cutoff"
        if weighttype.endswith("age"):
            units = r"$\log \,\rm yr$"
        elif weighttype.endswith("mass"):
            units = r"$\log \,\rm M_\odot$"
        elif weighttype.endswith("light"):
            units = r"$\log \,\rm erg\, s^{-1}$"

        cond_noagn = (df_ages["A_V"] == A_V) & np.isnan(df_ages["x_AGN"]) & np.isnan(df_ages["alpha_nu"])
        cond_agn = (df_ages["A_V"] == A_V) & ~np.isnan(df_ages["x_AGN"]) & ~np.isnan(df_ages["alpha_nu"])

        axs[cc].axhline(df_ages.loc[cond_noagn, f"{colname} (input)"].values, color=colour_input, ls="--", label=f"{weighttype} (input)")
        axs[cc].axhline(df_ages.loc[cond_noagn, f"{colname} (regularised)"].values, color=colour_regul, label=f"{weighttype} (regularised, no AGN)")
        axs[cc].axhline(df_ages.loc[cond_noagn, f"{colname} (MC mean)"].values, color=colour_mc, label=f"{weighttype} (MC, no AGN)")
        axs[cc].axhspan(ymin=df_ages.loc[cond_noagn, f"{colname} (MC mean)"].values - df_ages.loc[cond_noagn, f"{colname} (MC error)"].values, 
                   ymax=df_ages.loc[cond_noagn, f"{colname} (MC mean)"].values + df_ages.loc[cond_noagn, f"{colname} (MC error)"].values, 
                   color=colour_mc, alpha=0.2)

        alpha = 1. / len(x_AGN_vals)
        for x_AGN in [x for x in x_AGN_vals if ~np.isnan(x)]:
            cond_x_AGN = cond_agn & (df_ages["x_AGN"] == x_AGN)
            axs[cc].plot(df_ages.loc[cond_x_AGN, "alpha_nu"].values, df_ages.loc[cond_x_AGN, f"{colname} (regularised)"].values, "o", alpha=alpha, color=colour_regul, label=f"{weighttype} (regul) " + r"($x_{\rm AGN} = %.1f$)" % x_AGN)
            axs[cc].errorbar(x=df_ages.loc[cond_x_AGN, "alpha_nu"].values, 
                        y=df_ages.loc[cond_x_AGN, f"{colname} (MC mean)"], 
                        yerr=df_ages.loc[cond_x_AGN, f"{colname} (MC error)"], 
                        linestyle="none", marker="D", alpha=alpha, color=colour_mc, label=f"{weighttype} (MC) " + r"($x_{\rm AGN} = %.1f$)" % x_AGN)
            alpha += 1. / len(x_AGN_vals)

        # Decorations 
        axs[cc].legend(loc="center left", bbox_to_anchor=[1.05, 0.5], fontsize="x-small")
        axs[cc].grid()
        axs[cc].set_xlabel(r"$\alpha_\nu$")
        axs[cc].set_ylabel(f"{weighttype} ({units})")
    if savefigs:
        pp.savefig(fig, bbox_inches="tight")

if savefigs:
    pp.close()


