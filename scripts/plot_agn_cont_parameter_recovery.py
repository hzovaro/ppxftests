import os, sys 
import numpy as np
import pandas as pd
from itertools import product

from ppxftests.ssputils import load_ssp_templates

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.size"] = 14
plt.ion()
# plt.close("all")

from IPython.core.debugger import Tracer

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
    # How accurately do we recover A_V, x_AGN and alpha_nu?
    ###############################################################################
    #//////////////////////////////////////////////////////////////////////////////
    # Plot the alpha_nu template weifghts vs. A_V, x_AGN
    #//////////////////////////////////////////////////////////////////////////////
    vmin = 0
    vmax = 2.0
    for rr, A_V in enumerate(A_V_vals):
        fig, axs = plt.subplots(nrows=2, ncols=len(alpha_nu_vals) - 1, figsize=(10, 5.38))
        fig.subplots_adjust(wspace=0, hspace=0)
        cond_A_V = df["A_V (input)"] == A_V
        for cc, alpha_nu in enumerate([a for a in alpha_nu_vals if ~np.isnan(a)]):
            cond_alpha_nu = df["alpha_nu (input)"] == alpha_nu
            # Make an image 
            im_mc = np.full((len(ppxf_alpha_nu_vals), len(x_AGN_vals) - 1), np.nan)
            im_regul = np.full((len(ppxf_alpha_nu_vals), len(x_AGN_vals) - 1), np.nan)
    
            # x_AGN > 0
            for xx, x_AGN in enumerate([x for x in x_AGN_vals if ~np.isnan(x)]):
                cond_x_AGN = (df["x_AGN (input)"] == x_AGN)
                cond = cond_x_AGN & cond_A_V & cond_alpha_nu
                im_mc[:, xx] = df.loc[cond, "x_AGN (individual, MC mean)"].values.item()
                im_regul[:, xx] = df.loc[cond, "x_AGN (individual, regularised)"].values.item()

            m = axs[0][cc].imshow(im_regul, origin="lower", cmap="bone_r", vmin=vmin, vmax=vmax)  
            m = axs[1][cc].imshow(im_mc, origin="lower", cmap="bone_r", vmin=vmin, vmax=vmax)  

            # Decorations
            axs[0][cc].set_title(r"Input $\alpha_\nu = %.1f$" % alpha_nu)
            _ = [axs[rr][0].set_ylabel(r"Template $\alpha_\nu$") for rr in range(axs.shape[0])]
            _ = [axs[rr][0].set_yticklabels(ppxf_alpha_nu_vals) for rr in range(axs.shape[0])]
            _ = [axs[rr][0].set_yticks(range(len(ppxf_alpha_nu_vals))) for rr in range(axs.shape[0])]
            _ = [[ax.set_yticks([]) for ax in axs[rr][1:]] for rr in range(axs.shape[0])]

        bbox = axs[-1][-1].get_position()
        cax = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.05, 2 * bbox.height])
        fig.suptitle(f"ga{gal:004} " + r"($A_V = %.1f$)" % A_V)
        axs[0][0].text(s="Regularised fit", x=0.1, y=0.8, transform=axs[0][0].transAxes)
        axs[1][0].text(s="MC fit", x=0.1, y=0.8, transform=axs[1][0].transAxes)
        plt.colorbar(mappable=m, cax=cax)
        [ax.set_xticks(range(len(x_AGN_vals) - 1)) for ax in axs.flat]
        [ax.set_xticklabels([f"{x:.1f}" for x in x_AGN_vals if ~np.isnan(x)]) for ax in axs.flat]
        [ax.set_xlabel(r"Input $x_{\rm AGN}$") for ax in axs.flat]
        cax.set_ylabel(r"Best-fit template weight")

        fig.savefig(os.path.join(fig_path, f"ga{gal:004}_alpha_nu_av={A_V:.0f}.pdf"), bbox_inches="tight", format="pdf")

    #//////////////////////////////////////////////////////////////////////////////
    # Plot input vs. output A_V 
    #//////////////////////////////////////////////////////////////////////////////
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 7))
    fig.subplots_adjust(hspace=0, wspace=0)
    axs[0][1].set_title(f"ga{gal:004}")

    # Colourmaps
    cmap_alpha_nu = plt.cm.get_cmap("cool", len(alpha_nu_vals))

    # Case: no AGN continuum
    ax = axs.flat[0]
    ax.text(s=r"No AGN continuum", x=0.05, y=0.95, transform=ax.transAxes, verticalalignment="top", zorder=99999)
    cond_x_AGN = np.isnan(df["x_AGN (input)"])
    cond_alpha_nu = np.isnan(df["alpha_nu (input)"])
    cond = cond_x_AGN & cond_alpha_nu
    # MC
    ax.errorbar(x=df.loc[cond, "A_V (input)"].values, y=df.loc[cond, "A_V (input)"] - df.loc[cond, "A_V (MC mean)"].values, yerr=df.loc[cond, "A_V (MC error)"].values,
                    marker="D", linestyle="none", markersize=5, markeredgecolor="black", color="black")
    # Regul
    ax.scatter(x=df.loc[cond, "A_V (input)"].values, y=df.loc[cond, "A_V (input)"] - df.loc[cond, "A_V (regularised)"].values,
               marker="o", edgecolors="grey", color="black",
               label=r"Regularised fit: $\alpha_\nu = %.1f$" % alpha_nu, zorder=9999)

    # Case: AGN continuum
    for ax, x_AGN in zip(axs.flat[1:], [x for x in x_AGN_vals if ~np.isnan(x)]):
        ax.text(s=r"$x_{\rm AGN} = %.1f$" % x_AGN, x=0.05, y=0.95, transform=ax.transAxes, verticalalignment="top", zorder=99999)
        cond_x_AGN = df["x_AGN (input)"] == x_AGN

        for aa, alpha_nu in enumerate([a for a in alpha_nu_vals if ~np.isnan(a)]):
            cond_alpha_nu = df["alpha_nu (input)"] == alpha_nu
            cond = cond_x_AGN & cond_alpha_nu
            # MC
            ax.errorbar(x=df.loc[cond, "A_V (input)"].values, y=df.loc[cond, "A_V (input)"] - df.loc[cond, "A_V (MC mean)"].values, yerr=df.loc[cond, "A_V (MC error)"].values,
                        marker="D", linestyle="none", markersize=5, markeredgecolor="black", ecolor="black", color=cmap_alpha_nu(aa),
                        label=r"MC fit: $\alpha_\nu = %.1f$" % alpha_nu)
            # Regul
            ax.scatter(x=df.loc[cond, "A_V (input)"].values, y=df.loc[cond, "A_V (input)"] - df.loc[cond, "A_V (regularised)"].values,
                       marker="o", edgecolors="grey", color=cmap_alpha_nu(aa),
                       label=r"Regularised fit: $\alpha_\nu = %.1f$" % alpha_nu, zorder=9999)

    # Decorations
    for ax in axs.flat:
        ax.axhline(0, lw=0.5, color="black")
        ax.grid()
    for ax in [axs[rr][0] for rr in range(axs.shape[0])]:
        ax.set_ylabel(r"$\Delta A_V$ (input - measured)")
    for ax in [axs[-1][cc] for cc in range(axs.shape[1])]:
        ax.set_xlabel(r"$A_V$ (input)")
    axs[0][-1].legend(loc="center left", bbox_to_anchor=[1.05, 0.0])

    # Save 
    fig.savefig(os.path.join(fig_path, f"ga{gal:004}_A_V_errors.pdf"), bbox_inches="tight", format="pdf")

    #//////////////////////////////////////////////////////////////////////////////
    # Plot input vs. output A_V 
    #//////////////////////////////////////////////////////////////////////////////
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 7))
    fig.subplots_adjust(hspace=0, wspace=0)
    axs[0][1].set_title(f"ga{gal:004}")

    # Colourmaps
    cmap_alpha_nu = plt.cm.get_cmap("cool", len(alpha_nu_vals))

    # Case: no AGN continuum
    ax = axs.flat[0]
    ax.text(s=r"No AGN continuum", x=0.05, y=0.95, transform=ax.transAxes, verticalalignment="top", zorder=99999)
    cond_x_AGN = np.isnan(df["x_AGN (input)"])
    cond_alpha_nu = np.isnan(df["alpha_nu (input)"])
    cond = cond_x_AGN & cond_alpha_nu
    # MC
    ax.errorbar(x=df.loc[cond, "A_V (input)"], 
                y=(df.loc[cond, "A_V (input)"] - df.loc[cond, "A_V (MC mean)"]) / df.loc[cond, "A_V (input)"] * 100, 
                yerr=df.loc[cond, "A_V (MC error)"] / df.loc[cond, "A_V (input)"] * 100,
                marker="D", linestyle="none", markersize=5, markeredgecolor="black", color="black")
    # Regul
    ax.scatter(x=df.loc[cond, "A_V (input)"], 
               y=(df.loc[cond, "A_V (input)"] - df.loc[cond, "A_V (regularised)"]) / df.loc[cond, "A_V (input)"] * 100,
               marker="o", edgecolors="grey", color="black",
               label=r"Regularised fit: $\alpha_\nu = %.1f$" % alpha_nu, zorder=9999)

    # Case: AGN continuum
    for ax, x_AGN in zip(axs.flat[1:], [x for x in x_AGN_vals if ~np.isnan(x)]):
        ax.text(s=r"$x_{\rm AGN} = %.1f$" % x_AGN, x=0.05, y=0.95, transform=ax.transAxes, verticalalignment="top", zorder=99999)
        cond_x_AGN = df["x_AGN (input)"] == x_AGN

        for aa, alpha_nu in enumerate([a for a in alpha_nu_vals if ~np.isnan(a)]):
            cond_alpha_nu = df["alpha_nu (input)"] == alpha_nu
            cond = cond_x_AGN & cond_alpha_nu
            # MC
            ax.errorbar(x=df.loc[cond, "A_V (input)"], 
                        y=(df.loc[cond, "A_V (input)"] - df.loc[cond, "A_V (MC mean)"]) / df.loc[cond, "A_V (input)"] * 100, 
                        yerr=df.loc[cond, "A_V (MC error)"] / df.loc[cond, "A_V (input)"] * 100,
                        marker="D", linestyle="none", markersize=5, markeredgecolor="black", ecolor="black", color=cmap_alpha_nu(aa),
                        label=r"MC fit: $\alpha_\nu = %.1f$" % alpha_nu)
            # Regul
            ax.scatter(x=df.loc[cond, "A_V (input)"], 
                       y=df.loc[cond, "A_V (input)"] - df.loc[cond, "A_V (regularised)"],
                       marker="o", edgecolors="grey", color=cmap_alpha_nu(aa),
                       label=r"Regularised fit: $\alpha_\nu = %.1f$" % alpha_nu, zorder=9999)

    # Decorations
    for ax in axs.flat:
        ax.axhline(0, lw=0.5, color="black")
        ax.grid()
    for ax in [axs[rr][0] for rr in range(axs.shape[0])]:
        ax.set_ylabel(r"Error in $A_V$ (\%)")
    for ax in [axs[-1][cc] for cc in range(axs.shape[1])]:
        ax.set_xlabel(r"$A_V$ (input)")
    axs[0][-1].legend(loc="center left", bbox_to_anchor=[1.05, 0.0])

    # Save 
    fig.savefig(os.path.join(fig_path, f"ga{gal:004}_A_V_errors_pc.pdf"), bbox_inches="tight", format="pdf")

    #//////////////////////////////////////////////////////////////////////////////
    # Plot input vs. output x_AGN 
    #//////////////////////////////////////////////////////////////////////////////
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 7))
    fig.subplots_adjust(hspace=0, wspace=0)
    axs[0][1].set_title(f"ga{gal:004}")

    # Colourmaps
    cmap_alpha_nu = plt.cm.get_cmap("cool", len(alpha_nu_vals))

    # Case: AGN continuum
    for ax, A_V in zip(axs.flat, A_V_vals):
        ax.text(s=r"$A_V = %.1f$" % A_V, x=0.95, y=0.95, transform=ax.transAxes, verticalalignment="top", horizontalalignment="right", zorder=99999)
        cond_A_V = df["A_V (input)"] == A_V

        for aa, alpha_nu in enumerate([a for a in alpha_nu_vals if ~np.isnan(a)]):
            cond_alpha_nu = df["alpha_nu (input)"] == alpha_nu
            cond = cond_A_V & cond_alpha_nu
            # MC
            ax.errorbar(x=df.loc[cond, "x_AGN (input)"], y=df.loc[cond, "x_AGN (input)"] - df.loc[cond, "x_AGN (total, MC mean)"], yerr=df.loc[cond, "x_AGN (total, MC error)"],
                        marker="D", linestyle="none", markersize=5, markeredgecolor="black", ecolor="black", color=cmap_alpha_nu(aa),
                        label=r"MC fit: $\alpha_\nu = %.1f$" % alpha_nu)
            # Regul
            ax.scatter(x=df.loc[cond, "x_AGN (input)"], y=df.loc[cond, "x_AGN (input)"] - df.loc[cond, "x_AGN (total, regularised)"],
                       marker="o", edgecolors="grey", color=cmap_alpha_nu(aa),
                       label=r"Regularised fit: $\alpha_\nu = %.1f$" % alpha_nu, zorder=9999)

    # Decorations
    for ax in axs.flat:
        ax.axhline(0, lw=0.5, color="black")
        ax.grid()
    for ax in [axs[rr][0] for rr in range(axs.shape[0])]:
        ax.set_ylabel(r"$\Delta x_{\rm AGN}$ (input - measured)")
    for ax in [axs[-1][cc] for cc in range(axs.shape[1])]:
        ax.set_xlabel(r"$x_{\rm AGN}$ (input)")
    axs[0][-1].legend(loc="center left", bbox_to_anchor=[1.05, 0.0])

    # Save 
    fig.savefig(os.path.join(fig_path, f"ga{gal:004}_x_AGN_errors.pdf"), bbox_inches="tight", format="pdf")

    #//////////////////////////////////////////////////////////////////////////////
    # Plot input vs. output x_AGN (express as an error in %)
    #//////////////////////////////////////////////////////////////////////////////
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 7))
    fig.subplots_adjust(hspace=0, wspace=0)
    axs[0][1].set_title(f"ga{gal:004}")

    # Colourmaps
    cmap_alpha_nu = plt.cm.get_cmap("cool", len(alpha_nu_vals))

    # Case: no AGN continuum
    ax = axs.flat[0]
    ax.text(s=r"No AGN continuum", x=0.95, y=0.95, transform=ax.transAxes, verticalalignment="top", horizontalalignment="right", zorder=99999)
    cond_A_V = np.isnan(df["A_V (input)"])
    cond_alpha_nu = np.isnan(df["alpha_nu (input)"])
    cond = cond_A_V & cond_alpha_nu
    # MC
    ax.errorbar(x=df.loc[cond, "x_AGN (input)"], 
                y=(df.loc[cond, "x_AGN (input)"] - df.loc[cond, "x_AGN (total, MC mean)"]) / df.loc[cond, "x_AGN (input)"] * 100, 
                yerr=df.loc[cond, "x_AGN (total, MC error)"] / df.loc[cond, "x_AGN (input)"] * 100,
                marker="D", linestyle="none", markersize=5, markeredgecolor="black", color="black")
    # Regul
    ax.scatter(x=df.loc[cond, "x_AGN (input)"], 
               y=(df.loc[cond, "x_AGN (input)"] - df.loc[cond, "x_AGN (total, regularised)"]) / df.loc[cond, "x_AGN (input)"] * 100,
               marker="o", edgecolors="grey", color="black",
               label=r"Regularised fit: $\alpha_\nu = %.1f$" % alpha_nu, zorder=9999)

    # Case: AGN continuum
    for ax, A_V in zip(axs.flat[1:], A_V_vals):
        ax.text(s=r"$A_V = %.1f$" % A_V, x=0.95, y=0.95, transform=ax.transAxes, verticalalignment="top", horizontalalignment="right", zorder=99999)
        cond_A_V = df["A_V (input)"] == A_V

        for aa, alpha_nu in enumerate([a for a in alpha_nu_vals if ~np.isnan(a)]):
            cond_alpha_nu = df["alpha_nu (input)"] == alpha_nu
            cond = cond_A_V & cond_alpha_nu
            # MC
            ax.errorbar(x=df.loc[cond, "x_AGN (input)"], 
                        y=(df.loc[cond, "x_AGN (input)"] - df.loc[cond, "x_AGN (total, MC mean)"]) / df.loc[cond, "x_AGN (input)"] * 100,
                        yerr=df.loc[cond, "x_AGN (total, MC error)"] / df.loc[cond, "x_AGN (input)"] * 100,
                        marker="D", linestyle="none", markersize=5, markeredgecolor="black", ecolor="black", color=cmap_alpha_nu(aa),
                        label=r"MC fit: $\alpha_\nu = %.1f$" % alpha_nu)
            # Regul
            ax.scatter(x=df.loc[cond, "x_AGN (input)"], 
                       y=(df.loc[cond, "x_AGN (input)"] - df.loc[cond, "x_AGN (total, regularised)"]) / df.loc[cond, "x_AGN (input)"] * 100,
                       marker="o", edgecolors="grey", color=cmap_alpha_nu(aa),
                       label=r"Regularised fit: $\alpha_\nu = %.1f$" % alpha_nu, zorder=9999)

    # Decorations
    for ax in axs.flat:
        ax.axhline(0, lw=0.5, color="black")
        ax.grid()
    for ax in [axs[rr][0] for rr in range(axs.shape[0])]:
        ax.set_ylabel(r"Error in $x_{\rm AGN}$ (\%)")
    for ax in [axs[-1][cc] for cc in range(axs.shape[1])]:
        ax.set_xlabel(r"$x_{\rm AGN}$ (input)")
    axs[0][-1].legend(loc="center left", bbox_to_anchor=[1.05, 0.0])

    # Save 
    fig.savefig(os.path.join(fig_path, f"ga{gal:004}_x_AGN_errors_pc.pdf"), bbox_inches="tight", format="pdf")
