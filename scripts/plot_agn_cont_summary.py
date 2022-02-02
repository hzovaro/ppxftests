import os, sys 
import numpy as np
import pandas as pd
from itertools import product

from ppxftests.ssputils import load_ssp_templates

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

"""
For each galaxy we saved a DataFrame for in test_agn_continuum.py, plot
- "summary" showing the SFHs, MW/LW mean ages vs. time, etc.

For the sample as a whole, plot
- LW/MW age (for some given cutoff) vs. 
    - x_AGN
    - alpha_nu 
- accuracy of fitted PL continuum: not sure  

"""

data_path = "/priv/meggs3/u5708159/ppxftests/"
fig_path = "/priv/meggs3/u5708159/ppxftests/figs/agn_continuum"

###############################################################################
# Settings
###############################################################################
fit_agn_cont = True
savefigs = True

isochrones = "Padova"

###############################################################################
# Load the DataFrame
###############################################################################
gal = int(sys.argv[1])
df = pd.read_hdf(os.path.join(data_path, f"ga{gal}_agncont_agncontfit={'true' if fit_agn_cont else 'false'}.hd5"), key="agn")

x_AGN_vals = df["x_AGN"].unique()
alpha_nu_vals = df["alpha_nu"].unique()

# Multi-page pdfs
if savefigs:
    pp_sfh = PdfPages(os.path.join(fig_path, f"ga{gal}_agncont_agncontfit={'true' if fit_agn_cont else 'false'}_sfhs.pdf"))
    pp_meas = PdfPages(os.path.join(fig_path, f"ga{gal}_agncont_agncontfit={'true' if fit_agn_cont else 'false'}_meas.pdf"))

###############################################################################
# Summary plot
###############################################################################
# Get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates(isochrones)


# For each combination of alpha_nu, x_AGN, make this plot
for alpha_nu, x_AGN in product(alpha_nu_vals, x_AGN_vals):
    # IF both are NaN, then do NOT add an AGN continuum
    if np.isnan(x_AGN) and np.isnan(alpha_nu):
        title_str = f"ga{gal:004} (no AGN continuum added)"
        cond = np.isnan(df["x_AGN"]) & np.isnan(df["alpha_nu"])
    elif np.isnan(x_AGN) or np.isnan(alpha_nu):
        continue
    else:
        title_str = f"ga{gal:004} " + r"($\alpha_\nu = %.2f, \, x_{\rm AGN} = %.2f$)" % (alpha_nu, x_AGN)
        cond = (df["x_AGN"] == x_AGN) & (df["alpha_nu"] == alpha_nu)

    #//////////////////////////////////////////////////////////////////////////////
    # Plot the SFH
    #//////////////////////////////////////////////////////////////////////////////
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    for weighttype, ax in zip(["Mass", "Light"], axs):
        norm = np.nansum(df.loc[cond, f"SFH {weighttype[0]}W 1D (input)"].values.item())
        
        ax.step(x=ages, y=df.loc[cond, f"SFH {weighttype[0]}W 1D (input)"].values.item() / norm, where="mid", color="blue", label="SFH (input)", linewidth=2.5)
        ax.step(x=ages, y=df.loc[cond, f"SFH {weighttype[0]}W 1D (regularised)"].values.item()[0] / norm, where="mid", color="indigo", label="SFH (regularised)", linewidth=1.0)
        ax.step(x=ages, y=df.loc[cond, f"SFH {weighttype[0]}W 1D (MC mean)"].values.item()[0] / norm, where="mid", color="cornflowerblue", label="SFH (MC)", linewidth=1.0)
        ax.errorbar(x=ages, 
                        y=df.loc[cond, f"SFH {weighttype[0]}W 1D (MC mean)"].values.item()[0] / norm,
                        yerr=df.loc[cond, f"SFH {weighttype[0]}W 1D (MC error)"].values.item()[0] / norm,
                        linestyle="none", color="cornflowerblue")
        ax.axhline(1e-4, color="k", ls="--", linewidth=1)

        # Decorations
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend(loc="best", fontsize="x-small")
        ax.grid()
        ax.set_ylabel(f"{weighttype} fraction")
        ax.set_xlabel("Age (yr)")
        ax.autoscale(axis="x", tight=True, enable=True)

    axs[0].set_title(title_str)
    if savefigs:
        pp_sfh.savefig(fig, bbox_inches="tight")

    #//////////////////////////////////////////////////////////////////////////////
    # Plot the mean mass- and light-weighted age vs. age threshold
    # expressed as an ERROR
    #//////////////////////////////////////////////////////////////////////////////
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    fig.subplots_adjust(left=0.025, right=1 - 0.025, top=1 - 0.05, bottom=0.025)
    fig.suptitle(title_str)

    colour_lists = [
        ("red", "maroon", "pink"),
        ("green", "darkgreen", "lightgreen"),
        ("blue", "indigo", "cornflowerblue"),
        ("gold", "brown", "orange")
    ]

    # PLOT: 
    for cc, weighttype in enumerate(["Mass-weighted age", "Light-weighted age", "Cumulative mass", "Cumulative light"]):
        colname = f"{weighttype} vs. age cutoff"
        colour_input, colour_regul, colour_mc = colour_lists[cc]

        #//////////////////////////////////////////////////////////////////////
        # Plot the measured values vs. age
        axs[0][cc].step(x=ages[1:], y=df.loc[cond, f"{colname} (input)"].values.item(), label=f"{weighttype}", where="mid", linewidth=2.5, color=colour_input)
        axs[0][cc].step(x=ages[1:], y=df.loc[cond, f"{colname} (regularised)"].values.item()[0], label=f"{weighttype} (regularised)", where="mid", color=colour_regul)
        y_meas = df.loc[cond, f"{colname} (MC mean)"].values.item()[0]
        y_err = df.loc[cond, f"{colname} (MC error)"].values.item()[0]
        axs[0][cc].step(x=ages[1:], y=y_meas, where="mid", color=colour_mc, label=f"{weighttype} (MC)")
        axs[0][cc].fill_between(x=ages[1:], y1=y_meas - y_err, y2=y_meas + y_err, step="mid", alpha=0.2, color=colour_mc)

        # Decorations  
        axs[0][cc].axvspan(ages[1], ages[np.argwhere(np.isfinite(df.loc[cond, f"{colname} (input)"].values.item()))[0][0]], color="grey", alpha=0.2)  
        axs[0][cc].set_xscale("log")
        axs[0][cc].legend(loc="best", fontsize="x-small")
        axs[0][cc].set_xlabel("Age threshold (yr)")
        axs[0][cc].set_ylabel(f"{weighttype} (log yr)" if weighttype.endswith("age") else f"{weighttype} " + r"($M_\odot$)")
        axs[0][cc].set_xlim([ages[0], ages[-1]])
        axs[0][cc].grid()

        #//////////////////////////////////////////////////////////////////////
        # Plot the measured values - the input values ("delta") vs. age
        axs[1][cc].step(x=ages[1:], y=df.loc[cond, f"{colname} (regularised)"].values.item()[0] - df.loc[cond, f"{colname} (input)"].values.item(), 
                    label=f"{colname} (regularised)", where="mid", color=colour_regul)
        log_y_meas = df.loc[cond, f"{colname} (MC mean)"].values.item()[0]
        log_y_input = df.loc[cond, f"{colname} (input)"].values.item()
        log_y_err = df.loc[cond, f"{colname} (MC error)"].values.item()[0]
        log_dy = log_y_meas - log_y_input
        log_dy_lower = log_y_meas - log_y_err - log_y_input
        log_dy_upper = log_y_meas + log_y_err - log_y_input
        axs[1][cc].step(x=ages[1:], y=log_dy, where="mid", color=colour_mc, label=f"{weighttype} (MC)")
        axs[1][cc].fill_between(x=ages[1:], y1=log_dy_lower, y2=log_dy_upper, step="mid", alpha=0.2, color=colour_mc)
        
        # Decorations
        axs[1][cc].axvspan(ages[1], ages[np.argwhere(np.isfinite(log_y_input))[0][0]], color="grey", alpha=0.2)
        axs[1][cc].axhline(0, color="k")   
        axs[1][cc].set_xscale("log")
        axs[1][cc].legend(loc="best", fontsize="x-small")
        axs[1][cc].set_xlabel("Age threshold (yr)")
        axs[1][cc].set_ylabel(r"$\Delta$" + f" {colname} error (log yr)")
        axs[1][cc].set_xlim([ages[0], ages[-1]])
        axs[1][cc].grid()
        axs[1][cc].autoscale(axis="x", tight=True, enable=True)

    #//////////////////////////////////////////////////////////////////////////////
    # AGN template weights 
    #//////////////////////////////////////////////////////////////////////////////
    alpha_nu_vals_ppxf = df.loc[cond, "ppxf alpha_nu_vals"].values.item()[0]
    w_ppxf_regul = df.loc[cond, "AGN template weights (regularised)"].values.item()[0]
    x_AGN_fit_vals_regul = (1 / w_ppxf_regul - 1)**(-1)
    w_ppxf_mc = df.loc[cond, "AGN template weights (MC mean)"].values.item()[0]
    w_ppxf_mc_err = df.loc[cond, "AGN template weights (MC error)"].values.item()[0]
    x_AGN_fit_vals_mc = (1 / w_ppxf_mc - 1)**(-1)
    x_AGN_fit_vals_mc_err = w_ppxf_mc_err / (1 - w_ppxf_mc)**2

    axs[0][-1].plot(alpha_nu_vals_ppxf, x_AGN_fit_vals_regul, "bo", label="AGN template weights (regularised)")
    axs[0][-1].errorbar(x=alpha_nu_vals_ppxf, y=x_AGN_fit_vals_mc, yerr=x_AGN_fit_vals_mc_err, 
                marker="o", color="cornflowerblue", linestyle="none", label="AGN template weights (MC)")
    if ~np.isnan(alpha_nu):
        axs[0][-1].axvline(alpha_nu, color="black", label=r"$\alpha_\nu$ (input)")
        axs[0][-1].axhline(x_AGN, color="black", label=r"$x_{\rm AGN}$ (input)")
        axs[0][-1].axhline(np.nansum(x_AGN_fit_vals_regul), ls="--", color="purple", label=r"Total $x_{\rm AGN}$ (regularised)")
        axs[0][-1].axhline(np.nansum(x_AGN_fit_vals_mc), ls="--", color="cornflowerblue", label=r"Total $x_{\rm AGN}$ (MC)")
        axs[0][-1].axhspan(ymin=np.nansum(x_AGN_fit_vals_mc) - np.sqrt(np.nansum(x_AGN_fit_vals_mc_err**2)),
                   ymax=np.nansum(x_AGN_fit_vals_mc) + np.sqrt(np.nansum(x_AGN_fit_vals_mc_err**2)), 
                   alpha=0.2, color="cornflowerblue")
    else:
        axs[0][-1].axhline(0, color="black", label=r"$x_{\rm AGN}$ (input)")
    axs[0][-1].legend(fontsize="x-small", loc="best")
    axs[0][-1].set_xticks(alpha_nu_vals_ppxf)
    axs[0][-1].set_xlabel(r"$\alpha_\nu$")
    axs[0][-1].set_ylabel(r"$x_{\rm AGN}$")
    axs[0][-1].grid()
    axs[0][-1].set_xlim([0, 3.0])

    axs[1][-1].set_visible(False)

    # Save figures
    if savefigs:
        pp_meas.savefig(fig, bbox_inches="tight")

    if not savefigs:
        Tracer()()
    plt.close("all")


# Finish writing to file
if savefigs:
    pp_sfh.close()
    pp_meas.close()
