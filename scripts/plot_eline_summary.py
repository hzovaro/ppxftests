
###############################################################################
# Plot MW/LW age, cumulative measures, etc. as a function of cutoff age,
# for each fitting run.
###############################################################################
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
Want to investigate how the measured quantities (ages, etc.) vary as a function
of the number of components fitted.
"""

data_path = "/priv/meggs3/u5708159/ppxftests/elines/"
fig_path = "/priv/meggs3/u5708159/ppxftests/figs/elines/"

###############################################################################
# Settings
###############################################################################
savefigs = True
debug = True

isochrones = "Padova"

###############################################################################
# Load the DataFrame
###############################################################################
gal = int(sys.argv[1]) 
if not debug:
    df_fname = f"ga{gal:004d}.hd5"
else:
    df_fname = f"ga{gal:004d}_DEBUG.hd5"
df = pd.read_hdf(os.path.join(data_path, df_fname), key="elines")

ngascomponents_vals = df["Number of emission line components in fit"].unique()
x_AGN = df["x_AGN (input)"].unique()
alpha_nu = df["alpha_nu (input)"].unique()

# Multi-page pdfs
if savefigs:
    pp_sfh = PdfPages(os.path.join(fig_path, f"ga{gal}_sfhs.pdf"))
    pp_meas = PdfPages(os.path.join(fig_path, f"ga{gal}_meas.pdf"))

###############################################################################
# Summary plot
###############################################################################
# Get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates(isochrones)


# For each combination of alpha_nu, x_AGN, make this plot
for ngascomponents in ngascomponents_vals:
    print(f"Processing parameter combination ngascomponents = {ngascomponents:.1f}...")
    # IF both are NaN, then do NOT add an AGN continuum
    if np.isnan(ngascomponents):
        title_str = f"ga{gal:004} (no emission lines)"
        cond = np.isnan(df["Number of emission line components in fit"])
    else:
        title_str = f"ga{gal:004} " + r"(ngascomponents = %d)" % (ngascomponents)
        cond = df["Number of emission line components in fit"] == ngascomponents

    if not np.any(cond):
        print("Row missing from DataFrame! Skipping...")
        continue
  
    #//////////////////////////////////////////////////////////////////////////////
    # Plot the SFH
    #//////////////////////////////////////////////////////////////////////////////
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))
    for weighttype, ax in zip(["Mass", "Light"], axs):
        norm = np.nansum(df.loc[cond, f"SFH {weighttype[0]}W 1D (input)"].values.item())
        
        ax.step(x=ages, y=df.loc[cond, f"SFH {weighttype[0]}W 1D (input)"].values.item() / norm, where="mid", color="blue", label="SFH (input)", linewidth=2.5)
        ax.step(x=ages, y=df.loc[cond, f"SFH {weighttype[0]}W 1D (regularised)"].values.item() / norm, where="mid", color="indigo", label="SFH (regularised)", linewidth=1.0)
        ax.step(x=ages, y=df.loc[cond, f"SFH {weighttype[0]}W 1D (MC mean)"].values.item() / norm, where="mid", color="cornflowerblue", label="SFH (MC)", linewidth=1.0)
        ax.errorbar(x=ages, 
                        y=df.loc[cond, f"SFH {weighttype[0]}W 1D (MC mean)"].values.item() / norm,
                        yerr=df.loc[cond, f"SFH {weighttype[0]}W 1D (MC error)"].values.item() / norm,
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
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(22, 8))
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
        axs[0][cc].step(x=ages[1:], y=df.loc[cond, f"{colname} (regularised)"].values.item(), label=f"{weighttype} (regularised)", where="mid", color=colour_regul)
        y_meas = df.loc[cond, f"{colname} (MC mean)"].values.item()
        y_err = df.loc[cond, f"{colname} (MC error)"].values.item()
        axs[0][cc].step(x=ages[1:], y=y_meas, where="mid", color=colour_mc, label=f"{weighttype} (MC)")
        axs[0][cc].fill_between(x=ages[1:], y1=y_meas - y_err, y2=y_meas + y_err, step="mid", alpha=0.2, color=colour_mc)

        # Decorations  
        axs[0][cc].axvspan(ages[1], ages[np.argwhere(np.isfinite(df.loc[cond, f"{colname} (input)"].values.item()))[0]], color="grey", alpha=0.2)  
        axs[0][cc].set_xscale("log")
        axs[0][cc].legend(loc="upper left", fontsize="x-small")
        axs[0][cc].set_xlabel("Age threshold (yr)")
        axs[0][cc].set_ylabel(f"{weighttype} (log yr)" if weighttype.endswith("age") else f"{weighttype} " + r"($M_\odot$)")
        axs[0][cc].set_xlim([ages[0], ages[-1]])
        axs[0][cc].grid()

        #//////////////////////////////////////////////////////////////////////
        # Plot the measured values - the input values ("delta") vs. age
        axs[1][cc].step(x=ages[1:], y=df.loc[cond, f"{colname} (regularised)"].values.item() - df.loc[cond, f"{colname} (input)"].values.item(), 
                    label=f"{colname} (regularised)", where="mid", color=colour_regul)
        log_y_meas = df.loc[cond, f"{colname} (MC mean)"].values.item()
        log_y_input = df.loc[cond, f"{colname} (input)"].values.item()
        log_y_err = df.loc[cond, f"{colname} (MC error)"].values.item()
        log_dy = log_y_meas - log_y_input
        log_dy_lower = log_y_meas - log_y_err - log_y_input
        log_dy_upper = log_y_meas + log_y_err - log_y_input
        axs[1][cc].step(x=ages[1:], y=log_dy, where="mid", color=colour_mc, label=f"{weighttype} (MC)")
        axs[1][cc].fill_between(x=ages[1:], y1=log_dy_lower, y2=log_dy_upper, step="mid", alpha=0.2, color=colour_mc)
        
        # Decorations
        axs[1][cc].axvspan(ages[1], ages[np.argwhere(np.isfinite(log_y_input))[0][0]], color="grey", alpha=0.2)
        axs[1][cc].axhline(0, color="k")   
        axs[1][cc].set_xscale("log")
        axs[1][cc].legend(loc="upper left", fontsize="x-small")
        axs[1][cc].set_xlabel("Age threshold (yr)")
        axs[1][cc].set_ylabel(r"$\Delta$" + f" {colname} error (log yr)")
        axs[1][cc].set_xlim([ages[0], ages[-1]])
        axs[1][cc].grid()
        axs[1][cc].autoscale(axis="x", tight=True, enable=True)

    #//////////////////////////////////////////////////////////////////////////////
    # AGN template weights 
    #//////////////////////////////////////////////////////////////////////////////
    alpha_nu_vals_ppxf = df.loc[cond, "ppxf alpha_nu_vals"].values.item()
    x_AGN_fit_vals_regul = df.loc[cond, "x_AGN (individual, regularised)"].values.item()
    x_AGN_fit_vals_mc = df.loc[cond, "x_AGN (individual, MC mean)"].values.item()
    x_AGN_fit_vals_mc_err = df.loc[cond, "x_AGN (individual, MC error)"].values.item()

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

    #//////////////////////////////////////////////////////////////////////////////
    # Add as text the A_V 
    #//////////////////////////////////////////////////////////////////////////////
    s = r"Input $A_V = %.2f$" % df.loc[cond, "A_V (input)"].values[0] + "\n" +\
        r"Regularised $A_V = %.2f$" % df.loc[cond, "A_V (regularised)"].values[0] + "\n" +\
        r"MC $A_V = %.2f \pm %.2f$" % (df.loc[cond, "A_V (MC mean)"].values[0], df.loc[cond, "A_V (MC error)"].values[0])
    axs[0][0].text(s=s, x=0.95, y=0.05, transform=axs[0][0].transAxes, verticalalignment="bottom", horizontalalignment="right")

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