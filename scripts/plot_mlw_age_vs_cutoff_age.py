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

data_path = "/priv/meggs3/u5708159/ppxftests/ext_and_agn/"
fig_path = "/priv/meggs3/u5708159/ppxftests/figs/"

###############################################################################
# Settings
###############################################################################
savefigs = True
isochrones = "Padova"

###############################################################################
# Load the DataFrame
###############################################################################
gal = int(sys.argv[1]) 

# True == plot the results from the fits w/ the 4th-order polynomial; 
# False == plot the results from the fit w/ the extinction curve 
mpoly = False  

# Load the DataFrame
df_fname = f"ga{gal:004d}_{'mpoly' if mpoly else 'ext'}.hd5"
df = pd.read_hdf(os.path.join(data_path, df_fname), key="ext_and_agn")

# Parameter combination to plot
x_AGN = np.nan
alpha_nu = np.nan
A_V = 0.0

###############################################################################
# Summary plot
###############################################################################
# Get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates(isochrones)

# For each combination of alpha_nu, x_AGN, make this plot
print(f"Processing parameter combination alpha_nu = {alpha_nu:.1f}, x_AGN = {x_AGN:.1f}, A_V = {A_V:.1f}...")
# IF both are NaN, then do NOT add an AGN continuum
if np.isnan(x_AGN) and np.isnan(alpha_nu):
    title_str = f"ga{gal:004} (no emission lines, no AGN continuum, " + r" $A_V = %.2f)$" % A_V
    cond = np.isnan(df["x_AGN (input)"]) & np.isnan(df["alpha_nu (input)"]) & (df["A_V (input)"] == A_V)
else:
    title_str = f"ga{gal:004} " + r"(no emission lines, $\alpha_\nu = %.2f, \, x_{\rm AGN} = %.2f,\,A_V = %.2f$)" % (alpha_nu, x_AGN, A_V)
    cond = (df["x_AGN (input)"] == x_AGN) & (df["alpha_nu (input)"] == alpha_nu) & (df["A_V (input)"] == A_V)

#//////////////////////////////////////////////////////////////////////////////
# Plot the mean mass- and light-weighted age vs. age threshold
# expressed as an ERROR
#//////////////////////////////////////////////////////////////////////////////
# weighttypes = ["Mass-weighted age", "Light-weighted age", "Cumulative mass", "Cumulative light"]
weighttypes = ["Mass-weighted age", "Light-weighted age"]
fig, axs = plt.subplots(nrows=2, ncols=len(weighttypes), figsize=(5 * len(weighttypes), 8))
fig.subplots_adjust(left=0.025, right=1 - 0.025, top=1 - 0.05, bottom=0.025)
fig.suptitle(title_str)

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
    axs[0][cc].step(x=ages[1:], y=df.loc[cond, f"{colname} (input)"].values.item(), label=f"{weighttype}", where="mid", linewidth=2.5, color=colour_input)
    axs[0][cc].step(x=ages[1:], y=df.loc[cond, f"{colname} (regularised)"].values.item()[0], label=f"{weighttype} (regularised)", where="mid", color=colour_regul)
    y_meas = df.loc[cond, f"{colname} (MC mean)"].values.item()[0]
    y_err = df.loc[cond, f"{colname} (MC error)"].values.item()[0]
    axs[0][cc].step(x=ages[1:], y=y_meas, where="mid", color=colour_mc, label=f"{weighttype} (MC)")
    axs[0][cc].fill_between(x=ages[1:], y1=y_meas - y_err, y2=y_meas + y_err, step="mid", alpha=0.2, color=colour_mc)

    # Decorations  
    axs[0][cc].axvspan(ages[1], ages[np.argwhere(np.isfinite(df.loc[cond, f"{colname} (input)"].values.item()))[0][0]], color="grey", alpha=0.2)  
    axs[0][cc].set_xscale("log")
    axs[0][cc].legend(loc="upper left", fontsize="small")
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
    axs[1][cc].legend(loc="upper left", fontsize="small")
    axs[1][cc].set_xlabel("Age threshold (yr)")
    axs[1][cc].set_ylabel(r"$\Delta$" + f" {colname} error (log yr)")
    axs[1][cc].set_xlim([ages[0], ages[-1]])
    axs[1][cc].grid()
    axs[1][cc].autoscale(axis="x", tight=True, enable=True) 

if savefigs:
    fig.savefig(os.path.join(fig_path, f"ga{gal:004}_mlw_age_vs_cutoff_age.pdf"), bbox_inches="tight", format="pdf")   
