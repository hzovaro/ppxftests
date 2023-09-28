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
# Plot the SFH
#//////////////////////////////////////////////////////////////////////////////
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
for weighttype, ax in zip(["Mass", "Light"], axs):
    ax.step(x=ages, y=df.loc[cond, f"SFH {weighttype[0]}W 1D (input)"].values.item(), where="mid", color="blue", label="SFH (input)", linewidth=2.5)
    ax.step(x=ages, y=df.loc[cond, f"SFH {weighttype[0]}W 1D (regularised)"].values.item()[0], where="mid", color="purple", label="SFH (regularised)", linewidth=1.5)
    ax.step(x=ages, y=df.loc[cond, f"SFH {weighttype[0]}W 1D (MC mean)"].values.item()[0], where="mid", color="cornflowerblue", label="SFH (MC)", linewidth=1.0)
    ax.errorbar(x=ages, 
                    y=df.loc[cond, f"SFH {weighttype[0]}W 1D (MC mean)"].values.item()[0],
                    yerr=df.loc[cond, f"SFH {weighttype[0]}W 1D (MC error)"].values.item()[0],
                    linestyle="none", color="cornflowerblue")
    # ax.axhline(1e-4, color="k", ls="--", linewidth=1)

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
axs[0].legend(loc="best", fontsize="small")
axs[0].set_title(title_str)

# Save figure
if savefigs:
    fig.savefig(os.path.join(fig_path, f"ga{gal:004}_input_vs_output_sfh.pdf"), bbox_inches="tight", format="pdf")
