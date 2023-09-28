
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

# from IPython.core.debugger import Tracer

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
debug = False

isochrones = "Padova"

###############################################################################
# Load the DataFrame
###############################################################################
gal = int(sys.argv[1]) 
if not debug:
    df_fname = f"ga{gal:004d}_ext.hd5"
else:
    df_fname = f"ga{gal:004d}_DEBUG.hd5"
df = pd.read_hdf(os.path.join(data_path, df_fname), key="elines")

ngas_input_vals = df["Number of emission line components in input"].unique()
ngas_input_vals[np.isnan(ngas_input_vals)] = 0

# In the component with 0 lines fitted, ngascomponents is nan; replace this with 0 
df.loc[np.isnan(df["Number of emission line components in fit"]), "Number of emission line components in fit"] = 0
# Chop off the last row
df = df.iloc[:5]

#//////////////////////////////////////////////////////////////////////////////
# Accuracy of age measurements 
# Stick with a 1 Gyr cutoff for now 
#//////////////////////////////////////////////////////////////////////////////
# Get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates(isochrones)

age_thresh = 1e9
age_thresh_idx = np.nanargmin(np.abs(ages - age_thresh))

df_ages = pd.DataFrame()
for ngas_input in ngas_input_vals:
    cond = df["Number of emission line components in input"] == ngas_input

    thisrow = {}
    thisrow["ngas_input"] = ngas_input
    for weighttype in ["Mass-weighted age", "Light-weighted age", "Cumulative mass", "Cumulative light"]:
        for meastype in ["regularised", "MC mean", "MC error", "input"]:
            thisrow[f"{weighttype} vs. age cutoff ({meastype})"] = df.loc[cond, f"{weighttype} vs. age cutoff ({meastype})"].values.item()[age_thresh_idx]

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
    pp = PdfPages(os.path.join(fig_path, f"ga{gal:004d}_eline_params.pdf"))

#//////////////////////////////////////////////////////////////////////////////
# Plot age estimates, etc. as a function of the number of fitted components
#//////////////////////////////////////////////////////////////////////////////
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
fig.suptitle("How do emission lines affect the ppxf fit?")
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

    cond_nolines = df_ages["ngas_input"] == 0
    cond_lines = df_ages["ngas_input"] > 0

    axs[cc].axhline(df_ages.loc[cond_nolines, f"{colname} (input)"].values, color=colour_input, ls="--", label=f"{weighttype} (input)")
    axs[cc].axhline(df_ages.loc[cond_nolines, f"{colname} (regularised)"].values, color=colour_regul, label=f"{weighttype} (regularised, no AGN)")
    axs[cc].axhline(df_ages.loc[cond_nolines, f"{colname} (MC mean)"].values, color=colour_mc, label=f"{weighttype} (MC, no AGN)")
    axs[cc].axhspan(ymin=df_ages.loc[cond_nolines, f"{colname} (MC mean)"].values - df_ages.loc[cond_nolines, f"{colname} (MC error)"].values, 
               ymax=df_ages.loc[cond_nolines, f"{colname} (MC mean)"].values + df_ages.loc[cond_nolines, f"{colname} (MC error)"].values, 
               color=colour_mc, alpha=0.2)

    axs[cc].plot(df_ages.loc[cond_lines, "ngas_input"].values, df_ages.loc[cond_lines, f"{colname} (regularised)"].values, "o", alpha=1.0, color=colour_regul, label=f"{weighttype} (regul)")
    axs[cc].errorbar(x=df_ages.loc[cond_lines, "ngas_input"].values, 
                y=df_ages.loc[cond_lines, f"{colname} (MC mean)"], 
                yerr=df_ages.loc[cond_lines, f"{colname} (MC error)"], 
                linestyle="none", marker="D", alpha=1.0, color=colour_mc, label=f"{weighttype} (MC)")

    # Decorations 
    axs[cc].legend(loc="center left", bbox_to_anchor=[1.05, 0.5], fontsize="x-small")
    axs[cc].grid()
    axs[cc].set_xlabel(r"Number of emission line components in input")
    axs[cc].set_ylabel(f"{weighttype} ({units})")
if savefigs:
    pp.savefig(fig, bbox_inches="tight")

if savefigs:
    pp.close()
