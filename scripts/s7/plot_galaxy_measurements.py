import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.ion()
plt.close("all")

"""
For each galaxy in our sample, plot the LW or MW ages, plus the best-fit x_AGN and A_V.
"""

s7_data_path = "/priv/meggs3/u5708159/S7/mar23/ppxf"
fig_path = "/priv/meggs3/u5708159/S7/mar23/ppxf/figs/paper/"

###############################################################################
# Settings
###############################################################################
savefigs = True
age_cutoff_strs = ["1e8", "1e9"]
weighttypes = ["LW", "MW"]

marker_dict = {
    "1e8": "d", 
    "1e9": "o",
}
colour_dict = {
    "FOURAS" : "blue", 
    "ONEKPC" : "red", 
    "RE1" : "green",
}

"""
In the same figure, plot (a) ages, (b) A_V and (c) x_AGN. 
Only show x-axis labels on the top/bottom w/ rotation.
"""

###############################################################################
# Create the figure in which to plot ALL aperture measurements together
###############################################################################
weighttype = "LW"

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))
ax_AGN, ax_AV, ax_ages = axs
fig.subplots_adjust(hspace=0)

dx = -0.5
for ap in ["FOURAS", "ONEKPC", "RE1"]:
    dx += 0.25

    ###############################################################################
    # Load the DataFrame
    ###############################################################################
    # Load the DataFrame containing ages 
    df = pd.read_hdf(os.path.join(s7_data_path, f"s7_ppxf_{ap}.hd5"), key="S7")
    gals = df.index.values

    ###############################################################################
    # Plot ages
    ###############################################################################
    for age_str in age_cutoff_strs:
        age = float(age_str)
        cond_good_regul = df[f"Cumulative light fraction (<{age/1e6:.2f} Myr) (regularised)"] > 10**(-2.5)
        cond_good_MC = df[f"Cumulative light fraction (<{age/1e6:.2f} Myr) (MC mean)"] > 10**(-2.5)
        
        #//////////////////////////////////////////////////////////////////////
        # Plot on the shared axes
        #//////////////////////////////////////////////////////////////////////
        # Reliable
        ax_ages.errorbar(x=df.loc[cond_good_MC, "Number"].values + dx, 
                    y=df.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"],
                    yerr=[
                        df.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 50th percentile)"] - df.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 16th percentile)"],
                        df.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 84th percentile)"] - df.loc[cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 50th percentile)"],
                    ],
                    marker=marker_dict[age_str], mfc=colour_dict[ap], mec=colour_dict[ap], ecolor=colour_dict[ap], alpha=0.5, linewidth=0.5, linestyle="none", markersize=5,
                    label=r"MC simulations ($\tau_{\rm cutoff} = %s$)" % age_cutoff_strs)
        # Unreliable
        ax_ages.errorbar(x=df.loc[~cond_good_MC, "Number"].values + dx, 
                    y=df.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC mean)"],
                    yerr=[
                        df.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 50th percentile)"] - df.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 16th percentile)"],
                        df.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 84th percentile)"] - df.loc[~cond_good_MC, f"{weighttype} age (<{age/1e6:.2f} Myr) (MC 50th percentile)"],
                    ],
                    marker=marker_dict[age_str], mfc="lightgrey", mec="grey", ecolor="grey", alpha=0.5, linewidth=0.5, linestyle="none", markersize=5,
                        label="MC simulations (unreliable, $\tau_{\rm cutoff} = %s$)" % age_cutoff_strs)
        
    ###############################################################################
    # Plot AGN
    ###############################################################################
    ax_AGN.errorbar(x=df["Number"].values + dx, 
                y=df[f"x_AGN (total, MC mean)"],
                yerr=[
                            df[f"x_AGN (total, MC 50th percentile)"] - df[f"x_AGN (total, MC 16th percentile)"],
                            df[f"x_AGN (total, MC 84th percentile)"] - df[f"x_AGN (total, MC 50th percentile)"],
                        ],
                marker="s", mfc=colour_dict[ap], mec=colour_dict[ap], ecolor=colour_dict[ap], alpha=0.5, linewidth=1.2, linestyle="none", markersize=4,)

    ###############################################################################
    # Plot A_V
    ###############################################################################
    ax_AV.errorbar(x=df["Number"].values + dx, 
                y=df[f"A_V (MC mean)"],
                yerr=[
                            df[f"A_V (MC 50th percentile)"] - df[f"A_V (MC 16th percentile)"],
                            df[f"A_V (MC 84th percentile)"] - df[f"A_V (MC 50th percentile)"],
                        ],
                marker="s", mfc=colour_dict[ap], mec=colour_dict[ap], ecolor=colour_dict[ap], alpha=0.5, linewidth=1.2, linestyle="none", markersize=4,)

# Decorations
ax_ages.set_xticklabels(gals, rotation="vertical", ha="center", va="top", fontsize="small")
ax_AGN.set_xticklabels(gals, rotation="vertical", ha="center", va="bottom", fontsize="small")
ax_ages.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
ax_AGN.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax_ages.set_ylim([6, 10.5])
ax_AV.set_ylim([-0.1, 4.5])
ax_ages.set_ylabel(f"{'Light' if weighttype == 'LW' else 'Mass'}-weighted age (log Myr)")
ax_AV.set_ylabel(r"$A_V$ (mag)")
ax_AGN.set_ylabel(r"$x_{\rm AGN}$")
for ax in axs:
    ax.set_xticks(range(len(gals)))
    ax.set_xticks(np.array(range(len(gals))) + 0.5, minor=True)
    ax.grid(b=True, which="minor", axis="x")
    ax.grid(b=True, which="major", axis="y")
    ax.autoscale(axis="x", tight=True, enable=True)
    ax.set_xlabel("Galaxy")
    ax.set_xlim([-1, len(gals)])

# Legend
legend_elements = [
    Patch(facecolor=colour_dict["RE1"], label=r"$1R_e$"),
    Patch(facecolor=colour_dict["ONEKPC"], label=r"1 kpc"),
    Patch(facecolor=colour_dict["FOURAS"], label='4"'),
]
legend_elements_ages = [
    Patch(facecolor="lightgrey", label="Unreliable"),
    Line2D([0], [0], markerfacecolor="k", markeredgecolor="none", color="none", marker=marker_dict["1e8"], label=r"$\tau_{\rm cutoff} = 100\,\rm Myr$"),
    Line2D([0], [0], markerfacecolor="k", markeredgecolor="none", color="none", marker=marker_dict["1e9"], label=r"$\tau_{\rm cutoff} = 1\,\rm Gyr$"),
]
axs[0].legend(handles=legend_elements, fontsize="small", loc="upper right", bbox_to_anchor=[0.99, 0.99])
axs[-1].legend(handles=legend_elements_ages, fontsize="small", loc="upper right", bbox_to_anchor=[0.99, 0.99])

if savefigs:
    fig.savefig(os.path.join(fig_path, f"{weighttype}_galaxy_measurements.pdf"), bbox_inches="tight", format="pdf")
