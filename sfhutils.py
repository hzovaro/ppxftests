# Imports
import os 
import numpy as np

from ppxftests.ppxf_plot import plot_sfh_mass_weighted
from ppxftests.ssputils import load_ssp_templates, get_bin_edges_and_widths

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

###########################################################################
def load_sfh(gal, isochrones="Padova", plotit=False):
    """
    Load a SFH from one of Phil's simulated galaxies.
    """
    assert type(gal) == int, f"gal must be an integer!"
    abspath = __file__.split("/sfhutils.py")[0]
    fname = os.path.join(abspath, "SFHs", f"SFH_ga{gal:04}.dat")
    assert os.path.exists(fname), f"SFH file {fname} not found!"

    # Load the file 
    f = open(fname)
    sfh_mw = np.array([l.split() for l in f.readlines()]).astype(float).T
    M_tot = np.nansum(sfh_mw)

    _, _, metallicities, ages = load_ssp_templates(isochrones)
    assert sfh_mw.shape[0] == len(metallicities),\
        f"Loaded SFH has zeroth dimension {sfh_mw.shape[0]} but should have dimension {len(metallicities)} for the {isochrones} isochrones!"
    assert sfh_mw.shape[1] == len(ages),\
        f"Loaded SFH has first dimension {sfh_mw.shape[1]} but should have dimension {len(ages)} for the {isochrones} isochrones!"

    # Plot the SFH
    if plotit:
        plot_sfh_mass_weighted(sfh_mw, ages, metallicities)
        plt.gcf().get_axes()[0].set_title(f"Galaxy {gal:004} " + r"- $M_{\rm tot} = %.4e\,\rm M_\odot$" % M_tot)

    return sfh_mw

###########################################################################
def compute_mw_age(sfh_mw, age_thresh, isochrones):
    """
    A function for computing the mass-weighted age from a star-formation
    history. 
    """
    # Sum the SFH over the metallicity dimension to get the 1D SFH
    sfh_mw_1D = np.nansum(sfh_mw, axis=0) if sfh_mw.ndim > 1 else sfh_mw

    # Get ages
    _, _, metallicities, ages = load_ssp_templates(isochrones)

    # Find the index of the threshold age in the template age array
    age_thresh_idx = np.nanargmin(np.abs(ages - age_thresh))
    
    # Compute the mass-weighted age 
    log_age_mw = np.nansum(sfh_mw_1D[:age_thresh_idx] * np.log10(ages[:age_thresh_idx])) / np.nansum(sfh_mw_1D[:age_thresh_idx])
    
    # Compute the corresponding index in the array (useful for plotting)
    log_age_mw_idx = (log_age_mw - np.log10(ages[0])) / (np.log10(ages[1]) - np.log10(ages[0]))
    
    return log_age_mw, log_age_mw_idx

###########################################################################
def compute_sfr_thresh_age(sfh_mw, sfr_thresh, isochrones):
    """
    A function for computing the most recent time at which the SFR exceeded
    a specified threshold in a given star formation history.
    """
    # Sum the SFH over the metallicity dimension to get the 1D SFH
    sfh_mw_1D = np.nansum(sfh_mw, axis=0) if sfh_mw.ndim > 1 else sfh_mw

    # Compute the bin edges and widths so that we can compute the mean SFR in each bin
    bin_edges, bin_widths = get_bin_edges_and_widths(isochrones=isochrones)
    
    # Compute the mean SFR in each bin
    sfr_avg = sfh_mw_1D / bin_widths
    
    # Find the first index where the SFR exceed a certain value
    if np.any(sfr_avg > sfr_thresh):
        age_idx = np.argwhere(sfr_avg > sfr_thresh)[0][0]
        return ages[age_idx], age_idx
    else:
        return np.nan, np.nan
  
###########################################################################
def compute_sb_zero_age(sfh_mw):
    """

    """
    # Sum the SFH over the metallicity dimension to get the 1D SFH
    sfh_mw_1D = np.nansum(sfh_mw, axis=0) if sfh_mw.ndim > 1 else sfh_mw
        
    first_nonzero_idx = np.argwhere(sfh_mw_1D > 0)[0][0]
    if np.any(sfh_mw_1D[first_nonzero_idx:] == 0):
        first_zero_idx = np.argwhere(sfh_mw_1D[first_nonzero_idx:] == 0)[0][0] + first_nonzero_idx
        return ages[first_zero_idx], first_zero_idx
    else:
        return np.nan, np.nan

###########################################################################
if __name__ == "__main__":
    # Check that our functions are all working properly
    from ppxftests.ssputils import load_ssp_templates

    # Load age and metallicity values
    isochrones = "Padova"
    _, _, metallicities, ages = load_ssp_templates(isochrones)
    bin_edges, bin_widths = get_bin_edges_and_widths(isochrones)
    N_ages = len(ages)
    N_metallicities = len(metallicities)

    # Load the SFH
    gal = 42
    sfh = load_sfh(gal, isochrones=isochrones, plotit=True)
    sfh_1D = np.nansum(sfh, axis=0)
    sfr_avg = sfh_1D / bin_widths
    M_tot = np.nansum(sfh)

    # Parameters for age estimators
    mass_frac_thresh = 1e-2
    age_thresh = 1e8
    age_thresh_idx = (np.log10(age_thresh) - np.log10(ages[0])) / (np.log10(ages[1]) - np.log10(ages[0]))
    sfr_thresh = 1

    age_sb, age_idx_sb = compute_sb_zero_age(sfh)  # "Starburst" age
    age_mw, age_idx_mw = compute_mw_age(sfh, age_thresh, isochrones)  # Mass-weighted age
    age_sfr, age_idx_sfr = compute_sfr_thresh_age(sfh, sfr_thresh, isochrones)  # SFR threshold

    print(f"compute_sb_zero_age(): {age_sb / 1e6:.2f} Myr")
    print(f"compute_mw_age(): {age_mw / 1e6:.2f} Myr")
    print(f"compute_sfr_thresh_age(): {age_sfr / 1e6:.2f} Myr")

    ####################################################################
    # Plot the SFH and the cumulative mass counting from t = 0
    ####################################################################
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.subplots_adjust(bottom=0.3, top=0.9)

    # Plot the SFH and the cumulative mass from t = 0
    ax.step(x=range(N_ages), y=sfh_1D / M_tot, color="black", where="mid", label="1D SFH")
    ax.step(x=range(N_ages), y=np.cumsum(sfh_1D) / M_tot, where="mid", label="Cumulative mass", linewidth=0.5)

    # Indicate each SB age measure
    ax.axvline(age_idx_sb, color="grey", label="Starburst age")
    ax.axvline(age_idx_mw, color="blue", label=f"Mass-weighted mean age (< {age_thresh / 1e6:.0f} Myr)")
    ax.axvline(age_idx_sfr, color="green", label="SFR threshold age")

    ax.axvline(age_thresh_idx, linestyle="--", color="grey", label="Age threshold")

    ax.set_yscale("log")
    ax.set_xticks(range(N_ages))
    ax.set_xticklabels(ages / 1e6, rotation="vertical", fontsize="x-small")
    ax.grid()
    ax.legend(fontsize="x-small")
    ax.autoscale(axis="x", tight=True, enable=True)
    ax.set_ylabel(r"Stellar mass fraction $M/M_{\rm tot}$")
    ax.set_xlabel("Bin age (Myr)")

    ####################################################################
    # Plot the mean SFR in each bin
    ####################################################################
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.subplots_adjust(bottom=0.3, top=0.9)
    ax.step(x=range(N_ages), y=sfr_avg, where="mid", label="Average SFR")

    # Indicate each SB age measure
    ax.axvline(age_idx_sb, color="grey", label="Starburst age")
    ax.axvline(age_idx_mw, color="blue", label=f"Mass-weighted mean age (< {age_thresh / 1e6:.0f} Myr)")
    ax.axvline(age_idx_sfr, color="green", label="SFR threshold age")

    ax.axhline(sfr_thresh, linestyle=":", color="grey", label="SFR threshold")
    ax.axvline(age_thresh_idx, linestyle="--", color="grey", label="Age threshold")

    ax.set_yscale("log")
    ax.set_xticks(range(N_ages))
    ax.set_xticklabels(ages / 1e6, rotation="vertical", fontsize="x-small")
    ax.grid()
    ax.legend(fontsize="x-small")
    ax.autoscale(axis="x", tight=True, enable=True)
    ax.set_ylabel(r"Mean SFR ($\rm M_\odot \, yr^{-1}$)")
    ax.set_xlabel("Bin age (Myr)")

    ####################################################################
    # Plot the mass-weighted mean age as a function of age threshold
    ####################################################################
    mw_mean_ages = [compute_mw_age(sfh, age_thresh, isochrones)[0] for age_thresh in ages]
    fig, ax = plt.subplots()
    ax.scatter(ages, mw_mean_ages)
    ax.set_xlabel("Age threshold (Myr)")
    ax.set_ylabel("Mass-weighted mean age (Myr)")
    ax.set_xscale("log")
    ax.set_yscale("log")

