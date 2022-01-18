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
def calculate_mw_age(sfh_mw, age_thresh, ages):
    """
    A function for computing the mass-weighted age from a star-formation
    history. 
    """
    # Sum the SFH over the metallicity dimension to get the 1D SFH
    sfh_mw_1D = np.nansum(sfh_mw, axis=0) if sfh_mw.ndim > 1 else sfh_mw

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
    sfr_avg = sfh_mw_1D / bin_sizes
    
    # Find the first index where the SFR exceed a certain value
    if np.any(sfr_avg > sfr_thresh):
        age_idx = np.argwhere(sfr_avg > sfr_thresh)[0][0]
        return ages[age_idx], age_idx
    else:
        return np.nan, np.nan
        
###########################################################################
if __name__ == "__main__":
    
    
