# Imports
import os 
import numpy as np
import pandas as pd

from ppxftests.ppxf_plot import plot_sfh_mass_weighted, plot_sfh_light_weighted, plot_sfr
from ppxftests.ssputils import load_ssp_templates, get_bin_edges_and_widths, log_rebin_and_convolve_stellar_templates
from ppxftests.mockspec import FWHM_WIFES_INST_A, VELSCALE_WIFES

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

_, _, metallicities_padova, ages_padova = load_ssp_templates("Padova")
_, _, metallicities_geneva, ages_geneva = load_ssp_templates("Geneva")
bin_edges_padova, bin_widths_padova = get_bin_edges_and_widths(isochrones="Padova")
bin_edges_geneva, bin_widths_geneva = get_bin_edges_and_widths(isochrones="Geneva")

###########################################################################
def make_gal_info_df():
    """
    Create a Pandas DataFrame containing the SMBH masses and Eddington 
    ratios for each galaxy, primarily so we can estimate stellar velocity
    dispersions using the M - sigma relation.
    """
    abspath = __file__.split("/sfhutils.py")[0]

    # Eddington ratios
    df_edd = pd.read_csv(os.path.join(abspath, "sim_galaxies", "eddratio.txt"))

    # BH masses
    df_mbh = pd.read_csv(os.path.join(abspath, "sim_galaxies", "mbh.txt"))

    # Merge
    df = df_edd.merge(df_mbh, right_index=True, left_index=True)
    df.index.name = "ID"

    # Compute stellar velocity dispersions for each galaxy
    # M_BH - sigma_* relation taken from Gültekin+2009
    # https://arxiv.org/pdf/0903.4897.pdf
    alpha = 8.12
    beta = 4.24
    df["sigma_* (km/s)"] = 200 * 10**(1 / beta * (np.log10(df["M_BH"]) - alpha))

    # Save to file 
    df.to_csv(os.path.join(abspath, "sim_galaxies", "gal_metadata.csv"))

    return

###########################################################################
def load_sfh(gal, isochrones="Padova", plotit=False):
    """
    Load a SFH from one of Phil's simulated galaxies.
    Return both mass-weighted and light-weighted template weights, where 
    the light-weighted template weights are computed assuming a reference
    wavelength of 5000 Å and instrumental parameters corresponding to WiFeS.
    """
    assert isochrones == "Padova", "for now, isochrones must be 'Padova'!"
    assert type(gal) == int, f"gal must be an integer!"
    abspath = __file__.split("/sfhutils.py")[0]
    fname = os.path.join(abspath, "sim_galaxies", "SFHs", f"SFH_ga{gal:04}.dat")
    assert os.path.exists(fname), f"SFH file {fname} not found!"

    # Load the file 
    f = open(fname)
    sfh_mw = np.array([l.split() for l in f.readlines()]).astype(float).T
    M_tot = np.nansum(sfh_mw)

    if isochrones == "Padova":
        metallicities, ages = metallicities_padova, ages_padova
    elif isochrones == "Geneva":
        metallicities, ages = metallicities_geneva, ages_geneva

    assert sfh_mw.shape[0] == len(metallicities),\
        f"Loaded SFH has zeroth dimension {sfh_mw.shape[0]} but should have dimension {len(metallicities)} for the {isochrones} isochrones!"
    assert sfh_mw.shape[1] == len(ages),\
        f"Loaded SFH has first dimension {sfh_mw.shape[1]} but should have dimension {len(ages)} for the {isochrones} isochrones!"

    # Convert mass weights to light weights 
    sfh_lw = convert_mass_weights_to_light_weights(sfh_mw, isochrones=isochrones)

    # Compute the bin edges and widths so that we can compute the mean SFR in each bin
    if isochrones == "Padova":
        bin_edges, bin_widths, ages = bin_edges_padova, bin_widths_padova, ages_padova
    elif isochrones == "Geneva":
        bin_edges, bin_widths, ages = bin_edges_geneva, bin_widths_geneva, ages_geneva

    # Compute the mean SFR in each bin
    sfr_avg = np.nansum(sfh_mw, axis=0) / bin_widths

    # Get the stellar velocity dispersion 
    df_metadata = pd.read_csv(os.path.join(abspath, "sim_galaxies", "gal_metadata.csv"))
    sigma_star_kms = df_metadata.loc[gal, "sigma_* (km/s)"]

    # Plot the SFH
    if plotit:
        plot_sfh_mass_weighted(sfh_mw, ages, metallicities)
        plt.gcf().get_axes()[0].set_title(f"Galaxy {gal:004} " + r"- $M_{\rm tot} = %.4e\,\rm M_\odot$" % M_tot)
        plot_sfh_light_weighted(sfh_lw, ages, metallicities)
        plt.gcf().get_axes()[0].set_title(f"Galaxy {gal:004} " + r"- $M_{\rm tot} = %.4e\,\rm M_\odot$" % M_tot)
        plot_sfr(sfr_avg, ages, metallicities)
        plt.gcf().get_axes()[0].set_title(f"Galaxy {gal:004} " + r"- $M_{\rm tot} = %.4e\,\rm M_\odot$" % M_tot)

    return sfh_mw, sfh_lw, sfr_avg, sigma_star_kms

###########################################################################
def convert_mass_weights_to_light_weights(sfh_mw, isochrones,
                                          metals_to_use=None,
                                          lambda_norm_A=5000,
                                          FWHM_inst_A=FWHM_WIFES_INST_A,
                                          velscale=VELSCALE_WIFES):
    """
    Mass-weighted template weights essentially indicate the total mass 
    contributed by each template in a system. Therefore, given that each
    template represents the spectrum in erg/s of 1 Msun, the weights are 
    in units of solar masses. Therefore, the total spectrum is given by 

                F = sum w_i * f_i

    where w_i is the mass weight and f_i is the spectrum of template i, 
    corresponding to a SSP with a total mass of 1 Msun. 

    The weights returned by ppxf are different, due to the necessary 
    normalisation of the stellar templates and of the input spectrum. The 
    weights returned by ppxf have the defition

                F / F(lambda_0) = sum w'_i * [f_i / f_i(lambda_0)]

    The mass weights can therefore be derived from the ppxf weights via 

                w_i,MW = w'_i * F(lambda_0) / f_i(lambda_0)

    However, sometimes it is useful to express the weights not by the amount of 
    mass that each template contributes, but by the amount of light each 
    template contributes at a specified wavelength. In this case, we can 
    compute the light-weighted weights via 

                w_i,LW = w'_i * F(lambda_0)

    The light weights essentially tell us how many erg/s/Å each template 
    contributes at the reference wavelength (lambda_norm_A). Added together,
    they give the total flux in erg/s/Å in the continuum at the reference
    wavelength. 

    Therefore, if we have the mass-weighted template weights, we can compute 
    the light-weighted weights via 

                w_i,LW = w_i,MW * f_i(lambda_0)

    However, to do this requires f_i(lambda_0) for each template. We therefore
    need to load each template and bin & convolve it in the same way that we
    need to do in run_ppxf() to compute the normalisation factors.

    """

    # Load the stellar templates, so that we can compute the normalisation
    # factors for each template
    stellar_templates, lambda_vals_ssp_log, metallicities, ages =\
        log_rebin_and_convolve_stellar_templates(isochrones=isochrones, 
                                                 metals_to_use=metals_to_use, 
                                                 FWHM_inst_A=FWHM_inst_A, 
                                                 velscale=velscale)

    N_metallicities = len(metallicities)
    N_ages = len(ages)
    N_lambda = len(lambda_vals_ssp_log)
    stellar_templates = np.reshape(stellar_templates, 
                                   (N_lambda, N_metallicities, N_ages))

    # Compute normalisation factors
    lambda_norm_idx = np.nanargmin(np.abs(np.exp(lambda_vals_ssp_log) - lambda_norm_A))
    stellar_template_norms = stellar_templates[lambda_norm_idx]

    # Multiply the mass-weighted template weights to get the light weights
    sfh_lw = sfh_mw * stellar_template_norms

    return sfh_lw
 
###########################################################################
def compute_mass(sfh_mw, isochrones,
                 age_thresh_lower=None,
                 age_thresh_upper=None):
    """
    Given a SFH, compute the total mass in a specified age range.
    """
    # Sum the SFH over the metallicity dimension to get the 1D SFH
    sfh_mw_1D = np.nansum(sfh_mw, axis=0) if sfh_mw.ndim > 1 else sfh_mw

    ages = ages_padova if isochrones == "Padova" else ages_geneva
    if age_thresh_lower is None:
        age_thresh_lower = ages[0]
    if age_thresh_upper is None: 
        age_thresh_upper = ages[-1]

    # Find the index of the threshold age in the template age array
    age_thresh_lower_idx = np.nanargmin(np.abs(ages - age_thresh_lower))
    age_thresh_upper_idx = np.nanargmin(np.abs(ages - age_thresh_upper))

    M = np.nansum(sfh_mw_1D[age_thresh_lower_idx:age_thresh_upper_idx])

    return M

###########################################################################
def compute_mw_age(sfh_mw, isochrones,
                   age_thresh_lower=None,
                   age_thresh_upper=None):
    """
    A function for computing the mass-weighted age from a star-formation
    history. 
    """
    assert isochrones in ["Padova", "Geneva"],\
        "isochrones must be either 'Padova' or 'Geneva'!"

    # Sum the SFH over the metallicity dimension to get the 1D SFH
    sfh_mw_1D = np.nansum(sfh_mw, axis=0) if sfh_mw.ndim > 1 else sfh_mw

    ages = ages_padova if isochrones == "Padova" else ages_geneva
    if age_thresh_lower is None:
        age_thresh_lower = ages[0]
    if age_thresh_upper is None: 
        age_thresh_upper = ages[-1]

    # Find the index of the threshold age in the template age array
    age_thresh_lower_idx = np.nanargmin(np.abs(ages - age_thresh_lower))
    age_thresh_upper_idx = np.nanargmin(np.abs(ages - age_thresh_upper))
    
    # Compute the mass-weighted age 
    log_age_mw = np.nansum(sfh_mw_1D[age_thresh_lower_idx:age_thresh_upper_idx] * np.log10(ages[age_thresh_lower_idx:age_thresh_upper_idx])) / np.nansum(sfh_mw_1D[age_thresh_lower_idx:age_thresh_upper_idx])
    
    # Compute the corresponding index in the array (useful for plotting)
    log_age_mw_idx = (log_age_mw - np.log10(ages[0])) / (np.log10(ages[1]) - np.log10(ages[0]))
    
    return log_age_mw, log_age_mw_idx

###########################################################################
def compute_sfrw_age(sfr_avg, isochrones,
                     age_thresh_lower=None,
                     age_thresh_upper=None):
    """
    A function for computing the SFR-weighted age from a star-formation
    history. 
    """
    assert isochrones in ["Padova", "Geneva"],\
        "isochrones must be either 'Padova' or 'Geneva'!"

    ages = ages_padova if isochrones == "Padova" else ages_geneva
    if age_thresh_lower is None:
        age_thresh_lower = ages[0]
    if age_thresh_upper is None: 
        age_thresh_upper = ages[-1]

    # Find the index of the threshold age in the template age array
    age_thresh_lower_idx = np.nanargmin(np.abs(ages - age_thresh_lower))
    age_thresh_upper_idx = np.nanargmin(np.abs(ages - age_thresh_upper))
    
    # Compute the mass-weighted age 
    log_age_sfrw = np.nansum(sfr_avg[age_thresh_lower_idx:age_thresh_upper_idx] * np.log10(ages[age_thresh_lower_idx:age_thresh_upper_idx])) / np.nansum(sfr_avg[age_thresh_lower_idx:age_thresh_upper_idx])
    
    # Compute the corresponding index in the array (useful for plotting)
    log_age_sfrw_idx = (log_age_sfrw - np.log10(ages[0])) / (np.log10(ages[1]) - np.log10(ages[0]))
    
    return log_age_sfrw, log_age_sfrw_idx

###########################################################################
def compute_lw_age(sfh_lw, isochrones,
                   age_thresh_lower=None,
                   age_thresh_upper=None):
    """
    A function for computing the mass-weighted age from a star-formation
    history. 
    """
    assert isochrones in ["Padova", "Geneva"],\
        "isochrones must be either 'Padova' or 'Geneva'!"

    # Sum the SFH over the metallicity dimension to get the 1D SFH
    sfh_lw_1D = np.nansum(sfh_lw, axis=0) if sfh_lw.ndim > 1 else sfh_lw

    ages = ages_padova if isochrones == "Padova" else ages_geneva
    if age_thresh_lower is None:
        age_thresh_lower = ages[0]
    if age_thresh_upper is None: 
        age_thresh_upper = ages[-1]

    # Find the index of the threshold age in the template age array
    age_thresh_lower_idx = np.nanargmin(np.abs(ages - age_thresh_lower))
    age_thresh_upper_idx = np.nanargmin(np.abs(ages - age_thresh_upper))
    
    # Compute the light-weighted age 
    log_age_lw = np.nansum(sfh_lw_1D[age_thresh_lower_idx:age_thresh_upper_idx] * np.log10(ages[age_thresh_lower_idx:age_thresh_upper_idx])) / np.nansum(sfh_lw_1D[age_thresh_lower_idx:age_thresh_upper_idx])
    
    # Compute the corresponding index in the array (useful for plotting)
    log_age_lw_idx = (log_age_lw - np.log10(ages[0])) / (np.log10(ages[1]) - np.log10(ages[0]))
    
    return log_age_lw, log_age_lw_idx


###########################################################################
def compute_sfr_thresh_age(sfh_mw, sfr_thresh, isochrones):
    """
    A function for computing the most recent time at which the SFR exceeded
    a specified threshold in a given star formation history.
    """
    assert isochrones in ["Padova", "Geneva"],\
        "isochrones must be either 'Padova' or 'Geneva'!"

    # Sum the SFH over the metallicity dimension to get the 1D SFH
    sfh_mw_1D = np.nansum(sfh_mw, axis=0) if sfh_mw.ndim > 1 else sfh_mw

    # Compute the bin edges and widths so that we can compute the mean SFR in each bin
    if isochrones == "Padova":
        bin_edges, bin_widths, ages = bin_edges_padova, bin_widths_padova, ages_padova
    elif isochrones == "Geneva":
        bin_edges, bin_widths, ages = bin_edges_geneva, bin_widths_geneva, ages_geneva

    # Compute the mean SFR in each bin
    sfr_avg = sfh_mw_1D / bin_widths
    
    # Find the first index where the SFR exceed a certain value
    if np.any(sfr_avg > sfr_thresh):
        age_idx = np.argwhere(sfr_avg > sfr_thresh)[0][0]
        return np.log10(ages[age_idx]), age_idx
    else:
        return np.nan, np.nan
  
###########################################################################
def compute_sb_zero_age(sfh_mw, isochrones):
    """

    """
    # Sum the SFH over the metallicity dimension to get the 1D SFH
    sfh_mw_1D = np.nansum(sfh_mw, axis=0) if sfh_mw.ndim > 1 else sfh_mw

    if isochrones == "Padova":
        ages = ages_padova
    elif isochrones == "Geneva":
        ages = ages_geneva

    first_nonzero_idx = np.argwhere(sfh_mw_1D > 0)[0][0]
    if np.any(sfh_mw_1D[first_nonzero_idx:] == 0):
        first_zero_idx = np.argwhere(sfh_mw_1D[first_nonzero_idx:] == 0)[0][0] + first_nonzero_idx
        return np.log10(ages[first_zero_idx]), first_zero_idx
    else:
        return np.nan, np.nan


###########################################################################
# Convenience functions for computing mean quantities from a list of ppxf instances 
###########################################################################
def compute_mean_1D_sfh(pp_list, isochrones, weighttype):

    """
    Convenience function for computing the mean SFH given a list of ppxf
    instances.
    """
    assert isochrones in ["Padova", "Geneva"],\
        "isochrones must be either 'Padova' or 'Geneva'!"
    assert weighttype == "lw" or weighttype == "mw",\
        "weighttype must be 'lw' or 'mw'!"

    if weighttype == "lw":
        sfh_list = [pp.sfh_lw_1D for pp in pp_list]
    elif weighttype == "mw":
        sfh_list = [pp.sfh_mw_1D for pp in pp_list]
    sfh_1D_mean = np.nansum(np.array(sfh_list), axis=0) / len(sfh_list)

    return sfh_1D_mean

def compute_mean_mass(pp_list, isochrones, age_thresh_lower, age_thresh_upper):

    """
    Convenience function for computing the mean & std. dev. of the total mass
    in the range [age_thresh_lower, age_thresh_upper] given a list of ppxf instances.
    """
    assert isochrones in ["Padova", "Geneva"],\
        "isochrones must be either 'Padova' or 'Geneva'!"

    sfh_list = [pp.weights_mass_weighted for pp in pp_list]
    mass_list = [compute_mass(sfh_mw, isochrones, age_thresh_lower, age_thresh_upper) for sfh_mw in sfh_list]
    mass_mean = np.nanmean(mass_list)
    mass_std = np.nanstd(mass_list)
    return mass_mean, mass_std


def compute_mean_sfr(pp_list, isochrones):

    """
    Convenience function for computing the mean SFR given a list of ppxf
    instances.
    """
    assert isochrones in ["Padova", "Geneva"],\
        "isochrones must be either 'Padova' or 'Geneva'!"

    sfr_list = [pp.sfr_mean for pp in pp_list]
    sfr_mean = np.nansum(np.array(sfr_list), axis=0) / len(sfr_list)
    return sfr_mean

def compute_mean_age(pp_list, isochrones, weighttype, age_thresh_lower, age_thresh_upper):

    """
    Convenience function for computing the mean & std. dev. of the mass-
    weighted age in the range [age_thresh_lower, age_thresh_upper] given 
    a list of ppxf instances.
    """
    assert isochrones in ["Padova", "Geneva"],\
        "isochrones must be either 'Padova' or 'Geneva'!"
    assert weighttype == "lw" or weighttype == "mw" or weighttype == "sfrw",\
        "weighttype must be 'lw' or 'mw'!"
    
    if weighttype == "mw":
        sfh_list = [pp.weights_mass_weighted for pp in pp_list]
        age_list = [10**compute_mw_age(sfh, isochrones, age_thresh_lower, age_thresh_upper)[0] for sfh in sfh_list]
        age_mean = np.nanmean(age_list)
        age_std = np.nanstd(age_list)
        
    elif weighttype == "lw":
        sfh_list = [pp.weights_light_weighted for pp in pp_list]
        age_list = [10**compute_lw_age(sfh, isochrones, age_thresh_lower, age_thresh_upper)[0] for sfh in sfh_list]
        age_mean = np.nanmean(age_list)
        age_std = np.nanstd(age_list)

    elif weighttype == "sfrw":
        sfr_avg_list = [pp.sfr_mean for pp in pp_list]
        age_list = [10**compute_sfrw_age(sfr_avg, isochrones, age_thresh_lower, age_thresh_upper)[0] for sfr_avg in sfr_avg_list]
        age_mean = np.nanmean(age_list)
        age_std = np.nanstd(age_list)
    
    return age_mean, age_std

def compute_mean_sfr_thresh_age(pp_list, isochrones, sfr_thresh):

    """
    Convenience function for computing the mean and std. dev. in the
    SFR threshold age from a list of ppxf instances.
    """
    assert isochrones in ["Padova", "Geneva"],\
        "isochrones must be either 'Padova' or 'Geneva'!"
    sfr_age_list = [10**compute_sfr_thresh_age(pp.weights_mass_weighted, sfr_thresh, isochrones)[0] for pp in pp_list]
    sfr_age_mean = np.nanmean(sfr_age_list)
    sfr_age_std = np.nanstd(sfr_age_list)

    return sfr_age_mean, sfr_age_std


###########################################################################
if __name__ == "__main__":
    # Check that our functions are all working properly
    from ppxftests.ssputils import load_ssp_templates
    from ppxftests.ppxf_plot import plot_sfh_mass_weighted, plot_sfh_light_weighted
    from ppxftests.mockspec import create_mock_spectrum

    # Load age and metallicity values
    isochrones = "Padova"
    _, _, metallicities, ages = load_ssp_templates(isochrones)
    bin_edges, bin_widths = get_bin_edges_and_widths(isochrones)
    N_ages = len(ages)
    N_metallicities = len(metallicities)

    # Load the SFH
    # gal = 42
    # sfh = load_sfh(gal, isochrones=isochrones, plotit=True)
    sfh = np.zeros((N_metallicities, N_ages))
    sfh[1, 5] = 1e9

    sfh_1D = np.nansum(sfh, axis=0)
    sfr_avg = sfh_1D / bin_widths
    M_tot = np.nansum(sfh)

    ####################################################################
    # Test conversion between light weights and mass weights
    ####################################################################
    sfh_lw = convert_mass_weights_to_light_weights(sfh, isochrones)
    plot_sfh_mass_weighted(sfh, ages=ages, metallicities=metallicities)
    plot_sfh_light_weighted(sfh_lw, ages=ages, metallicities=metallicities)

    spec, spec_err, lambda_vals_wifes_A =\
        create_mock_spectrum(sfh_mass_weighted=sfh,
                             agn_continuum=False,
                             isochrones=isochrones, z=0, SNR=1e10, sigma_star_kms=0.1,
                             plotit=True)

    # Check: for this simple SFH, the flux in the spectrum at 5000Å
    # should equal the template weight at [1, 5].
    # Slight discrepancy (on order of <1% are probably because the WiFeS 
    # wavelength resolution is coarser than that of the templates, meaning 
    # the measured flux isn't precisely at 5000Å)
    print(f"Spectrum: log F(5000Å) = {np.log10(spec[np.nanargmin(np.abs(lambda_vals_wifes_A - 5000))]):.4f}")
    print(f"Template weight: log w_(1, 5) = {np.log10(sfh_lw[1, 5]):.4f}")

    ####################################################################
    # Test age estimation methods 
    ####################################################################
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

