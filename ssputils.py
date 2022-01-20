import os
import numpy as np
from scipy import ndimage
import ppxf.ppxf_util as util

"""
A collection of functions for loading & manipulating the SSP templates.
"""

###############################################################################
# Convenience function for loading stellar templates
############################################################################## 
def load_ssp_templates(isochrones, metals_to_use=None):
    # Input checking
    assert isochrones == "Geneva" or isochrones == "Padova",\
        "isochrones must be either Padova or Geneva!"
    if isochrones == "Padova":
        if metals_to_use is None:
            metals_to_use = ['004', '008', '019']
        else:
            for m in metals_to_use:
                assert m in ['004', '008', '019'],\
                    f"Metallicity {m} for the {isochrones} isochrones not found!"
    elif isochrones == "Geneva":
        if metals_to_use is None:
            metals_to_use = ['001', '004', '008', '020', '040']
        else:
            for m in metals_to_use:
                assert m in ['001', '004', '008', '020', '040'],\
                    f"Metallicity {m} for the {isochrones} isochrones not found!"

    ###########################################################################
    # List of template names - one for each metallicity
    abspath = __file__.split("/ssputils.py")[0]
    ssp_template_path = os.path.join(abspath, "SSP_templates")
    ssp_template_fnames = [f"SSP{isochrones}.z{m}.npz" for m in metals_to_use]

    ###########################################################################
    # Determine how many different templates there are (i.e. N_ages x N_metallicities)
    metallicities = []
    ages = []
    for ssp_template_fname in ssp_template_fnames:
        f = np.load(os.path.join(ssp_template_path, f"SSP{isochrones}", ssp_template_fname))
        metallicities.append(f["metallicity"].item())
        ages = f["ages"] if ages == [] else ages
        lambda_vals_linear = f["lambda_vals_A"]

    # Template dimensions
    N_ages = len(ages)
    N_metallicities = len(metallicities)
    N_lambda = len(lambda_vals_linear)

    ###########################################################################
    # Create a big 3D array to hold the spectra
    stellar_templates_linear = np.zeros((N_lambda, N_metallicities, N_ages))

    for mm, ssp_template_fname in enumerate(ssp_template_fnames):
        f = np.load(os.path.join(ssp_template_path, f"SSP{isochrones}", ssp_template_fname))
        
        # Get the spectra & wavelength values
        spectra_ssp_linear = f["L_vals"]
        lambda_vals_linear = f["lambda_vals_A"]

        # Store in the big array 
        stellar_templates_linear[:, mm, :] = spectra_ssp_linear

    return stellar_templates_linear, lambda_vals_linear, metallicities, ages

###############################################################################
# Function for logarithmically re-binning the stellar templates 
###############################################################################
def log_rebin_and_convolve_stellar_templates(isochrones, metals_to_use, FWHM_inst_A, velscale):
    """
    Convenience function for logarithmically rebinning and convolving the 
    stellar 
    """
    # Load the SSP templates
    stellar_templates_linear, lambda_vals_ssp_linear, metallicities, ages =\
        load_ssp_templates(isochrones, metals_to_use)

    # Reshape stellar templates so that its dimensions are (N_lambda, N_ages * N_metallicities)
    N_metallicities = len(metallicities)
    N_ages = len(ages)
    N_lambda_ssp_linear = len(lambda_vals_ssp_linear)
    stellar_templates_linear =\
        stellar_templates_linear.reshape((N_lambda_ssp_linear, N_ages * N_metallicities))

    # Extract the wavelength range for the logarithmically rebinned templates
    _, lambda_vals_ssp_log, _ = util.log_rebin(np.array(
        [lambda_vals_ssp_linear[0], lambda_vals_ssp_linear[-1]]),
        stellar_templates_linear[:, 0], velscale=velscale)
    N_lambda_ssp_log = len(lambda_vals_ssp_log)

    # Create an array to store the logarithmically rebinned & convolved spectra in
    stellar_templates_log_conv = np.zeros((N_lambda_ssp_log, N_ages * N_metallicities))

    # Gonzalez-Delgado spectra have a constant spectral sampling of 0.3 A.
    dlambda_A_ssp = 0.30
    FWHM_ssp_A = 2 * np.sqrt(2 * np.log(2)) * dlambda_A_ssp  # This assumes that sigma_diff_px = dlambda_A_ssp.
    if FWHM_inst_A > 0:
        FWHM_diff_A = np.sqrt(FWHM_inst_A**2 - FWHM_ssp_A**2)
        sigma_diff_px = FWHM_diff_A / (2 * np.sqrt(2 * np.log(2))) / dlambda_A_ssp  # sigma_diff_px difference in pixels
    else:
        sigma_diff_px = 0

    # Convolve each SSP template to the instrumental resolution 
    stars_templates_linear_conv = np.zeros(stellar_templates_linear.shape)
    for ii in range(N_ages * N_metallicities):
        # Convolve
        if sigma_diff_px > 0:
            stars_templates_linear_conv[:, ii] =\
                ndimage.gaussian_filter1d(stellar_templates_linear[:, ii], sigma_diff_px)
        else:
            stars_templates_linear_conv[:, ii] = stellar_templates_linear[:, ii]

        # Logarithmically rebin
        spec_ssp_log, lambda_vals_ssp_log, velscale_temp =\
            util.log_rebin(np.array(
                [lambda_vals_ssp_linear[0], lambda_vals_ssp_linear[-1]]),
                stars_templates_linear_conv[:, ii],
                velscale=velscale)
        stellar_templates_log_conv[:, ii] = spec_ssp_log

    return stellar_templates_log_conv, lambda_vals_ssp_log, metallicities, ages

###############################################################################
# Function for computing the bin widths - useful for computing mean SFR in 
# each bin
############################################################################## 
def get_bin_edges_and_widths(isochrones):
    # Get the corresponding ages 
    _, _, _, ages = load_ssp_templates(isochrones)

    # Compute the bin edges
    bin_edges = np.zeros(len(ages) + 1)
    for aa in range(1, len(ages)):
        bin_edges[aa] = 10**(0.5 * (np.log10(ages[aa - 1]) + np.log10(ages[aa])) )

    # Compute the edges of the first and last bins
    delta_log_age = np.diff(np.log10(ages))[0]
    age_0 = 10**(np.log10(ages[0]) - delta_log_age)
    age_last = 10**(np.log10(ages[-1]) + delta_log_age)
    bin_edges[0] = 10**(0.5 * (np.log10(age_0) + np.log10(ages[0])) )
    bin_edges[-1] = 10**(0.5 * (np.log10(ages[-1]) + np.log10(age_last)))
    
    # Finally, compute the sizes of the bins
    bin_widths = np.diff(bin_edges)

    return bin_edges, bin_widths

##############################################################################
# Basic tests
##############################################################################
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt 
    plt.ion()
    plt.close("all")

    # Load templates
    isochrones = sys.argv[1]
    stellar_templates_linear, lambda_vals_linear, metallicities, ages =\
        load_ssp_templates(isochrones)

    # Test 2: check binning and convolution function
    VELSCALE_WIFES = 45.9896038
    FWHM_WIFES_INST_A = 1.4
    stellar_templates_log, lambda_vals_ssp_log, N_metallicities, N_ages =\
        log_rebin_and_convolve_stellar_templates(isochrones=isochrones, 
                                                 metals_to_use=None, 
                                                 FWHM_inst_A=0, 
                                                 velscale=VELSCALE_WIFES)
    stellar_templates_log_conv, lambda_vals_ssp_log_conv, N_metallicities, N_ages =\
        log_rebin_and_convolve_stellar_templates(isochrones=isochrones, 
                                                 metals_to_use=None, 
                                                 FWHM_inst_A=FWHM_WIFES_INST_A, 
                                                 velscale=VELSCALE_WIFES)
    # Reshape 
    N_lambda = len(lambda_vals_ssp_log)
    N_lambda_conv = len(lambda_vals_ssp_log_conv)
    stellar_templates_log_conv = np.reshape(stellar_templates_log_conv, 
                                            (N_lambda_conv, N_metallicities, N_ages))
    stellar_templates_log = np.reshape(stellar_templates_log, 
                                       (N_lambda, N_metallicities, N_ages))

    # Check: does convolution affect the template norms?
    lambda_norm_A = 5000
    lambda_norm_idx = np.nanargmin(np.abs(np.exp(lambda_vals_ssp_log_conv) - lambda_norm_A))
    stellar_template_norms_conv = stellar_templates_log_conv[lambda_norm_idx]
    lambda_norm_A = 5000
    lambda_norm_idx = np.nanargmin(np.abs(np.exp(lambda_vals_ssp_log) - lambda_norm_A))
    stellar_template_norms = stellar_templates_log[lambda_norm_idx]
    fig, ax = plt.subplots()
    ax.step(x=range(N_ages * N_metallicities), 
            y=np.abs(stellar_template_norms.flatten() - stellar_template_norms_conv.flatten()) / stellar_template_norms.flatten())


    # Plot before & after convolution
    met_idx = 1
    age_idx = 5
    fig, ax = plt.subplots(figsize=(15, 5))
    # Without convolution
    F_lambda = stellar_templates_log[:, met_idx, age_idx]
    ax.plot(np.exp(lambda_vals_ssp_log), F_lambda, color="grey",
            label=r"Without convolution")
    # With convolution
    F_lambda = stellar_templates_log_conv[:, met_idx, age_idx]
    ax.plot(np.exp(lambda_vals_ssp_log_conv), F_lambda, color="black",
            label=r"With convolution")
    ax.set_xlabel(f"Wavelength $\lambda$ (Å)")
    ax.set_ylabel(r"$F_\lambda(\lambda) / F_\lambda(5000\,Å)$")
    ax.set_title(r"$Z = %.3f, t = %.2f\,\rm Myr$" % (metallicities[met_idx], ages[age_idx] / 1e6))
    ax.legend()  

    # Test 1: Plot a few templates to check that they look as expected 
    for met_idx in range(len(metallicities)):
        fig, ax = plt.subplots(figsize=(15, 5))
        lambda_idx = np.nanargmin(np.abs(lambda_vals_linear - 5000))
        for age_idx in [0, 10, 25, -1]:
            F_lambda = stellar_templates_linear[:, met_idx, age_idx]
            F_lambda /= F_lambda[lambda_idx]
            ax.plot(lambda_vals_linear, F_lambda,
                    label=r"$Z = %.3f, t = %.2f\,\rm Myr$" % (metallicities[met_idx], ages[age_idx] / 1e6))
        ax.set_xlabel(f"Wavelength $\lambda$ (Å)")
        ax.set_ylabel(r"$F_\lambda(\lambda) / F_\lambda(5000\,Å)$")
        ax.legend()  

    # Check that the bin sizes are correct
    bin_edges, bin_widths = get_bin_edges_and_widths(isochrones=isochrones)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(x=ages, y=np.ones(len(ages)), label="Bin centres")
    ax.scatter(x=bin_edges, y=np.ones(len(bin_edges)), label="Bin edges")
    ax.set_xscale("log")
    ax.set_xlabel("Age (yr)")
    ax.legend()

    plt.show()


