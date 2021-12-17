# Imports
import os 
import numpy as np

from scipy import constants
from scipy.signal import convolve
from scipy.interpolate import CubicSpline

from itertools import product

import ppxf.ppxf_util as util

from cosmocalc import get_dist
from ppxftests.ppxf_plot import plot_sfh_mass_weighted
from ppxftests.ssputils import load_ssp_templates

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

###############################################################################
# Paths 
FWHM_WIFES_INST_A = 1.4

###############################################################################
def create_mock_spectrum(sfh_mass_weighted, isochrones, sigma_star_kms, z, SNR,
                         metals_to_use=None, plotit=True):
    
    """
    Create a mock spectrum given an input star formation history, stellar 
    velocity dispersion, redshift and SNR.

    Inputs:
    sfh_mass_weighted       an N x M array of mass weights corresponding to 
                            the SSP templates, where N = number of metallicity
                            dimensions and M = number of age dimensions. The 
                            weights should be in units of solar masses.
    isochrones              which set of isochrones to use; must be either 
                            Padova or Geneva 
    metals_to_use           List of template metallicities (in string form) to
                            assume in the mock spectrum. If None, then use 
                            all of the available metallicities.
    sigma_star_kms          stellar velocity dispersion, in km/s
    z                       redshift
    SNR                     Assumed signal-to-noise ratio in the outoput 
                            spectrum.

    Returns:
    spec, spec_err          mock spectrum and corresponding 1-sigma errors, in
                            units of erg/s. 

    """
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
    N_metallicities, N_ages = sfh_mass_weighted.shape
    assert N_metallicities == len(metals_to_use),\
        f"sfh_mass_weighted.shape[0] = {N_metallicities} but len(metals_to_use) = {len(metals_to_use)}!"

    ###########################################################################
    # WIFES Instrument properties
    ###########################################################################
    # Compute the width of the LSF kernel we need to apply to the templates
    FWHM_inst_A = FWHM_WIFES_INST_A    # for the WiFeS COMB cube; as measured using sky lines in the b3000 grating
    dlambda_A_ssp = 0.30  # Gonzalez-Delgado spectra_linear have a constant spectral sampling of 0.3 A.
    # Assuming that sigma = dlambda_A_ssp.
    FWHM_ssp_A = 2 * np.sqrt(2 * np.log(2)) * dlambda_A_ssp
    FWHM_LSF_A = np.sqrt(FWHM_inst_A**2 - FWHM_ssp_A**2)
    sigma_LSF_A = FWHM_LSF_A / (2 * np.sqrt(2 * np.log(2)))

    # WiFeS wavelength grid ("COMB" setting)
    N_lambda_wifes = 4520
    lambda_start_wifes_A = 3500.0
    dlambda_wifes_A = 0.7746262160168323
    lambda_vals_wifes_A = np.arange(N_lambda_wifes) * dlambda_wifes_A + lambda_start_wifes_A

    oversample_factor = 4
    lambda_vals_wifes_oversampled_A = np.arange(N_lambda_wifes * oversample_factor) * dlambda_wifes_A / oversample_factor + lambda_start_wifes_A

    # Compute the velocity scale ("velscale") parameter from the WiFeS wavelength sampling
    _, _, velscale_oversampled =\
            util.log_rebin(np.array([lambda_vals_wifes_oversampled_A[0], lambda_vals_wifes_oversampled_A[-1]]),
                           np.zeros(N_lambda_wifes * oversample_factor))

    ###########################################################################
    # Load stellar templates
    ###########################################################################
    stellar_templates_linear, lambda_vals_ssp_linear, metallicities, ages =\
        load_ssp_templates(isochrones, metals_to_use)
    # Note: stellar_templates_linear has shape (N_lambda, N_metallicities, N_ages)

    ###########################################################################
    # Create the mock spectrum
    ###########################################################################
    # Some settings for plotting
    fig_w = 12
    fig_h = 5

    # 1. Sum the templates by their weights to create a single spectrum
    spec_linear = np.nansum(np.nansum(sfh_mass_weighted[None, :, :] * stellar_templates_linear, axis=1), axis=1)

    # Plot to check
    if plotit:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_w, fig_h))
        ax.plot(lambda_vals_ssp_linear, spec_linear, color="black", label="Spectrum")
        for mm, aa in product(range(N_metallicities), range(N_ages)):
            w = sfh_mass_weighted[mm, aa]
            if w > 0:
                ax.plot(lambda_vals_ssp_linear, stellar_templates_linear[:, mm, aa] * w, 
                        label=f"t = {ages[aa] / 1e6:.2f} Myr, m = {metallicities[mm]:.4f}, w = {w:g}")
        ax.set_ylabel(f"$L$ (erg/s/$\AA$/M$_\odot$)")
        ax.set_xlabel(f"$\lambda$")
        # ax.legend()
        ax.autoscale(enable="True", axis="x", tight=True)

    ###########################################################################
    # 2. Logarithmically re-bin
    spec_log, lambda_vals_ssp_log, velscale_temp = util.log_rebin(
        np.array([lambda_vals_ssp_linear[0], lambda_vals_ssp_linear[-1]]),
        spec_linear, velscale=velscale_oversampled)

    ###########################################################################
    # 3a. Create the kernel corresponding to the LOSVD
    delta_lnlambda = np.diff(lambda_vals_ssp_log)[0]
    delta_lnlambda_vals = (np.arange(400) - 200) * delta_lnlambda

    # 3b. convert the x-axis to units of delta v (km/s) by multiplying by c (in km/s)
    c_kms = constants.c / 1e3
    delta_v_vals_kms = delta_lnlambda_vals * c_kms
    kernel_losvd = 1 / (np.sqrt(2 * np.pi) * sigma_star_kms) *\
             np.exp(- (delta_v_vals_kms**2) / (2 * sigma_star_kms**2))

    # Plot to check
    if plotit:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(delta_v_vals_kms, kernel_losvd)
        ax.axvline(0, color="black")
        ax.set_xlabel(r"$\Delta v$")
        ax.set_title("LOSVD kernel")

    ###########################################################################
    # 4. Convolve the LOSVD kernel with the mock spectrum
    spec_log_conv = convolve(spec_log, kernel_losvd, mode="same") / np.nansum(kernel_losvd)

    ###########################################################################
    # 5. Apply the redshift 
    lambda_vals_ssp_log_redshifted = lambda_vals_ssp_log + np.log(1 + z)

    ###########################################################################
    # 6. Interpolate to the WiFeS wavelength grid (corresponding to the COMB data cube) using a cubic spline
    cs = CubicSpline(np.exp(lambda_vals_ssp_log_redshifted), spec_log_conv)
    spec_wifes_conv = cs(lambda_vals_wifes_oversampled_A)

    ###########################################################################
    # 7. Convolve by the line spread function
    lambda_vals_lsf_oversampled_A = (np.arange(100) - 50) * dlambda_wifes_A / 4
    kernel_lsf = np.exp(- (lambda_vals_lsf_oversampled_A**2) / (2 * sigma_LSF_A**2))

    spec_wifes_conv_lsf = convolve(spec_wifes_conv, kernel_lsf, mode="same") / np.nansum(kernel_lsf)

    ###########################################################################
    # 8. Downsample to the WiFeS wavelength grid (corresponding to the COMB data cube)
    spec_wifes = np.nansum(spec_wifes_conv_lsf.reshape(-1, oversample_factor), axis=1) / oversample_factor

    ###########################################################################
    # 9. Add noise. 
    spec_err = spec_wifes / SNR
    noise = np.random.normal(loc=0, scale=spec_err)
    spec = spec_wifes + noise

    # Plot to check
    if plotit:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_w, fig_h))
        ax.errorbar(x=lambda_vals_wifes_A, y=spec, yerr=spec_err, color="red")

        ax.legend() 
        ax.set_title("Mock spectrum")
        ax.set_xlabel(f"$\lambda$")

        # Plot the SFH
        plot_sfh_mass_weighted(sfh_mass_weighted, ages, metallicities)
        plt.gcf().get_axes()[0].set_title("Input SFH")

    ###########################################################################
    # 10. Return.
    return spec, spec_err, lambda_vals_wifes_A

###############################################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###############################################################################
if __name__ == "__main__":
    ###########################################################################
    # Mock spectra options
    ###########################################################################
    isochrones = "Padova"  # Set of isochrones to use 
    
    ###########################################################################
    # GALAXY PROPERTIES
    ###########################################################################
    sigma_star_kms = 350   # LOS velocity dispersion, km/s
    z = 0.05               # Redshift 
    SNR = 25               # S/N ratio

    ###########################################################################
    # DEFINE THE SFH
    ###########################################################################
    # Idea 1: use a Gaussian kernel to smooth "delta-function"-like SFHs
    # Idea 2: are the templates logarithmically spaced in age? If so, could use e.g. every 2nd template 
    sfh_mass_weighted = np.zeros((3, 74))
    sfh_mass_weighted[1, 10] = 1e7
    sfh_mass_weighted[2, 60] = 1e10

    ###########################################################################
    # CREATE THE MOCK SPECTRUM
    ###########################################################################
    spec, spec_err, lambda_valsA = create_mock_spectrum(sfh_mass_weighted=sfh_mass_weighted,
                                          isochrones=isochrones,
                                          z=z, SNR=SNR, sigma_star_kms=sigma_star_kms)    
