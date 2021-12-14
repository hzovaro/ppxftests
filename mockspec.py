# Imports
import os 
import numpy as np

from scipy import constants
from scipy.signal import convolve
from scipy.interpolate import CubicSpline

from itertools import product

import ppxf.ppxf_util as util

from cosmocalc import get_dist
from ppxf_plot import plot_sfh_mass_weighted

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

###############################################################################
# Paths 
ssp_template_path = "/home/u5708159/python/Modules/ppxftests/SSP_templates"

N_AGES_PADOVA = 74
N_MET_PADOVA = 3
N_AGES_GENEVA = 46
N_MET_GENEVA = 5

###############################################################################
def create_mock_spectrum(sfh_mass_weighted, isochrones):

    assert isochrones == "Padova" or isochrones == "Geneva",\
        "isochrones must be either Padova or Geneva!"
    N_metallicities = N_MET_PADOVA if isochrones == "Padova" else N_MET_GENEVA
    N_ages = N_AGES_PADOVA if isochrones == "Padova" else N_AGES_GENEVA
    assert sfh_mass_weighted.shape == (N_metallicities, N_ages),\
        f"sfh_mass_weighted must have dimensions ({N_metallicities}, {N_ages})!"

    ###########################################################################
    # Instrument properties
    ###########################################################################
    # Compute the width of the LSF kernel we need to apply to the templates
    FWHM_inst_A = 1.4      # for the WiFeS COMB cube; as measured using sky lines in the b3000 grating
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
    # Load the templates 
    ###########################################################################
    # List of template names - one for each metallicity
    ssp_template_fnames =\
        [os.path.join(ssp_template_path, f"SSP{isochrones}", f) for f in os.listdir(os.path.join(ssp_template_path, f"SSP{isochrones}")) if f.endswith(".npz")]

    ###########################################################################
    # Determine how many different templates there are (i.e. N_ages x N_metallicities)
    metallicities = []
    ages = []
    for ssp_template_fname in ssp_template_fnames:
        f = np.load(os.path.join(ssp_template_path, ssp_template_fname))
        metallicities.append(f["metallicity"].item())
        ages = f["ages"] if ages == [] else ages
        lambda_vals_ssp_linear = f["lambda_vals_A"]

    # Template dimensions
    N_ages = len(ages)
    N_metallicities = len(metallicities)
    N_lambda = len(lambda_vals_ssp_linear)

    ###########################################################################
    # Create a big 3D array to hold the spectra
    spec_arr_linear = np.zeros((N_metallicities, N_ages, N_lambda))

    for mm, ssp_template_fname in enumerate(ssp_template_fnames):
        f = np.load(os.path.join(ssp_template_path, ssp_template_fname))
        
        # Get the spectra & wavelength values
        spectra_ssp_linear = f["L_vals"]
        lambda_vals_ssp_linear = f["lambda_vals_A"]

        # Store in the big array 
        spec_arr_linear[mm, :, :] = spectra_ssp_linear.T

    ###########################################################################
    # Create the mock spectrum
    ###########################################################################
    # Some settings for plotting
    fig_w = 12
    fig_h = 5
    lambda_1 = 5800
    lambda_2 = 6100

    # 1. Sum the templates by their weights to create a single spectrum
    spec_linear = np.nansum(np.nansum(sfh_mass_weighted[:, :, None] * spec_arr_linear, axis=0), axis=0)

    # Plot to check
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_w, fig_h))
    ax.plot(lambda_vals_ssp_linear, spec_linear, color="black", label="Spectrum")
    for mm, aa in product(range(N_metallicities), range(N_ages)):
        w = sfh_mass_weighted[mm, aa]
        if w > 0:
            ax.plot(lambda_vals_ssp_linear, spec_arr_linear[mm, aa, :] * w, 
                    label=f"t = {ages[aa] / 1e6:.2f} Myr, m = {metallicities[mm]:.4f}, w = {w:g}")
    ax.set_ylabel(f"$L$ (erg/s/$\AA$/M$_\odot$)")
    ax.set_xlabel(f"$\lambda$")
    ax.legend()
    ax.autoscale(enable="True", axis="x", tight=True)

    ###########################################################################
    # 2. Logarithmically re-bin
    spec_log, lambda_vals_ssp_log, velscale_temp = util.log_rebin(
        np.array([lambda_vals_ssp_linear[0], lambda_vals_ssp_linear[-1]]),
        spec_linear, velscale=velscale_oversampled)

    # Plot to check
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_w, fig_h))
    ax.plot(lambda_vals_ssp_linear, spec_linear, color="black", label="Normalised, linear spectrum")
    ax.plot(np.exp(lambda_vals_ssp_log), spec_log, color="red", label="Normalised, logarithmically-binned spectrum")

    ax.set_ylabel(f"$L$ + offset (normalised)")
    ax.set_xlabel(f"$\lambda$")
    ax.legend()
    ax.autoscale(enable="True", axis="x", tight=True)

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
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(delta_v_vals_kms, kernel_losvd)
    ax.axvline(0, color="black")
    ax.set_xlabel(r"$\Delta v$")

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(delta_lnlambda_vals, kernel_losvd)
    ax.axvline(0, color="black")
    ax.set_xlabel(r"$\Delta ln \lambda $")

    ###########################################################################
    # 4. Convolve the LOSVD kernel with the mock spectrum
    spec_log_conv = convolve(spec_log, kernel_losvd, mode="same") / np.nansum(kernel_losvd)

    # Plot to check
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_w, fig_h))
    ax.plot(np.exp(lambda_vals_ssp_log), spec_log, color="black", label="Before convolution with LOSV")
    ax.plot(np.exp(lambda_vals_ssp_log), spec_log_conv, color="red", label="After convolution with LOSVD")

    ax.set_ylabel(f"$L$")
    ax.set_xlabel(f"$\lambda$")
    ax.legend()
    ax.autoscale(enable="True", axis="x", tight=True)

    ###########################################################################
    # 5. Apply the redshift 
    lambda_vals_ssp_log_redshifted = lambda_vals_ssp_log + np.log(1 + z)

    # Plot to check
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_w, fig_h))
    ax.plot(np.exp(lambda_vals_ssp_log), spec_log_conv, color="black", label="Before redshifting")
    ax.plot(np.exp(lambda_vals_ssp_log_redshifted), spec_log_conv, color="red", label="After redshifting")
        
    ax.set_ylabel(f"$L$")
    ax.set_xlabel(f"$\lambda$")
    ax.legend()
    ax.autoscale(enable="True", axis="x", tight=True)

    ###########################################################################
    # 6. Interpolate to the WiFeS wavelength grid (corresponding to the COMB data cube) using a cubic spline
    cs = CubicSpline(np.exp(lambda_vals_ssp_log_redshifted), spec_log_conv)
    spec_wifes_conv = cs(lambda_vals_wifes_oversampled_A)

    # Plot to check
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_w, fig_h))
    ax.step(np.exp(lambda_vals_ssp_log_redshifted), spec_log_conv, color="black", label="Before interpolation", where="mid")
    ax.step(lambda_vals_wifes_oversampled_A, spec_wifes_conv, color="red", label="Interpolated to WiFeS wavelength grid", where="mid")

    ax.legend() 
    ax.set_xlabel(f"$\lambda$")


    ###########################################################################
    # 7. Convolve by the line spread function
    lambda_vals_lsf_oversampled_A = (np.arange(100) - 50) * dlambda_wifes_A / 4
    kernel_lsf = np.exp(- (lambda_vals_lsf_oversampled_A**2) / (2 * sigma_LSF_A**2))

    spec_wifes_conv_lsf = convolve(spec_wifes_conv, kernel_lsf, mode="same") / np.nansum(kernel_lsf)

    # Plot to check
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_w, fig_h))
    ax.step(lambda_vals_wifes_oversampled_A, spec_wifes_conv, color="black", label="Before convolution with LSF", where="mid")
    ax.step(lambda_vals_wifes_oversampled_A, spec_wifes_conv_lsf, color="red", label="After convolution with LSF", where="mid")

    ax.legend() 
    ax.set_xlabel(f"$\lambda$")

    ###########################################################################
    # 8. Downsample to the WiFeS wavelength grid (corresponding to the COMB data cube)
    spec_wifes = np.nansum(spec_wifes_conv_lsf.reshape(-1, oversample_factor), axis=1) / oversample_factor

    # Plot to check
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_w, fig_h))
    ax.step(lambda_vals_wifes_oversampled_A, spec_wifes_conv, color="black", label="Before downsampling", where="mid")
    ax.step(lambda_vals_wifes_A, spec_wifes, color="red", label="After downsampling", where="mid")

    ax.legend() 
    ax.set_xlabel(f"$\lambda$")

    ###########################################################################
    # Convert to units of erg/s/cm2/A
    D_A_Mpc, D_L_Mpc = get_dist(z, H0=70.0, WM=0.3)
    D_L_cm = D_L_Mpc * 1e6 * 3.086e18
    spec_wifes_flambda = spec_wifes * 1 / (4 * np.pi * D_L_cm**2)

    ###########################################################################
    # 9. Add noise. 
    spec_wifes_flambda_err = spec_wifes_flambda / SNR
    noise = np.random.normal(loc=0, scale=spec_wifes_flambda_err)
    spec_wifes_noisy = spec_wifes_flambda + noise

    # Plot to check
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_w, fig_h))
    ax.step(lambda_vals_wifes_A, spec_wifes_flambda, color="black", label="Before noise", where="mid")
    ax.step(lambda_vals_wifes_A, spec_wifes_noisy + 2e-17, color="red", label="After noise", where="mid")

    ax.legend() 
    ax.set_xlabel(f"$\lambda$")

    # Plot the SFH
    plot_sfh_mass_weighted(sfh_mass_weighted, ages, metallicities)
    plt.gcf().get_axes()[0].set_title("Input SFH")

    ###########################################################################
    # 10. Return.
    return spec_wifes_noisy, spec_wifes_flambda_err

###############################################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###############################################################################
if __name__ == "__main__":
    ###########################################################################
    # Mock spectra options
    ###########################################################################
    isochrones = "Padova"  # Set of isochrones to use 
    N_metallicities = N_MET_PADOVA if isochrones == "Padova" else N_MET_GENEVA
    N_ages = N_AGES_PADOVA if isochrones == "Padova" else N_AGES_GENEVA
    
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
    sfh_mass_weighted = np.zeros((N_metallicities, N_ages))
    sfh_mass_weighted[1, 10] = 1e7
    sfh_mass_weighted[2, 60] = 1e10

    ###########################################################################
    # CREATE THE MOCK SPECTRUM
    ###########################################################################
    spec, spec_err = create_mock_spectrum(sfh_mass_weighted, isochrones)    
