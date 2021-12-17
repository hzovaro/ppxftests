# imports 
import os
import numpy as np
from time import time
from tqdm import tqdm
from itertools import product
from scipy import constants, ndimage

import ppxf.ppxf_util as util

from ppxftests.run_ppxf import run_ppxf
from ppxftests.ssputils import load_ssp_templates
from ppxftests.mockspec import create_mock_spectrum
from ppxftests.ppxf_plot import plot_sfh_mass_weighted

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")
fig_path = "/priv/meggs3/u5708159/ppxftests/figs/"

from IPython.core.debugger import Tracer

"""
Reducing the number of templates: 
by how much can we degrade the temporal & metallicity sampling of the template 
grid & still accurately capture the SFH? (i.e., see if the fitting process can 
be sped up by using fewer templates)
For now, see what happens when we apply 1x metallicity templates in ppxf 
to an input spectrum that was created using 3x metallicity templates.
"""

################################################################################################################
# INPUT PARAMETERS
################################################################################################################
# SFH properties
oversample_factor = 24
FWHM_age = 1e6  
age = 10e6  # Age of primary stellar pop., in yr
sigma_age = FWHM_age / (2 * np.sqrt(2 * np.log(2)))
 
met = 0.006   # Metallicity of primary stellar pop., in solar units
mass = 1e10   # Total stellar mass
sigma_met = 0.00001  # Width in metallicity distribution (Gaussian sigma)
sfh_components = [[age, met, FWHM_age, sigma_met, mass]] 

# Other parameters
isochrones = "Padova"
SNR = 100
z = 0.05
sigma_star_kms = 150

# Figure filename
fname_str = os.path.join(fig_path, "noise", f"age={age:g}_fwhm={FWHM_age:.2g}_SNR={SNR}")

################################################################################################################
# COMPUTE THE SFH
################################################################################################################
# Oversampled arrays in age and metallicity
stellar_templates_linear, lambda_vals_ssp_linear, metallicities, ages = load_ssp_templates(isochrones)
N_ages = len(ages)
N_metallicities = len(metallicities)
N_lambda_ssp_linear = len(lambda_vals_ssp_linear)

###########################################################################
# DEFINING LIGHT-WEIGHTED TEMPLATES 
"""
We need:
    - normalisation factors for every template AFTER they have been convolved 
        with the differential LSF and log-rebinned 
    - the median luminosity of the input spectrum (we can choose this!!!)
"""

#### PASTED FROM RUN_PPXF.PY ######
# Reshape stellar templates so that its dimensions are (N_lambda, N_ages * N_metallicities)
# stellar_templates_linear =\
    # stellar_templates_linear.reshape((N_lambda_ssp_linear, N_ages * N_metallicities))

# WiFeS wavelength grid ("COMB" setting)
FWHM_WIFES_INST_A = 1.4
FWHM_inst_A = FWHM_WIFES_INST_A
N_lambda_wifes = 4520
lambda_start_wifes_A = 3500.0
dlambda_wifes_A = 0.7746262160168323
lambda_vals_wifes_A = np.arange(N_lambda_wifes) * dlambda_wifes_A + lambda_start_wifes_A

# Rebin to a log scale: need to do this to get velscale 
_, _, velscale = util.log_rebin(
    np.array([lambda_vals_wifes_A[0], lambda_vals_wifes_A[-1]]), np.zeros(lambda_vals_wifes_A.shape))

# Extract the wavelength range for the logarithmically rebinned templates
_, lambda_vals_ssp_log, _ = util.log_rebin(np.array(
    [lambda_vals_ssp_linear[0], lambda_vals_ssp_linear[-1]]),
    stellar_templates_linear[:, 0, 0], velscale=velscale)
N_lambda_ssp_log = len(lambda_vals_ssp_log)

# Create an array to store the logarithmically rebinned spectra in
stellar_templates_log = np.zeros((N_lambda_ssp_log, N_metallicities, N_ages))

# Convolve each SSP template to the instrumental resolution
dlambda_A_ssp = 0.30  # Gonzalez-Delgado spectra_linear have a constant spectral sampling of 0.3 A.
FWHM_ssp_A = 2 * np.sqrt(2 * np.log(2)) * dlambda_A_ssp
FWHM_diff_A = np.sqrt(FWHM_inst_A**2 - FWHM_ssp_A**2)
sigma_diff_px = FWHM_diff_A / (2 * np.sqrt(2 * np.log(2))) / dlambda_A_ssp  # sigma_diff_px difference in pixels
stars_templates_linear_conv = np.zeros(stellar_templates_linear.shape)
for aa, mm in product(range(N_ages), range(N_metallicities)):
    # Convolve
    stars_templates_linear_conv[:, mm, aa] =\
        ndimage.gaussian_filter1d(stellar_templates_linear[:, mm, aa], sigma_diff_px)

    # Logarithmically rebin
    spec_ssp_log, lambda_vals_ssp_log, velscale_temp =\
        util.log_rebin(np.array(
            [lambda_vals_ssp_linear[0], lambda_vals_ssp_linear[-1]]),
            stars_templates_linear_conv[:, mm, aa],
            velscale=velscale)
    stellar_templates_log[:, mm, aa] = spec_ssp_log

# Normalise
stellar_template_norms = np.nanmedian(stellar_templates_log, axis=0)

#### END PASTE FROM RUN_PPXF.PY ######

# Simple Gaussian SFH
x0 = 20
y0 = 1.5
sigma_x = 10
sigma_y = 1
xx, yy = np.meshgrid(range(N_ages), range(N_metallicities))
sfh_lw = np.exp(- (xx - x0)**2 / (2 * sigma_x**2)) *\
         np.exp(- (yy - y0)**2 / (2 * sigma_y**2))

# Normalise 
sfh_lw /= np.sum(sfh_lw)

# Convert to mass weights 
L_median = 1e40
sfh_mw = sfh_lw / stellar_template_norms * L_median

plt.imshow(sfh_mw, aspect="auto")

sfh_input_mass_weighted = sfh_mw

"""
###########################################################################
ages_oversampled = np.exp(np.linspace(np.log(ages[0]), 
                                      np.log(ages[-1]), 
                                      N_ages * oversample_factor))
metallicities_oversampled = np.exp(np.linspace(np.log(metallicities[0]), 
                                      np.log(metallicities[-1]), 
                                      N_metallicities * oversample_factor))
xx, yy = np.meshgrid(ages_oversampled, metallicities_oversampled)
sfh_oversampled = np.zeros(xx.shape)

# Compute the SFH
fig, axs = plt.subplots(nrows=len(sfh_components) + 2, ncols=1, figsize=(8, 5 * (len(sfh_components) + 2)))
ii = 0
for age, met, sigma_age, sigma_met, mass in sfh_components:

    # Find the indices in the age & metallicity vectors corresponding to this age, metallicity
    age_idx = np.nanargmin(np.abs(ages_oversampled - age))
    met_idx = np.nanargmin(np.abs(metallicities_oversampled - met))

    # Define a "smooth" SFH
    sfh_component = np.exp(- (xx - age)**2 / (2 * sigma_age**2)) *\
                    np.exp(- (yy - met)**2 / (2 * sigma_met**2))
    sfh_component /= np.nansum(sfh_component)
    sfh_component *= mass
    
    # Add to the overall SFH
    sfh_oversampled += sfh_component
    
    # Plot
    bbox = axs[ii].get_position()
    cax = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.03, bbox.height])
    m = axs[ii].imshow(sfh_component,
                       extent=(np.log10(ages[0]), np.log10(ages[-1]), metallicities[0], metallicities[-1]), 
                       aspect="auto")
    plt.colorbar(mappable=m, cax=cax)
    cax.set_ylabel(r"Mass ($M_\odot$)")
    axs[ii].set_title(f"SFH component {ii + 1}")
    ii += 1

# Plot to compare 
bbox = axs[ii].get_position()
cax = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.03, bbox.height])
m = axs[ii].imshow(sfh_oversampled,
                   extent=(np.log10(ages[0]), np.log10(ages[-1]), metallicities[0], metallicities[-1]), 
                   aspect="auto")
plt.colorbar(mappable=m, cax=cax)
cax.set_ylabel(r"Mass ($M_\odot$)")
axs[ii].set_title(f"Total SFH (oversampled)")
ii += 1

# Downsample & plot again
def bin2d(a, K):
    m_bins = a.shape[0] // K
    n_bins = a.shape[1] // K
    return a.reshape(m_bins, K, n_bins, K).sum(3).sum(1)

sfh_input_mass_weighted = bin2d(sfh_oversampled, oversample_factor)
bbox = axs[ii].get_position()
cax = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.03, bbox.height])
m = axs[ii].imshow(np.log10(sfh_input_mass_weighted),
                   vmin=0,
                   extent=(np.log10(ages[0]), np.log10(ages[-1]), metallicities[0], metallicities[-1]), 
                   aspect="auto")
plt.colorbar(mappable=m, cax=cax)
cax.set_ylabel(r"Mass ($M_\odot$)")
axs[ii].set_title(f"Total SFH (downsampled)")
"""

###############################################################################
# Run N times with different noise realisations to look at the effects of noise
###############################################################################
pp_list = []
sfh_fit_mass_weighted_list = []
sfh_fit_light_weighted_list = []
iters = 100
for ii in tqdm(range(iters)):
    ###########################################################################
    # CREATE THE MOCK SPECTRUM WITH RANDOM NOISE
    ###########################################################################
    spec, spec_err, lambda_vals_A = create_mock_spectrum(
        sfh_mass_weighted=sfh_input_mass_weighted,
        isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
        plotit=True)  

    ###########################################################################
    # RUN PPXF
    ###########################################################################
    t = time()
    pp = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                           z=z, ngascomponents=1,
                           auto_adjust_regul=True,
                           isochrones="Padova",
                           fit_gas=False, tie_balmer=True,
                           delta_regul_min=1, regul_max=5e4, delta_delta_chi2_min=1,
                           plotit=False, savefigs=False, interactive_mode=False,
                           fname_str=fname_str)
    print(f"Total time in run_ppxf: {time() - t:.2f} seconds")

    pp_list.append(pp)
    sfh_fit_mass_weighted_list.append(pp.weights_mass_weighted)
    sfh_fit_light_weighted_list.append(pp.weights_light_weighted)

    # Compute the light-weighted stellar template weights so we can compare to the ppxf weights directly
    sfh_input_light_weighted = sfh_input_mass_weighted / pp.norm * pp.stellar_template_norms
    # sfh_input_light_weighted /= np.nansum(sfh_input_light_weighted)

    ###########################################################################
    # COMPARE THE INPUT AND OUTPUT
    ###########################################################################
    plt.close("all")
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 11))
    fig.subplots_adjust(hspace=0.3)
    # Mass-weighted 
    for ax in [axs[0][0], axs[1][0]]:
        ax.step(range(len(sfh_input_mass_weighted[0])),
                sfh_input_mass_weighted[0],
                color="black", where="mid", label="Input SFH")
        for jj in range(ii + 1):
            ax.step(range(len(sfh_fit_mass_weighted_list[jj][0])),
                    sfh_fit_mass_weighted_list[jj][0],
                    color="red", alpha=0.1, where="mid", label="ppxf fits" if jj == 0 else None)
        ax.legend()
        ax.set_xticks(range(len(ages)))
        ax.set_xlabel("Age (Myr)")
        ax.set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")
        ax.autoscale(axis="x", enable=True, tight=True)
        ax.set_ylim([1, None])
        ax.set_ylabel("Template weight (mass-weighted)")
    axs[1][0].set_yscale("log")

    # light-weighted 
    for ax in [axs[0][1], axs[1][1]]:
        ax.step(range(len(sfh_input_light_weighted[0])),
                sfh_input_light_weighted[0],
                color="black", where="mid", label="Input SFH")
        for jj in range(ii + 1):
            ax.step(range(len(sfh_fit_light_weighted_list[jj][0])),
                    sfh_fit_light_weighted_list[jj][0],
                    color="red", alpha=0.1, where="mid", label="ppxf fits" if jj == 0 else None)
        ax.legend()
        ax.set_xticks(range(len(ages)))
        ax.set_xlabel("Age (Myr)")
        ax.set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")
        ax.autoscale(axis="x", enable=True, tight=True)
        ax.set_ylabel("Template weight (light-weighted)")
    axs[1][1].set_yscale("log")

    Tracer()()

    # fig.savefig(f"{fname_str}_noise_test_niters={ii}.pdf", bbox_inches="tight")
