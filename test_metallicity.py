# imports 
import os
import numpy as np
from time import time

from ppxftests.run_ppxf import run_ppxf
from ppxftests.mockspec import create_mock_spectrum, get_age_and_metallicity_values
from ppxftests.ppxf_plot import plot_sfh_mass_weighted

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")
fig_path = "/priv/meggs3/u5708159/ppxftests/figs/"

# from IPython.core.debugger import Tracer

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
fname_str = os.path.join(fig_path, "metallicity", f"age={age:g}_fwhm={FWHM_age:.2g}")

################################################################################################################
# COMPUTE THE SFH
################################################################################################################
# Oversampled arrays in age and metallicity
ages, metallicities = get_age_and_metallicity_values(isochrones)
N_ages = len(ages)
N_metallicities = len(metallicities)

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

sfh_input = bin2d(sfh_oversampled, oversample_factor)
bbox = axs[ii].get_position()
cax = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.03, bbox.height])
m = axs[ii].imshow(np.log10(sfh_input),
                   vmin=0,
                   extent=(np.log10(ages[0]), np.log10(ages[-1]), metallicities[0], metallicities[-1]), 
                   aspect="auto")
plt.colorbar(mappable=m, cax=cax)
cax.set_ylabel(r"Mass ($M_\odot$)")
axs[ii].set_title(f"Total SFH (downsampled)")

###############################################################################
# Run N times with different noise realisations to look at the effects of noise
###############################################################################
pp_list = []
sfh_fit_list = []
for ii in range(10):
    ###########################################################################
    # CREATE THE MOCK SPECTRUM WITH RANDOM NOISE
    ###########################################################################
    spec, spec_err, lambda_vals_A = create_mock_spectrum(
        sfh_mass_weighted=sfh_input,
        isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
        plotit=False)    

    ###########################################################################
    # RUN PPXF
    ###########################################################################
    t = time()
    pp, sfh_fit = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                           z=z, ngascomponents=1,
                           auto_adjust_regul=True,
                           isochrones="Padova", metals_to_use=["004"], fit_gas=False,
                           tie_balmer=True,
                           delta_regul_min=1, regul_max=5e4, delta_delta_chi2_min=1,
                           plotit=True, savefigs=True, interactive_mode=False,
                           fname_str=fname_str)
    print(f"Total time in run_ppxf: {time() - t:.2f} seconds")

    pp_list.append(pp)
    sfh_fit_list.append(sfh_fit)

    ###########################################################################
    # COMPARE THE INPUT AND OUTPUT
    ###########################################################################
    # side-by-side comparison of the SFHs, plus residual map
    fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(12, 12), sharex=True)
    fig.subplots_adjust(hspace=0)
    # fig.suptitle(f"Input smoothing parameter: $\tau = 10^{int(np.log10(fwhm_age))}$ yr")
    plot_sfh_mass_weighted(np.expand_dims(sfh_input[0], axis=0), ages, metallicities[:1], ax=axs[0])
    axs[0].set_title("Input SFH")
    axs[0].set_xlabel("")
    axs[0].set_xticklabels([])
    plot_sfh_mass_weighted(sfh_fit, ages, metallicities[:1], ax=axs[1])
    axs[1].set_title("ppxf best-fit SFH")
    axs[1].set_xlabel("")
    axs[1].set_xticklabels([])

    # Plot the residual
    bbox = axs[2].get_position()
    cax = fig.add_axes([bbox.x0 + bbox.width, bbox.y0, 0.025, bbox.height])

    # Plot the SFH
    delta_sfh = sfh_input - sfh_fit
    m = axs[2].imshow(delta_sfh, cmap="coolwarm", 
                  origin="lower", aspect="auto",
                  vmin=-np.abs(np.nanmax(delta_sfh)), vmax=np.abs(np.nanmax(delta_sfh)))
    fig.colorbar(m, cax=cax)

    # Decorations
    axs[2].set_yticks(range(len(metallicities)))
    axs[2].set_yticklabels(["{:.3f}".format(met / 0.02) for met in metallicities])
    axs[2].set_ylabel(r"Metallicity ($Z_\odot$)")
    cax.set_ylabel(r"Residual ($\rm M_\odot$)")
    axs[2].set_title("Difference")

    # Plot the 2D SFHs
    axs[3].step(range(len(sfh_input[0])), sfh_input[0], color="black", where="mid", label="Input SFH")
    axs[3].step(range(len(sfh_fit[0])), sfh_fit[0], color="red", where="mid", label="ppxf fit")
    axs[3].legend()
    # axs[3].set_yscale("log")
    axs[3].set_ylabel(r"$M_* (\rm M_\odot)$")
    axs[3].set_ylim([10^0, None])
    axs[3].set_xticks(range(len(ages)))
    axs[3].set_xlabel("Age (Myr)")
    axs[3].set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")
    axs[3].autoscale(axis="x", enable=True, tight=True)
    axs[3].set_title("1D SFH")

    # Plot the 2D SFHs
    axs[4].step(range(len(sfh_input[0])), sfh_input[0], color="black", where="mid", label="Input SFH")
    axs[4].step(range(len(sfh_fit[0])), sfh_fit[0], color="red", where="mid", label="ppxf fit")
    axs[4].legend()
    axs[4].set_yscale("log")
    axs[4].set_ylabel(r"$M_* (\rm M_\odot)$")
    axs[4].set_ylim([10^0, None])
    axs[4].set_xticks(range(len(ages)))
    axs[4].set_xlabel("Age (Myr)")
    axs[4].set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")
    axs[4].autoscale(axis="x", enable=True, tight=True)
    axs[4].set_title("1D SFH")

    [ax.set_xticklabels([]) for ax in axs[:-1]]

    # Save 
    # fig.savefig(f"{fname_str}_input_output_comparison.pdf", format="pdf", bbox_inches="tight")
