# imports 
import os
import numpy as np
from itertools import product

from ppxftests.run_ppxf import run_ppxf
from ppxftests.mockspec import create_mock_spectrum, get_age_and_metallicity_values
from ppxftests.ppxf_plot import plot_sfh_mass_weighted

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")
fig_path = "/priv/meggs3/u5708159/ppxftests/figs/"

# from IPython.core.debugger import Tracer

"""
    Accurately recovering the smoothness of the SFH: try SFHs with varying 
    degrees of smoothness. Using the local minimum approach to find the optimal 
    value for regul, does ppxf accurately return the degree of smoothness?
    
    Testing plan: make 2x "base" SFHs - one with only an old population, and 
    one with a young & old component. Run ppxf with 4 different degrees of 
    smoothness - with starburst durations of $10^6$, $10^7$ and $10^8$ yr 
    (where we set the duration to the FWHM). See what happens.

    Based on this result, adopt a standard starburst duration (adopt whatever 
    value is able to be accurately recovered by ppxf) to use for future 
    experiments.

"""
# Running everything in a big ol' loop
for age, FWHM_age in product([1e6, 10e6, 100e6, 1e9], [1e6, 1e7, 1e8, 1e9]):
    plt.close("all")
    ################################################################################################################
    # INPUT PARAMETERS
    ################################################################################################################
    # SFH properties
    oversample_factor = 24
    sigma_age = FWHM_age / (2 * np.sqrt(2 * np.log(2)))

    # age_1 = 10e9    # Age of primary stellar pop., in yr
    met = 0.006   # Metallicity of primary stellar pop., in solar units
    mass = 1e10   # Total stellar mass
    sigma_met = 0.001  # Width in metallicity distribution (Gaussian sigma)
    # age_2 = 10e6    # Age of primary stellar pop., in yr
    # met_2 = 0.019   # Metallicity of primary stellar pop., in solar units
    # mass_2 = 1e7    # Total stellar mass
    # sigma_met_2 = 0.001  # Width in metallicity distribution (Gaussian sigma)
    sfh_components = [[age, met, FWHM_age, sigma_met, mass]] 
                      # [age_2, met_2, FWHM_age, sigma_met_2, mass_2]]

    # Other parameters
    isochrones = "Padova"
    SNR = 100
    z = 0.05
    sigma_star_kms = 250

    # Figure filename
    fname_str = os.path.join(fig_path, "regularisation", f"age={age:g}_fwhm={FWHM_age:.2g}")

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

    ###########################################################################
    # CREATE THE MOCK SPECTRUM
    ###########################################################################
    spec, spec_err, lambda_vals_A = create_mock_spectrum(
        sfh_mass_weighted=sfh_input,
        isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
        plotit=True)    

    ###########################################################################
    # RUN PPXF
    ###########################################################################
    pp, sfh_fit = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
                           z=z, ngascomponents=1,
                           auto_adjust_regul=True,
                           isochrones="Padova", tie_balmer=True,
                           delta_regul_min=1, regul_max=5e4,
                           plotit=True, savefigs=True,
                           fname_str=fname_str)

    ###########################################################################
    # COMPARE THE INPUT AND OUTPUT
    ###########################################################################
    # side-by-side comparison of the SFHs, plus residual map
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
    # fig.suptitle(f"Input smoothing parameter: $\tau = 10^{int(np.log10(fwhm_age))}$ yr")
    plot_sfh_mass_weighted(sfh_input, ages, metallicities, ax=axs[0])
    axs[0].set_title("Input SFH")
    axs[0].set_xlabel("")
    plot_sfh_mass_weighted(sfh_fit, ages, metallicities, ax=axs[1])
    axs[1].set_title("ppxf best-fit SFH")
    axs[1].set_xlabel("")

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
    axs[2].set_xticks(range(len(ages)))
    axs[2].set_xlabel("Age (Myr)")
    axs[2].set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")
    axs[2].set_title("Difference")

    [ax.set_xticklabels([]) for ax in axs[:-1]]

    fig.savefig(f"{fname_str}_input_output_comparison.pdf", format="pdf", bbox_inches="tight")
