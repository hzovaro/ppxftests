# Imports
import os 
import numpy as np
from numpy.random import RandomState

from scipy import constants
from scipy.signal import convolve
from scipy.interpolate import CubicSpline

import extinction

from itertools import product

import ppxf.ppxf_util as util

from cosmocalc import get_dist
# from spaxelsleuth.plotting.plotgrids import load_HII_grid, load_AGN_grid

from ppxftests.ppxf_plot import plot_sfh_mass_weighted
from ppxftests.ssputils import load_ssp_templates, get_bin_edges_and_widths

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.ion()
plt.close("all")

FWHM_WIFES_INST_A = 1.4
VELSCALE_WIFES = 45.9896038

###############################################################################
def get_wavelength_from_velocity(lambda_rest, v, units):
    assert units == 'm/s' or units == 'km/s', "units must be m/s or km/s!"
    assert isinstance(v, (np.floating, float)) or type(v) == int \
        or type(v) == np.ndarray, "v must be an int, float or a Numpy array!"
    if units == 'm/s':
        v_m_s = v
    elif units == 'km/s':
        v_m_s = v * 1e3
    lambda_obs = lambda_rest * np.sqrt((1 + v_m_s / constants.c) /
                                       (1 - v_m_s / constants.c))
    return lambda_obs

###############################################################################
def create_mock_spectrum(sfh_mass_weighted, isochrones, sigma_star_kms, z, SNR,
                         ngascomponents=0, sigma_gas_kms=[], v_gas_kms=[], 
                         eline_model=[], L_Ha_erg_s=[], 
                         agn_continuum=False, alpha_nu=2.0, x_AGN=0, lambda_norm_A=4020,
                         A_V=0.0,
                         seed=0,
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

    ngascomponents          Number of kinematic components in the emission 
                            lines. Set to 0 if no emission lines are to be 
                            included (default).
    sigma_gas_kms           List of velocity dispersions for each kinematic 
                            component in the emission lines. Must have length
                            equal to ngascomponents.
    v_gas_kms               List of radial velocities for each kinematic 
                            component in the emission lines. Must have length
                            equal to ngascomponents.
    eline_model             Photoionisation source to assume for the emission
                            lines. Options are "HII" for a typical high-Z
                            star-forming galaxy and "AGN" for a typical 
                            high-Z NLR. Must have length equal to 
                            ngascomponents.
    L_Ha_erg_s              Total Halpha emission line luminosity for each 
                            component in the emission lines. Must have length
                            equal to ngascomponents.
    agn_continuum           Whether to include an AGN power law
    alpha_nu                Power law exponent in frequency, defined by 
                                        F_nu \propto nu**(- alpha_nu)
                            The corresponding exponent in wavelength space is 
                            given by 
                                        alpha_lambda = 2 - alpha_nu
    x_AGN                   Fraction of continuum at lambda_norm_A from the 
                            power-law AGN continuum
    lambda_norm_A           Normalisation wavelength for the AGN continuum
    A_V                     Extinction to assume for the stellar component.
    seed                    Seed for the random number generator for making
                            noise.
    Returns:
    spec, spec_err          mock spectrum and corresponding 1-sigma errors, in
                            units of erg/s. 

    """
    assert isochrones == "Geneva" or isochrones == "Padova",\
        "isochrones must be either Padova or Geneva!"
    if isochrones == "Padova":
        if metals_to_use is None:
            metals_to_use = ["004", "008", "019"]
        else:
            for m in metals_to_use:
                assert m in ["004", "008", "019"],\
                    f"Metallicity {m} for the {isochrones} isochrones not found!"
    elif isochrones == "Geneva":
        if metals_to_use is None:
            metals_to_use = ["001", "004", "008", "020", "040"]
        else:
            for m in metals_to_use:
                assert m in ["001", "004", "008", "020", "040"],\
                    f"Metallicity {m} for the {isochrones} isochrones not found!"
    N_metallicities, N_ages = sfh_mass_weighted.shape
    assert N_metallicities == len(metals_to_use),\
        f"sfh_mass_weighted.shape[0] = {N_metallicities} but len(metals_to_use) = {len(metals_to_use)}!"

    if ngascomponents > 0:
        assert len(sigma_gas_kms) == ngascomponents,\
            "len(sigma_gas_kms) must be equal to ngascomponents!"
        assert len(v_gas_kms) == ngascomponents,\
            "len(v_gas_kms) must be equal to ngascomponents!"
        assert len(L_Ha_erg_s) == ngascomponents,\
            "len(L_Ha_erg_s) must be equal to ngascomponents!"
        assert len(eline_model) == ngascomponents,\
            "len(eline_model) must be equal to ngascomponents!"
        assert np.all([m in ["HII", "AGN", "BLR"] for m in eline_model]),\
            "entries in eline_model must be either 'HII', 'AGN' or 'BLR'!"
        
    if plotit:
        nrows = 5 if ngascomponents > 0 else 4
        fig, axs = plt.subplots(ncols=1, nrows=nrows, figsize=(7.5, 10))
        fig.subplots_adjust(hspace=0, left=0.1, right=0.9, bottom=0.05, top=0.95)
        ax_ctr = 0
        plot_norm = 1e39
        text_kwargs = {
             "x":0.02, "y":0.94, "ha": "left", "va":"top", 
            "bbox": dict(facecolor="lemonchiffon", edgecolor="k", boxstyle="round", alpha=0.9),
        }
        legend_kwargs = {
            "fontsize": 7,
            "loc": "upper right",
            "framealpha": 0.8,
        }

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
    # 1. Sum the templates by their weights to create a single spectrum
    spec_linear = np.nansum(np.nansum(sfh_mass_weighted[None, :, :] * stellar_templates_linear, axis=1), axis=1)

    # Plot to check
    if plotit:
        ax = axs[ax_ctr]
        ax.plot(lambda_vals_ssp_linear, spec_linear / plot_norm, color="black", label="Total spectrum")
        for mm, aa in product(range(N_metallicities), range(N_ages)):
            w = sfh_mass_weighted[mm, aa]
            if w > 0:
                ax.plot(lambda_vals_ssp_linear, stellar_templates_linear[:, mm, aa] * w / plot_norm, alpha=0.5, lw=0.5, color="grey")
        ax.set_ylabel(r"$L_{\lambda}(\lambda) \,\rm (10^{39}\,erg\,s^{-1}\,\AA^{-1}$)")
        ax.set_xlabel(r"Rest-frame wavelength ($\rm \AA$)")
        lines = [Line2D([0], [0], color="black", alpha=1.0),
                 Line2D([0], [0], color="black", alpha=0.5)]
        labels = ["Total spectrum", "Individual templates"]
        ax.legend(lines, labels, **legend_kwargs)
        ax.autoscale(enable="True", axis="x", tight=True)
        ax.text(s="Adding weighted templates", transform=ax.transAxes, **text_kwargs)
        ax_ctr += 1

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
    # if plotit:
    #     fig, ax = plt.subplots(nrows=1, ncols=1)
    #     ax.plot(delta_v_vals_kms, kernel_losvd)
    #     ax.axvline(0, color="black")
    #     ax.set_xlabel(r"$\Delta v$")
    #     ax.set_title("LOSVD kernel")

    ###########################################################################
    # 4a. Convolve the LOSVD kernel with the mock spectrum
    spec_log_conv = convolve(spec_log, kernel_losvd, mode="same") / np.nansum(kernel_losvd)

    if plotit:
        ax = axs[ax_ctr]
        ax.plot(np.exp(lambda_vals_ssp_log), spec_log / plot_norm, color="grey", lw=0.5, label="Before convolution with LOSVD")
        ax.plot(np.exp(lambda_vals_ssp_log), spec_log_conv / plot_norm, color="black", label="After convolution with LOSVD")
        ax.set_ylabel(r"$L_{\lambda}(\lambda) \,\rm (10^{39}\,erg\,s^{-1}\,\AA^{-1}$)")
        ax.set_xlabel(r"Rest-frame wavelength ($\rm \AA$)")
        ax.legend(**legend_kwargs)
        ax.autoscale(enable="True", axis="x", tight=True)
        ax.text(s="Convolution with LOSVD", transform=ax.transAxes, **text_kwargs)
        ax_ctr += 1

        # Add an extra axis 
        bbox = ax.get_position()
        ax_inset = fig.add_axes([bbox.x0 + 0.57, bbox.y0 + 0.03, bbox.width / 4, bbox.height / 3.])
        pts = (np.exp(lambda_vals_ssp_log) > 6562.8 - 50) & (np.exp(lambda_vals_ssp_log) < 6562.8 + 50)
        ax_inset.plot(np.exp(lambda_vals_ssp_log)[pts], spec_log[pts] / plot_norm, color="grey", lw=0.5, label="Before convolution with LOSVD")
        ax_inset.plot(np.exp(lambda_vals_ssp_log)[pts], spec_log_conv[pts] / plot_norm, color="black", label="After convolution with LOSVD")
        ax_inset.autoscale(enable="True", axis="x", tight=True)
        ax_inset.tick_params(axis="both", which="major", labelsize=7)

    ###########################################################################
    # 4b. Add emission lines, if needed
    if ngascomponents > 0:
        spec_log_lines = np.zeros(spec_log_conv.shape)
 
        eline_lambdas_A = {
            "NeV3347": 3346.79,
            "OIII3429": 3429.0,
            "OII3726": 3726.032,
            "OII3729": 3728.815,
            "NeIII3869": 3869.060,
            "HeI3889": 3889.0,
            "CaII H": 3969.0,
            "CaII K": 3934.0,
            "HEPSILON": 3970.072,
            "HDELTA": 4101.734, 
            "HGAMMA": 4340.464, 
            "HeI4471": 4471.479,
            "OIII4363": 4363.210, 
            "HBETA": 4861.325, 
            "OIII4959": 4958.911, 
            "OIII5007": 5006.843, 
            "HeI5876": 5875.624, 
            "OI6300": 6300.304, 
            "SIII6312": 6312.060,
            "OI6364": 6363.776,
            "NII6548": 6548.04, 
            "HALPHA": 6562.800, 
            "NII6583": 6583.460,
            "SII6716": 6716.440, 
            "SII6731": 6730.810,
        }

        for nn in range(ngascomponents):
            # Load a MAPPINGS grid
            if eline_model[nn] == "BLR":
                # For a broad line region, just add the Halpha, Hbeta and [OIII].
                # Reference for Ha/Hb ratios in BLRs: 
                #   https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.3570S/abstract
                # See section 4.2 for Halpha/Hbeta line ratios 
                # See section 3.4 for [OIII]/Hbeta line ratios
                for eline, A in zip(["HALPHA", 
                                     "HBETA", 
                                     "OIII5007", 
                                     "OIII4959"], 
                                    [L_Ha_erg_s[nn], 
                                     L_Ha_erg_s[nn] / 3, 
                                     L_Ha_erg_s[nn] / 10, 
                                     L_Ha_erg_s[nn] / 30]):

                    # Compute the central wavelength & width from the velocity & velocity dispersion
                    sigma_gas_A = sigma_gas_kms[nn] * eline_lambdas_A[eline] / (constants.c / 1e3)
                    lambda_gas_A = get_wavelength_from_velocity(lambda_rest=eline_lambdas_A[eline],
                                                                v=v_gas_kms[nn],
                                                                units='km/s')

                    # Define a Gaussian
                    line = A * np.exp( -(np.exp(lambda_vals_ssp_log) - lambda_gas_A)**2 / (2 * sigma_gas_A**2))

                    # Add to spectrum
                    spec_log_lines += line

            else:
                raise NotImplementedError("Grids need fixing!")
                if eline_model[nn] == "HII":
                    grid_df = load_HII_grid()
                    cond = grid_df["Nebular abundance (Z/Zsun)"] == 2.0
                    cond &= grid_df["log(U) (inner)"] == -3
                    cond &= grid_df["log(P/k)"] == 5.0
                elif eline_model[nn] == "AGN":
                    raise NotImplementedError("Grids need fixing!")
                    grid_df = load_AGN_grid()
                    cond = grid_df["Nebular abundance (Z/Zsun)"] == 2.0
                    cond &= grid_df["log(U) (inner)"] == -2.0
                    cond &= grid_df["log(P/k)"] == 6.0
                    cond &= grid_df["E_peak (log_10(keV))"] == -1.5

                eline_list = [e for e in eline_lambdas_A if e in grid_df.columns]
                for eline in eline_list:
                    # Compute the central wavelength & width from the velocity & velocity dispersion
                    sigma_gas_A = sigma_gas_kms[nn] * eline_lambdas_A[eline] / (constants.c / 1e3)
                    lambda_gas_A = get_wavelength_from_velocity(lambda_rest=eline_lambdas_A[eline],
                                                                v=v_gas_kms[nn],
                                                                units='km/s')

                    # Determine the strength of the line
                    # NOTE: 10^41 erg s^-1 corresponds to a SFR of ~0.5 Msun/yr
                    L_Hb_erg_s = L_Ha_erg_s[nn] * grid_df.loc[cond, "HBETA"].values[0] / grid_df.loc[cond, "HALPHA"].values[0]
                    A = grid_df.loc[cond, eline].values[0] * L_Hb_erg_s

                    # Define a Gaussian
                    line = A * np.exp( -(np.exp(lambda_vals_ssp_log) - lambda_gas_A)**2 / (2 * sigma_gas_A**2))

                    # Add to spectrum
                    spec_log_lines += line

        # Add to spectrum
        spec_log_conv_prev = np.copy(spec_log_conv)
        spec_log_conv += spec_log_lines

        if plotit:
            ax = axs[ax_ctr]
            ax.plot(np.exp(lambda_vals_ssp_log), spec_log_lines / plot_norm, color="green", label="Emission lines")
            ax.plot(np.exp(lambda_vals_ssp_log), spec_log_conv_prev / plot_norm, color="black", alpha=0.5, label="Without emission lines")
            ax.plot(np.exp(lambda_vals_ssp_log), spec_log_conv / plot_norm, color="black", label="With emission lines")
            ax.set_ylabel(r"$L_{\lambda}(\lambda) \,\rm (10^{39}\,erg\,s^{-1}\,\AA^{-1}$)")
            ax.set_xlabel(r"Rest-frame wavelength ($\rm \AA$)")
            ax.legend(**legend_kwargs)
            ax.autoscale(enable="True", axis="x", tight=True)
            ax.text(s="Adding emission lines", transform=ax.transAxes, **text_kwargs)
            ax_ctr += 1

    ###########################################################################
    # 4c. Add AGN continuum, if needed
    # References:
    # http://www.chara.gsu.edu/~crenshaw/4.AGN_Components.pdf
    # https://www.astro.rug.nl/~koopmans/lecture6.pdf
    # Typical values of alpha_nu range from 0.3 - 2; Groves+2004 suggest a 
    # range of 1.2 - 2.0
    if agn_continuum:
        # Compute the continuum normalisation
        alpha_lambda = 2 - alpha_nu

        # Add an AGN continuum
        lambda_norm_idx = np.nanargmin(np.abs(np.exp(lambda_vals_ssp_log) - lambda_norm_A))
        F_star_0 = spec_log_conv[lambda_norm_idx]
        F_agn_0 = x_AGN * F_star_0
        F_lambda_0 = F_agn_0 / lambda_norm_A**(-alpha_lambda)

        # Compute the continuum
        spec_log_agn = F_lambda_0 * np.exp(lambda_vals_ssp_log)**(-alpha_lambda)

        # Add to spectrum
        spec_log_conv_prev = np.copy(spec_log_conv)
        spec_log_conv += spec_log_agn

        if plotit:
            ax = axs[ax_ctr]
            ax.plot(np.exp(lambda_vals_ssp_log), spec_log_agn / plot_norm, color="magenta", 
                    label=r"AGN continuum")
            ax.plot(np.exp(lambda_vals_ssp_log), spec_log_conv_prev / plot_norm, color="black", lw=0.5, alpha=0.5, label="Without AGN continuum")
            ax.plot(np.exp(lambda_vals_ssp_log), spec_log_conv / plot_norm, color="black", label="With AGN continuum")
            ax.set_ylabel(r"$L_{\lambda}(\lambda) \,\rm (10^{39}\,erg\,s^{-1}\,\AA^{-1}$)")
            ax.set_xlabel(r"Rest-frame wavelength ($\rm \AA$)")
            ax.legend(**legend_kwargs)
            ax.autoscale(enable="True", axis="x", tight=True)
            ax.text(s="Adding AGN continuum", transform=ax.transAxes, **text_kwargs)
            ax_ctr += 1

    ###########################################################################
    # 4d. Apply extinction
    if A_V > 0:
        R_V = 4.05  # For the Calzetti+2000 extinction curve
        A_vals = extinction.calzetti00(wave=np.exp(lambda_vals_ssp_log), a_v=A_V, r_v=R_V, unit="aa")
        spec_log_conv_prev = np.copy(spec_log_conv)
        spec_log_conv *= 10**(-0.4 * A_vals)

        if plotit:
            ax = axs[ax_ctr]
            ax.plot(np.exp(lambda_vals_ssp_log), spec_log_conv_prev / plot_norm, color="black", lw=0.5, alpha=0.5, label="Before extinction")
            ax.plot(np.exp(lambda_vals_ssp_log), spec_log_conv / plot_norm, color="black", label="After extinction")
            ax.set_ylabel(r"$L_{\lambda}(\lambda) \,\rm (10^{39}\,erg\,s^{-1}\,\AA^{-1}$)")
            ax.set_xlabel(r"Rest-frame wavelength ($\rm \AA$)")
            ax.legend(**legend_kwargs)
            ax.autoscale(enable="True", axis="x", tight=True)
            ax.text(s="Applying extinction curve", transform=ax.transAxes, **text_kwargs)
            ax_ctr += 1

    ###########################################################################
    # 5. Apply the redshift 
    lambda_vals_ssp_log_redshifted = lambda_vals_ssp_log + np.log(1 + z)

    ##########################################################################
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
    if not np.isinf(SNR):
        rng = RandomState(seed)
        spec_err = spec_wifes / SNR
        noise = rng.normal(loc=0, scale=spec_err)
        spec = spec_wifes + noise
    else:
        spec_err = np.zeros(spec_wifes.shape)
        spec = spec_wifes

    # Plot to check
    if plotit:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 2.5))
        fig.subplots_adjust(hspace=0, left=0.1, right=0.9,)
        ax.errorbar(x=lambda_vals_wifes_A, y=spec / plot_norm, yerr=spec_err / plot_norm, linewidth=0.5, elinewidth=0.5, color="black")
        ax.set_xlabel(r"Observer-frame wavelength ($\rm \AA$)")
        ax.set_ylabel(r"$L_{\lambda}(\lambda) \,\rm (10^{39}\,erg\,s^{-1}\,\AA^{-1}$)")
        ax.autoscale(enable="True", axis="x", tight=True)
        ax.text(s="Final redshifted spectrum with noise", transform=ax.transAxes, **text_kwargs)
        ax.grid(color="lightgrey", alpha=0.5)

    # Tidy up the plot
    if plotit: 
        [ax.set_xticklabels([]) for ax in axs[:-1]]
        [ax.set_xlabel("") for ax in axs[:-1]]
        [ax.grid(color="lightgrey", alpha=0.5) for ax in axs]
        [ax.set_ylim([0, ax.get_ylim()[1] * 1.15]) for ax in axs]

    ###########################################################################
    # 10. Return.
    return spec, spec_err, lambda_vals_wifes_A

###############################################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###############################################################################
if __name__ == "__main__":
    from ppxftests.sfhutils import load_sfh, metallicities_padova, ages_padova
    from ppxftests.ppxf_plot import plot_sfh_mass_weighted, plot_sfh_light_weighted, plot_sfr
    import sys 

    savefigs = True  # If true, save figures for paper
    fig_path = "/priv/meggs3/u5708159/S7/mar23/ppxf/figs/paper/"

    ###########################################################################
    # Mock spectra options
    ###########################################################################
    isochrones = "Padova"  # Set of isochrones to use 
    z = 0.05               # Redshift 
    SNR = 50              # S/N ratio
    gals = [int(g) for g in sys.argv[1:]]               # Galaxy to load

    for gal in gals:
        ###########################################################################
        # DEFINE THE SFH
        ###########################################################################
        sfh_mw, sfh_lw, sfr_avg, sigma_star_kms = load_sfh(gal=gal, plotit=False)

        # Save 
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 6))
        fig.subplots_adjust(hspace=0)
        bbox = axs[0].get_position()
        bbox_new = [bbox.x0, bbox.y0, bbox.width, bbox.height * 0.7]
        axs[0].set_position(bbox_new)
        plot_sfh_mass_weighted(sfh_mw, ages_padova, metallicities_padova, ax=axs[2])
        plot_sfh_light_weighted(sfh_lw, ages_padova, metallicities_padova, ax=axs[1])
        plot_sfr(sfr_avg, ages_padova, metallicities_padova, ax=axs[0])
        [ax.set_xticklabels([]) for ax in axs[:2]]
        if savefigs:
            fig.savefig(os.path.join(fig_path, f"ga{gal:04d}_sfh.pdf"), format="pdf", bbox_inches="tight")

        ###########################################################################
        # CREATE THE MOCK SPECTRUM
        ###########################################################################
        spec, spec_err, lambda_valsA = create_mock_spectrum(
            sfh_mass_weighted=sfh_mw,
            isochrones=isochrones,
            ngascomponents=0,
            agn_continuum=True, alpha_nu=1.0, x_AGN=0.5,
            sigma_gas_kms=[40, 350], v_gas_kms=[0, -100], 
            A_V=1.5,
            z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
            plotit=True)

        # Save
        if savefigs:
            plt.figure(2).savefig(os.path.join(fig_path, f"ga{gal:004}_mock_spec_procedure.pdf"), bbox_inches="tight", format="pdf")
            plt.figure(3).savefig(os.path.join(fig_path, f"ga{gal:004}_mock_spec_final.pdf"), bbox_inches="tight", format="pdf")

        ###########################################################################
        # Plot SFHs for ga0001, ga0002, ga0011, ga0024.
        ###########################################################################
        gals = [1, 2, 11, 24]
        fig, axs = fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(13.3333, 8))
        fig.subplots_adjust(hspace=0)
        for gg, gal in enumerate(gals):
            sfh_mw, sfh_lw, sfr_avg, sigma_star_kms = load_sfh(gal=gal, plotit=False)
            M_tot = np.nansum(sfh_mw)
            plot_sfh_light_weighted(sfh_lw, ages_padova, metallicities_padova, ax=axs[gg])
            axs[gg].text(s=f"ga{gal:04d}" + r" - $\log_{10} (M_* [{\rm M_{\odot}}]) = %.3f$" % np.log10(M_tot), x=0.01, y=0.93, ha="left", va="top", transform=axs[gg].transAxes)
        [ax.set_xticklabels([]) for ax in axs[:2]]
        caxs = fig.get_axes()[4:]
        [cax.set_visible(False) for cax in caxs[1:]]
        cax = fig.get_axes()[4]
        bbox = cax.get_position()
        cax.set_position([bbox.x0, 
                         fig.get_axes()[-1].get_position().y0, 
                         bbox.width, 
                         4 * bbox.height])

        if savefigs:
            fig.savefig(os.path.join(fig_path, f"sim_gal_sfhs.pdf"), bbox_inches="tight", format="pdf")
