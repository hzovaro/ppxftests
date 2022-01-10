import numpy as np
from numpy.polynomial import polynomial
from time import time 

from astropy.io import fits

from ppxftests.run_ppxf import run_ppxf
from ppxftests.ssputils import load_ssp_templates
from ppxftests.mockspec import create_mock_spectrum
from ppxftests.ppxf_plot import plot_sfh_mass_weighted

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

"""
PROBLEM: ppxf doesn't accurately fit SFHs when we define the input using 
Gaussian functions in either logarithmic or linear age space. This is probably 
due to the way that regularisation is applied to the SFH.

SOLUTION: 
1. generate a SFH using a basic Gaussian function, or other method.
2. Run ppxf to fit the SFH. 
3. Take the output SFH from ppxf, and re-run ppxf on this. It SHOULD correctly
   fit the SFH.
"""
###########################################################################
# Settings
###########################################################################
isochrones = "Padova"
SNR = 100
sigma_star_kms = 250
z = 0.01

###########################################################################
# Generate the input SFH
###########################################################################
# Load the stellar templates so we can get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates(isochrones)
N_ages = len(ages)
N_metallicities = len(metallicities)

# Simple Gaussian SFH
xx, yy = np.meshgrid(range(N_ages), range(N_metallicities))
x0 = 5
y0 = 1.5
sigma_x = 1
sigma_y = 0.5
sfh_lw_young = np.exp(- (xx - x0)**2 / (2 * sigma_x**2)) *\
               np.exp(- (yy - y0)**2 / (2 * sigma_y**2))
sfh_lw_young /= np.nansum(sfh_lw_young)
sfh_mw_young = sfh_lw_young * 1e8

x0 = 69
y0 = 0
sigma_x = 1
sigma_y = 0.5
sfh_lw_old = np.exp(- (xx - x0)**2 / (2 * sigma_x**2)) *\
             np.exp(- (yy - y0)**2 / (2 * sigma_y**2))
sfh_lw_old /= np.nansum(sfh_lw_old)
sfh_mw_old = sfh_lw_old * 1e10

# Add the young & old components
sfh_mw = sfh_mw_old + sfh_mw_young

# Normalise 
# sfh_lw /= np.sum(sfh_lw)

# Convert to mass weights: total stellar mass of 10^10 Msun
# sfh_mw = sfh_lw * 1e10

spec, spec_err, lambda_vals_A = create_mock_spectrum(
    sfh_mass_weighted=sfh_mw,
    isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
    plotit=True)  

###########################################################################
# Fit with ppxf
###########################################################################
t = time()
pp1 = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
              z=z, ngascomponents=1,
              auto_adjust_regul=True,
              isochrones="Padova",
              fit_gas=False, tie_balmer=True,
              delta_regul_min=1, regul_max=5e4, delta_delta_chi2_min=1,
              plotit=False, savefigs=False, interactive_mode=False)
print(f"Total time in run_ppxf: {time() - t:.2f} seconds")

# CHECK: the input & output SFHs 
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 11))
fig.subplots_adjust(hspace=0.3)
# Mass-weighted 
for ax in [axs[0][0], axs[1][0]]:
    ax.step(range(len(sfh_mw[0])),
            sfh_mw[0],
            color="black", where="mid", label="Input SFH")
    ax.step(range(len(pp1.weights_mass_weighted[0])),
            pp1.weights_mass_weighted[0],
            color="red", alpha=0.1, where="mid", label="ppxf fits")
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
    # ax.step(range(len(sfh_lw[0])),
    #         sfh_lw[0],
    #         color="black", where="mid", label="Input SFH")
    ax.step(range(len(pp1.weights_light_weighted[0])),
            pp1.weights_light_weighted[0],
            color="red", alpha=0.1, where="mid", label="ppxf fits")
    ax.legend()
    ax.set_xticks(range(len(ages)))
    ax.set_xlabel("Age (Myr)")
    ax.set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")
    ax.autoscale(axis="x", enable=True, tight=True)
    ax.set_ylabel("Template weight (light-weighted)")
axs[1][1].set_yscale("log")

###########################################################################
# Re-create the input spectrum from ppxf 
###########################################################################
sfh_ppxf_lw = pp1.weights_light_weighted
sfh_ppxf_mw = pp1.weights_mass_weighted

spec_ppxf, spec_ppxf_err, lambda_vals_A = create_mock_spectrum(
    sfh_mass_weighted=sfh_ppxf_mw,
    isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
    plotit=True)  

###########################################################################
# Re-run ppxf to check that it correctly fits the input SFH
###########################################################################
t = time()
pp2 = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
               z=z, ngascomponents=1,
               auto_adjust_regul=True,
               isochrones="Padova",
               fit_gas=False, tie_balmer=True,
               delta_regul_min=1, regul_max=5e4, delta_delta_chi2_min=1,
               plotit=False, savefigs=False, interactive_mode=False)
print(f"Total time in run_ppxf: {time() - t:.2f} seconds")

# CHECK: the input & output SFHs 
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 11))
fig.subplots_adjust(hspace=0.3)
# Mass-weighted 
for ax in [axs[0][0], axs[1][0]]:
    ax.step(range(len(sfh_ppxf_mw[0])),
            sfh_ppxf_mw[0],
            color="black", where="mid", label="Input SFH")
    ax.step(range(len(pp2.weights_mass_weighted[0])),
            pp2.weights_mass_weighted[0],
            color="red", alpha=0.1, where="mid", label="ppxf fits")
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
    ax.step(range(len(sfh_ppxf_lw[0])),
            sfh_ppxf_lw[0],
            color="black", where="mid", label="Input SFH")
    ax.step(range(len(pp2.weights_light_weighted[0])),
            pp2.weights_light_weighted[0],
            color="red", alpha=0.1, where="mid", label="ppxf fits")
    ax.legend()
    ax.set_xticks(range(len(ages)))
    ax.set_xlabel("Age (Myr)")
    ax.set_xticklabels(["{:}".format(age / 1e6) for age in ages], rotation="vertical")
    ax.autoscale(axis="x", enable=True, tight=True)
    ax.set_ylabel("Template weight (light-weighted)")
axs[1][1].set_yscale("log")

###########################################################################
# Save the mass weights to a .npy file so that we can load it 
###########################################################################
# Create a .fits file 
header = fits.Header()
header["NAGES"] = N_ages
header["NMET"] = N_metallicities
header["ISCHRN"] = isochrones
header["MTOT"] = np.nansum(pp2.weights_mass_weighted)
header["LOGMTOT"] = np.log10(np.nansum(pp2.weights_mass_weighted))
hdu_primary = fits.PrimaryHDU(header=header)

hdu_sfh_mw = fits.ImageHDU(pp2.weights_mass_weighted, name="SFH_MW")
hdu_sfh_lw = fits.ImageHDU(pp2.weights_light_weighted, name="SFH_LW")
hdulist = fits.HDUList([hdu_primary, hdu_sfh_mw, hdu_sfh_lw])

hdulist.writeto("SFHs/sfh_mw_old+young.fits", overwrite=True)

###########################################################################
# TEST: load the .npy file, run it through ppxf
###########################################################################
sfh_mw_test = np.load("SFHs/sfh_mw_old+young.npy")
spec_test, spec_test_err, lambda_vals_A = create_mock_spectrum(
    sfh_mass_weighted=sfh_mw_test,
    isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
    plotit=True)  

pp_test = run_ppxf(spec=spec_test, spec_err=spec_test_err, lambda_vals_A=lambda_vals_A,
                   z=z, ngascomponents=1,
                   auto_adjust_regul=True,
                   isochrones="Padova",
                   metals_to_use=["004"],
                   fit_gas=False, tie_balmer=True,
                   delta_regul_min=1, regul_max=5e4, delta_delta_chi2_min=1,
                   plotit=False, savefigs=False, interactive_mode=False)

