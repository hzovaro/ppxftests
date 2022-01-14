# imports 
import os
import numpy as np
from time import time
from tqdm import tqdm
from itertools import product
from scipy import constants, ndimage
from astropy.io import fits

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
# SFH filename
sfh_fname = "SFHs/sfh_mw_old+young.fits"

# Other parameters
isochrones = "Padova"
SNR = 1e5
z = 0.01
sigma_star_kms = 250

# Figure filename
# fname_str = os.path.join(fig_path, "noise", f"age={age:g}_fwhm={FWHM_age:.2g}_SNR={SNR}")
fname_str = None

################################################################################################################
# LOAD THE SFH
################################################################################################################
hdulist = fits.open(sfh_fname)
sfh_input_mass_weighted = hdulist["SFH_MW"].data 
sfh_input_light_weighted = hdulist["SFH_LW"].data 

# Load the stellar templates so we can get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates(isochrones)
N_ages = len(ages)
N_metallicities = len(metallicities)

###############################################################################
# Run N times with different noise realisations to look at the effects of noise
###############################################################################
pp_list = []
sfh_fit_mass_weighted_list = []
sfh_fit_light_weighted_list = []
iters = 10
for ii in tqdm(range(iters)):
    ###########################################################################
    # CREATE THE MOCK SPECTRUM WITH RANDOM NOISE
    ###########################################################################
    spec, spec_err, lambda_vals_A = create_mock_spectrum(
        sfh_mass_weighted=sfh_input_mass_weighted,
        isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
        plotit=False)  

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

    # fig.savefig(f"{fname_str}_noise_test_niters={ii}.pdf", bbox_inches="tight")
