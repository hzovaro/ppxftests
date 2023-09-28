import os
import numpy as np

from astropy.io import fits

import matplotlib.pyplot as plt
 
from settings import lzifu_input_path, lzifu_output_path, gals_all, get_aperture_coords, extra_bad_pixel_ranges_dict

def load_galaxy_spectrum(gal, aperture, plotit=False):
    """
    Load the galaxy spectrum for galaxy gal from the desired aperture.
    """
    # Check inputs
    assert gal in gals_all
    rr, cc = get_aperture_coords(aperture)

    # Load the aperture spectrum
    hdulist_in = fits.open(os.path.join(lzifu_input_path, f"{gal}.fits.gz"))
    spec = hdulist_in["PRIMARY"].data[:, rr, cc]
    spec_err = np.sqrt(hdulist_in["VARIANCE"].data[:, rr, cc])
    norm = hdulist_in["NORM"].data[rr, cc]
    z = hdulist_in[0].header["Z"]

    N_lambda = hdulist_in[0].header["NAXIS3"]
    dlambda_A = hdulist_in[0].header["CDELT3"]
    lambda_0_A = hdulist_in[0].header["CRVAL3"]
    lambda_vals_obs_A = np.array(list(range(N_lambda))) * dlambda_A + lambda_0_A
    lambda_vals_rest_A = lambda_vals_obs_A / (1 + z)

    # Define bad pixel ranges
    bad_pixel_ranges_A = [
         [(6300 - 10) / (1 + z), (6300 + 10) / (1 + z)], # Sky line at 6300
         [(5577 - 10) / (1 + z), (5577 + 10) / (1 + z)], # Sky line at 5577
         [(6360 - 10) / (1 + z), (6360 + 10) / (1 + z)], # Sky line at 6360
         [(5700 - 10) / (1 + z), (5700 + 10) / (1 + z)], # Sky line at 5700                              
         [(5889 - 30), (5889 + 20)], # Interstellar Na D + He line 
         [(5889 - 10) / (1 + z), (5889 + 10) / (1 + z)], # solar NaD line
    ]
    if gal in extra_bad_pixel_ranges_dict:
        bad_pixel_ranges_A += extra_bad_pixel_ranges_dict[gal]

    # Load the LZIFU fit
    lzifu_fname = f"{gal}_merge_comp.fits"
    hdulist = fits.open(os.path.join(lzifu_output_path, lzifu_fname))
    spec_cont_lzifu = hdulist["CONTINUUM"].data[:, rr, cc] 
    spec_elines = hdulist["LINE"].data[:, rr, cc]
    spec_cont_only = spec - spec_elines

    # Plot to check
    if plotit:
        _, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(x=lambda_vals_obs_A, y=spec, yerr=spec_err, color="k", label="Data")
        ax.plot(lambda_vals_obs_A, spec_cont_lzifu, color="green", label="Continuum fit")
        ax.plot(lambda_vals_obs_A, spec_elines, color="magenta", label="Emission line fit")
        ax.plot(lambda_vals_obs_A, spec_cont_lzifu + spec_elines, color="orange", label="Total fit")
        ax.plot(lambda_vals_obs_A, spec_cont_only, color="red", label="Data minus emission lines")
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel(r"Normalised flux ($F_\lambda$)")
        ax.legend()

    return lambda_vals_obs_A, lambda_vals_rest_A, spec, spec_cont_only, spec_err, norm, bad_pixel_ranges_A