import os
import numpy as np

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
    ssp_template_path = "/home/u5708159/python/Modules/ppxftests/SSP_templates"
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

