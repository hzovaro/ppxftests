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
    abspath = __file__.split("/ssputils.py")[0]
    ssp_template_path = os.path.join(abspath, "SSP_templates")
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


###############################################################################
# Function for computing the bin widths - useful for computing mean SFR in 
# each bin
############################################################################## 
def get_bin_edges_and_widths(isochrones):
    # Get the corresponding ages 
    _, _, _, ages = load_ssp_templates(isochrones)

    # Compute the bin edges
    bin_edges = np.zeros(len(ages) + 1)
    for aa in range(1, len(ages)):
        bin_edges[aa] = 10**(0.5 * (np.log10(ages[aa - 1]) + np.log10(ages[aa])) )

    # Compute the edges of the first and last bins
    delta_log_age = np.diff(np.log10(ages))[0]
    age_0 = 10**(np.log10(ages[0]) - delta_log_age)
    age_last = 10**(np.log10(ages[-1]) + delta_log_age)
    bin_edges[0] = 10**(0.5 * (np.log10(age_0) + np.log10(ages[0])) )
    bin_edges[-1] = 10**(0.5 * (np.log10(ages[-1]) + np.log10(age_last)))
    
    # Finally, compute the sizes of the bins
    bin_widths = np.diff(bin_edges)

    return bin_edges, bin_widths

##############################################################################
# Basic tests
##############################################################################
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt 
    plt.ion()
    plt.close("all")

    # Load templates
    isochrones = sys.argv[1]
    stellar_templates_linear, lambda_vals_linear, metallicities, ages =\
        load_ssp_templates(isochrones)

    # Plot a few templates to check that they look as expected 
    for met_idx in range(len(metallicities)):
        fig, ax = plt.subplots(figsize=(15, 5))
        lambda_idx = np.nanargmin(np.abs(lambda_vals_linear - 5000))
        for age_idx in [0, 10, 25, -1]:
            F_lambda = stellar_templates_linear[:, met_idx, age_idx]
            F_lambda /= F_lambda[lambda_idx]
            ax.plot(lambda_vals_linear, F_lambda,
                    label=r"$Z = %.3f, t = %.2f\,\rm Myr$" % (metallicities[met_idx], ages[age_idx] / 1e6))
        ax.set_xlabel(f"Wavelength $\lambda$ (Å)")
        ax.set_ylabel(r"$F_\lambda(\lambda) / F_\lambda(5000\,Å)$")
        ax.legend()  

    # Check that the bin sizes are correct
    bin_edges, bin_widths = get_bin_edges_and_widths(isochrones=isochrones)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(x=ages, y=np.ones(len(ages)), label="Bin centres")
    ax.scatter(x=bin_edges, y=np.ones(len(bin_edges)), label="Bin edges")
    ax.set_xscale("log")
    ax.set_xlabel("Age (yr)")
    ax.legend()

    plt.show()


