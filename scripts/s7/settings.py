from enum import Enum
import numpy as np
import os

from ppxftests.ssputils import log_rebin_and_convolve_stellar_templates

CLEAN = False   # Whether to use the CLEAN keyword in ppxf
FWHM_WIFES_INST_A = 1.4   # WiFeS LSF resolution

# Load the stellar ages to put in the DataFrame
isochrones = "Padova"
metals_to_use = ['004', '008', '019']
_, _, metallicities, ages =\
    log_rebin_and_convolve_stellar_templates(isochrones=isochrones, metals_to_use=['004', '008', '019'], 
                                                FWHM_inst_A=FWHM_WIFES_INST_A, 
                                                velscale=1.0) # NOTE: velscale here is a dummy parameter since we just want the ages, not the templates
# Paths
s7_data_path = "/priv/meggs3/u5708159/S7/mar23"
lzifu_input_path = f"{s7_data_path}/LZIFU/data/"
lzifu_output_path = f"{s7_data_path}/LZIFU/products/"
if CLEAN:
    ppxf_output_path = f"{s7_data_path}/ppxf/clean/"
    fig_path = f"{s7_data_path}/ppxf/figs_clean/paper/"
else:
    ppxf_output_path = f"{s7_data_path}/ppxf/noclean/"
    fig_path = f"{s7_data_path}/ppxf/figs_noclean/paper/"

# Aperture type
class Aperture(Enum):
    RE1 = 0
    FOURAS = 1
    SDSS = 2
    ONEKPC = 3

def get_aperture_coords(aperture):
    """Get the coordinates of the spectrum in the FITS files corresponding to the input aperture."""
    return np.unravel_index(aperture.value, (2, 2))

# List of all galaxies
gals_all = [g.strip("\n") for g in open(os.path.join(s7_data_path, "gal_list.txt")).readlines()]
