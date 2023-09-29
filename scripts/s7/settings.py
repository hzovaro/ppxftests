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
test_path = f"{s7_data_path}/ppxf/tests/"  # Path for storing misc. files/figures from tests
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

def get_aperture_label(aperture):
    """Return a nice LaTeX-formatted string to use in plots."""
    if aperture == Aperture.RE1:
        return r"$1R_e$"
    elif aperture == Aperture.FOURAS:
        return r"$4''$"
    elif aperture == Aperture.ONEKPC:
        return r"$1\,\rm kpc$"  
    else:
        raise ValueError()

def get_aperture_coords(aperture):
    """Get the coordinates of the spectrum in the FITS files corresponding to the input aperture."""
    return np.unravel_index(aperture.value, (2, 2))

# List of all galaxies
gals_all = [g.strip("\n") for g in open(os.path.join(s7_data_path, "gal_list.txt")).readlines()]

# Extra wavelength regions to mask out for some galaxies with particularly bad emission line residuals or other issues 
extra_bad_pixel_ranges_dict = {
    "ESO362-G18": [
        (4861 - 75, 5007 + 75),
        (6562.8 - 100, 6562.8 + 100),
    ],
    "FAIRALL49": [
        (4861 - 75, 5007 + 75),
        (6562.8 - 100, 6562.8 + 100),
    ],
    "IC4329A": [  # NOTE: not successful
        (4861 - 75, 5007 + 75),  # Hbeta + [OIII]
        (6562.8 - 300, 6562.8 + 300),  # Halpha 
        (5889 - 100, 5889 + 100),  # NaD
    ],
    "MCG-03-34-064": [
        (4861 - 75, 5007 + 75),  # Hbeta + [OIII]
        (6562.8 - 100, 6562.8 + 100),  # Halpha 
        (4650, 5400),
        (5600, 5900),
    ],
    "NGC424": [
        (4861 - 100, 5007 + 100),  # Hbeta + [OIII]
        (5300 - 25, 5300 + 25), # Strong mystery emission line
        (5718 - 25, 5718 + 25), # Strong mystery emission line
        (6562.8 - 300, 6562.8 + 150),  # Halpha plus strong blueward emission line
    ],
    "NGC1667": [  # Really bad flux cal error between blue/red arms
        (5500, 9000)
    ],
    "NGC6860": [
        (4861 - 100, 5007 + 100), # Hbeta + [OIII]
        (5000, 5500),
        (6562.8 - 200, 6562.8 + 200), # Halpha 
    ],
    "NGC7469": [  # NOTE: not successful
        (4100, 6000),
        (6300, 9000)
    ],
}

gals_unreliable_stellar_measurements = [
    "IC4329A",
    "NGC7469",
]