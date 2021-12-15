# imports 
import os
import numpy as np

from ppxftests.run_ppxf import run_ppxf
from ppxftests.mockspec import create_mock_spectrum, get_age_and_metallicity_values
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