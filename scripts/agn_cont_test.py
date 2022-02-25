import os, sys
import numpy as np
from time import time 
from tqdm import tqdm
from itertools import product
import multiprocessing
import pandas as pd

from astropy.io import fits

from ppxftests.run_ppxf import run_ppxf
from ppxftests.ssputils import load_ssp_templates
from ppxftests.mockspec import create_mock_spectrum
from ppxftests.sfhutils import load_sfh, compute_mw_age, compute_lw_age, compute_cumulative_mass, compute_cumulative_light
from ppxftests.sfhutils import compute_mean_age, compute_mean_mass, compute_mean_sfr, compute_mean_1D_sfh
from ppxftests.ppxf_plot import plot_sfh_mass_weighted, plot_sfh_light_weighted

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

isochrones = "Padova"
gal = 10
SNR = 100
z = 0.05
A_V = 2.0
sfh_mw_input, sfh_lw_input, sfr_avg_input, sigma_star_kms = load_sfh(gal, plotit=False)
        
# Create spectrum
spec, spec_err, lambda_vals_A = create_mock_spectrum(
    sfh_mass_weighted=sfh_mw_input,
    agn_continuum=True, alpha_nu=1.0, x_AGN=0.5,
    ngascomponents=1, sigma_gas_kms=[70], v_gas_kms=[0], eline_model=["HII"], L_Ha_erg_s=[1e41],
    isochrones=isochrones, z=z, SNR=SNR, sigma_star_kms=sigma_star_kms,
    A_V=A_V, seed=0, plotit=False)


# run ppxf
pp = run_ppxf(spec=spec, spec_err=spec_err, lambda_vals_A=lambda_vals_A,
            isochrones=isochrones,
            z=z, fit_gas=True, ngascomponents=1,
            fit_agn_cont=True,
            reddening=1.0, mdegree=-1,
            regularisation_method="fixed", regul_fixed=100,
            plotit=True)

# Need to figure out how/why the weights are weird 
# Is it because the weights correspond to the spectrum BEFORE the extinction
# is applied?

