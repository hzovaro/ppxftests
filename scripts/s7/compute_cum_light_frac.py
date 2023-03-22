import os, sys 
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

from ppxftests.ssputils import load_ssp_templates

from IPython.core.debugger import Tracer

"""
Compute the cumulative light fraction as a function of cutoff age 
Store in the DataFrame & save back to file.
"""
s7_data_path = "/priv/meggs3/u5708159/S7/ppxf"

###############################################################################
# Load the DataFrame
###############################################################################
# Load the DataFrame containing all S7 galaxies
df_fname = f"s7_ppxf.hd5"
df_all = pd.read_hdf(os.path.join(s7_data_path, df_fname), key="s7")

# Get the age & metallicity dimensions
_, _, metallicities, ages = load_ssp_templates("Padova")

###############################################################################
# Compute the cumulative light fraction as a function of cutoff age 
###############################################################################
gals = df_all["Galaxy"].unique()

for gal in gals:
    cond = df_all["Galaxy"] == gal
    cumfrac = 