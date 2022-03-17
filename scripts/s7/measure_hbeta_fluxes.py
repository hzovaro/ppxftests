import os
import numpy as np
import pandas as pd
from itertools import product

from astropy.io import fits
import extinction

from cosmocalc import get_dist

import matplotlib.pyplot as plt
plt.ion()
plt.close("all")

from IPython.core.debugger import Tracer

"""
Measure the total Hbeta fluxes of galaxies in the S7 sample from the LZIFU
fits. This will give us an idea of the strength of the AGN continua in the 
sample, and in turn the possible effect this will have upon accurate recovery
of the SFH by ppxf.

"""

eline_lambdas_A = {
            "NeV3347" : 3346.79,
            "OIII3429" : 3429.0,
            "OII3726" : 3726.032,
            "OII3729" : 3728.815,
            "NEIII3869" : 3869.060,
            "HeI3889" : 3889.0,
            "HEPSILON" : 3970.072,
            "HDELTA" : 4101.734, 
            "HGAMMA" : 4340.464, 
            "HEI4471" : 4471.479,
            "OIII4363" : 4363.210, 
            "HBETA" : 4861.325, 
            "OIII4959" : 4958.911, 
            "OIII5007" : 5006.843, 
            "HEI5876" : 5875.624, 
            "OI6300" : 6300.304, 
            "SIII6312" : 6312.060,
            "OI6364" : 6363.776,
            "NII6548" : 6548.04, 
            "HALPHA" : 6562.800, 
            "NII6583" : 6583.460,
            "SII6716" : 6716.440, 
            "SII6731" : 6730.810,
            "SIII9069": 9068.600,
            "SIII9531": 9531.100
}

s7_dir = os.getenv("S7_DIR")
fnames = [f for f in os.listdir(os.path.join(s7_dir, "2_Post-processed_mergecomps")) if not f.startswith("._")]
gals = [f.split("_best_components.fits")[0] for f in fnames]

df = pd.DataFrame(index=gals)

for gal, fname in zip(gals, fnames):
    hdulist = fits.open(os.path.join(s7_dir, "2_Post-processed_mergecomps", fname))

    # Compute extinction
    # Extract Halpha and Hbeta fluxes from the zeroth slice of the extension, 
    # which contains the total flux in all emission line components
    F_Ha_map = hdulist["HALPHA"].data[0] * 1e-16
    F_Hb_map = hdulist["HBETA"].data[0] * 1e-16
    F_Ha_err_map = hdulist["HALPHA_ERR"].data[0] * 1e-16
    F_Hb_err_map = hdulist["HBETA_ERR"].data[0] * 1e-16
    N_y, N_x = F_Hb_map.shape

    # Take a S/N cut 
    F_Ha_map[F_Ha_map / F_Ha_err_map < 3] = np.nan
    F_Hb_map[F_Hb_map / F_Hb_err_map < 3] = np.nan

    # Correct for extinction
    ratio_map = F_Ha_map / F_Hb_map
    E_ba_map = 2.5 * (np.log10(ratio_map)) - 2.5 * np.log10(2.86)

    # Calculate ( A(Ha) - A(Hb) ) / E(B-V) from extinction curve
    R_V = 3.1
    E_ba_over_E_BV = float(extinction.fm07(np.array([4861.325]), a_v=1.0) - extinction.fm07(np.array([6562.800]), a_v=1.0)) /  1.0 * R_V
    E_BV_map = 1 / E_ba_over_E_BV * E_ba_map

    # Calculate A(V)
    A_V_map = R_V * E_BV_map
    A_V_map[np.isinf(A_V_map)] = np.nan
    A_V_map[A_V_map < 0] = np.nan
    A_V_map[A_V_map > 15] = np.nan  # This eliminates a single spaxel in NGC1097 which has a measured A_V of > 20 and a corresponding log Hb ~ 47!!

    # Extract total emission line fluxes for each line, correct for extinction
    # and store in DataFrame.
    for eline in [e for e in ["OII3726", "OII3729","HALPHA", "HBETA", "OIII5007", "NII6583", "SII6716", "SII6731"] if e in hdulist]:

        # Extract Halpha and Hbeta fluxes from the zeroth slice of the extension, 
        # which contains the total flux in all emission line components
        F_map = hdulist[f"{eline}"].data[0] * 1e-16 if hdulist[f"{eline}"].data.ndim == 3 else hdulist[f"{eline}"].data * 1e-16
        F_err_map = hdulist[f"{eline}_ERR"].data[0] * 1e-16 if hdulist[f"{eline}_ERR"].data.ndim == 3 else hdulist[f"{eline}_ERR"].data * 1e-16
        N_y, N_x = F_map.shape

        # Take a S/N cut 
        F_map[F_map / F_err_map < 3] = np.nan

        # Compute the extinction at the wavelengths of Halpha and Hbeta
        A_map = np.zeros((N_y, N_x))
        for yy, xx in product(range(N_y), range(N_x)):
            A_map[yy, xx] = extinction.fm07(np.array([eline_lambdas_A[eline]]), a_v=A_V_map[yy, xx])[0] if ~np.isnan(A_V_map[yy, xx]) else 0
 
        # Apply the extinction correction
        F_ext_corr_map = F_map * 10**(0.4 * A_map)

        # Convert to erg s^-1
        D_A_Mpc, D_L_Mpc = get_dist(z=hdulist[0].header["Z"])
        L_ext_corr_map = F_ext_corr_map * 4 * np.pi * (D_L_Mpc * 1e6 * 3.086e18)**2

        # Compute total Hbeta luminosity
        L_ext_corr_tot = np.nansum(L_ext_corr_map)

        # Store in a DataFrame for reference 
        df.loc[gal, f"L_{eline} (total)"] = L_ext_corr_tot
        df.loc[gal, f"log L_{eline} (total)"] = np.log10(L_ext_corr_tot)

# Plot a histogram of the Hbeta luminosities
plt.hist(df["log L_HBETA (total)"], bins=20, label=r"H$\beta$")
plt.hist(df["log L_OIII5007 (total)"], bins=20, range=(38, 44), label=r"$\rm[O\,III]$")
plt.xlabel(r"$\log_{10} \left( L [\rm erg\,s^{-1}]\right)$")
plt.ylabel(r"$N$")
plt.legend()
plt.grid()

df.to_csv("s7_total_line_fluxes.csv")
