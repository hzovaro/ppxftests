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
s7_dir = os.getenv("S7_DIR")
fnames = [f for f in os.listdir(os.path.join(s7_dir, "2_Post-processed_mergecomps")) if not f.startswith("._")]
gals = [f.split("_best_components.fits")[0] for f in fnames]

df = pd.DataFrame(index=gals)

for gal, fname in zip(gals, fnames):
    hdulist = fits.open(os.path.join(s7_dir, "2_Post-processed_mergecomps", fname))

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

    # Compute the extinction at the wavelengths of Halpha and Hbeta
    A_Ha_map = np.zeros((N_y, N_x))
    A_Hb_map = np.zeros((N_y, N_x))
    for yy, xx in product(range(N_y), range(N_x)):
        A_Ha_map[yy, xx] = extinction.fm07(np.array([6562.800]), a_v=A_V_map[yy, xx])[0] if ~np.isnan(A_V_map[yy, xx]) else 0
        A_Hb_map[yy, xx] = extinction.fm07(np.array([4861.325]), a_v=A_V_map[yy, xx])[0] if ~np.isnan(A_V_map[yy, xx]) else 0

    # Apply the extinction correction
    F_Ha_ext_corr_map = F_Ha_map * 10**(0.4 * A_Ha_map)
    F_Hb_ext_corr_map = F_Hb_map * 10**(0.4 * A_Hb_map)

    # Convert to erg s^-1
    D_A_Mpc, D_L_Mpc = get_dist(z=hdulist[0].header["Z"])
    L_Ha_ext_corr_map = F_Ha_ext_corr_map * 4 * np.pi * (D_L_Mpc * 1e6 * 3.086e18)**2
    L_Hb_ext_corr_map = F_Hb_ext_corr_map * 4 * np.pi * (D_L_Mpc * 1e6 * 3.086e18)**2

    # Compute total Hbeta luminosity
    L_Ha_ext_corr_tot = np.nansum(L_Ha_ext_corr_map)
    L_Hb_ext_corr_tot = np.nansum(L_Hb_ext_corr_map)
    # print(f"{gal}: {np.log10(L_Hb_ext_corr_tot):.3f}")

    # Tracer()() if gal == "NGC1097" else None

    # Store in a DataFrame for reference 
    df.loc[gal, "L_Hb (total)"] = L_Hb_ext_corr_tot
    df.loc[gal, "L_Ha (total)"] = L_Ha_ext_corr_tot
    df.loc[gal, "log L_Hb (total)"] = np.log10(L_Hb_ext_corr_tot)
    df.loc[gal, "log L_Ha (total)"] = np.log10(L_Ha_ext_corr_tot)

    # Tracer()() if gal == "NGC1097" else None

# Plot a histogram of the Hbeta luminosities
plt.hist(df["log L_Hb (total)"], bins=20)
plt.xlabel(r"$\log_{10} \left( L_{\rm H\beta} [\rm erg\,s^{-1}]\right)$")
plt.ylabel(r"$N$")
plt.grid()


