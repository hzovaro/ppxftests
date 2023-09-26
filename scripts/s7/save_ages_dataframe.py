import os
import numpy as np
import pandas as pd
from ppxftests.ssputils import load_ssp_templates

s7_data_path = "/priv/meggs3/u5708159/S7/mar23/ppxf"

for ap in ["FOURAS", "ONEKPC", "RE1"]:

    ###############################################################################
    # Load the DataFrame
    ###############################################################################
    # Load the DataFrame containing all S7 galaxies
    df_fname = f"s7_ppxf_{ap}.hd5"
    df_all = pd.read_hdf(os.path.join(s7_data_path, df_fname), key="s7")

    gals = df_all.index.unique()
    gals = gals.sort_values()

    # Get the age & metallicity dimensions
    _, _, _, ages = load_ssp_templates("Padova")

    ###############################################################################
    # Load the DataFrame from each galaxy & store the relevant values in a single 
    # big DataFrame
    ###############################################################################
    df = pd.DataFrame(index=gals)
    df.index.name = "Galaxy"

    for gg, gal in enumerate(gals):
        # Open the DataFrame
        df_gal = df_all[df_all.index == gal]

        # Extract x_AGN and A_V measurements
        df.loc[gal, "x_AGN (total, regularised)"] = df_gal["x_AGN (total, regularised)"].values[0]
        df.loc[gal, "x_AGN (total, MC mean)"] = df_gal["x_AGN (total, MC mean)"].values[0]
        df.loc[gal, "x_AGN (total, MC error)"] = df_gal["x_AGN (total, MC error)"].values[0]
        df.loc[gal, "x_AGN (total, MC 16th percentile)"] = df_gal["x_AGN (total, MC 16th percentile)"].values[0]
        df.loc[gal, "x_AGN (total, MC 50th percentile)"] = df_gal["x_AGN (total, MC 50th percentile)"].values[0]
        df.loc[gal, "x_AGN (total, MC 84th percentile)"] = df_gal["x_AGN (total, MC 84th percentile)"].values[0]
        df.loc[gal, "A_V (regularised)"] = df_gal["A_V (regularised)"].values[0]
        df.loc[gal, "A_V (MC mean)"] = df_gal["A_V (MC mean)"].values[0]
        df.loc[gal, "A_V (MC error)"] = df_gal["A_V (MC error)"].values[0]
        df.loc[gal, "A_V (MC 16th percentile)"] = df_gal["A_V (MC 16th percentile)"].values[0]
        df.loc[gal, "A_V (MC 50th percentile)"] = df_gal["A_V (MC 50th percentile)"].values[0]
        df.loc[gal, "A_V (MC 84th percentile)"] = df_gal["A_V (MC 84th percentile)"].values[0]
        df.loc[gal, "Number"] = gg

        # Extract the age index at the appropriate time 
        for age in ages[1:]:
            age_idx = np.nanargmin(np.abs(ages - age))
            for col in ["MC mean", "MC error", "MC 50th percentile", "MC 16th percentile", "MC 84th percentile", "regularised"]:
                df.loc[gal, f"MW age (<{age/1e6:.2f} Myr) ({col})"] =\
                    df_gal[f"Mass-weighted age vs. age cutoff ({col})"].values.item()[age_idx - 1]
                df.loc[gal, f"LW age (<{age/1e6:.2f} Myr) ({col})"] =\
                    df_gal[f"Light-weighted age vs. age cutoff ({col})"].values.item()[age_idx - 1]
                # Also compute the cumulative light fraction
                df.loc[gal, f"Cumulative light fraction (<{age/1e6:.2f} Myr) ({col})"] =\
                    df_gal[f"Cumulative light fraction vs. age cutoff ({col})"].values.item()[age_idx - 1]
        df.loc[gal, "Number"] = gg

    ###############################################################################
    # Save the "ages" DataFrame to file
    ###############################################################################
    # Save the full version
    df.to_hdf(os.path.join(s7_data_path, f"s7_ppxf_{ap}_measurements_full_version.hd5"), key="S7 measurements")

    # Only save a select number of age values to .csv for Aman
    cols = [c for c in df if ("1000.0" in c or "100.0" in c) and "age" in c]
    cols += [c for c in df if "17783.0" in c and "age" in c]
    df_ages = df[cols]
    df_ages = df_ages.rename(columns=dict(zip(cols, [c.replace(" (<17783.00 Myr)", "") + " (log yr)" for c in cols])))
    df_ages.to_csv(os.path.join(s7_data_path, f"s7_ppxf_{ap}_ages.csv"))
