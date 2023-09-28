import os 
import numpy as np
import pandas as pd
from tqdm import tqdm

from settings import ages, ppxf_output_path, Aperture, gals_all

overwrite_dataframe = True

for ap in [Aperture.FOURAS, Aperture.ONEKPC, Aperture.RE1]:

    # Open the DataFrame that will contain ALL galaxies
    df_fname = f"s7_ppxf_{ap.name}.hd5"
    df_all = pd.DataFrame()

    # One by one, open the DataFrames containing results for individual galaxies
    for gg, gal in tqdm(enumerate(gals_all)):

        # Filenames
        df_ppxf_fname = f"s7_ppxf_{gal}_{ap.name}.hd5" 
        df_stekin_fname = f"s7_ppxf_kinematics_{gal}_{ap.name}.hd5" 

        # Open the dataFrames
        df_ppxf = pd.read_hdf(os.path.join(ppxf_output_path, df_ppxf_fname)).set_index("Galaxy")
        df_stekin = pd.read_hdf(os.path.join(ppxf_output_path, df_stekin_fname)).set_index("Galaxy")
        df_thisgal = df_ppxf.merge(df_stekin[[c for c in df_stekin if c not in df_ppxf]], left_index=True, right_index=True)

        # Compute LW/MW ages at all cutoff ages  
        for age in ages[1:]:
            age_idx = np.nanargmin(np.abs(ages - age))
            for col in ["MC mean", "MC error", "MC 50th percentile", "MC 16th percentile", "MC 84th percentile", "regularised"]:
                df_thisgal.loc[gal, f"MW age (<{age/1e6:.2f} Myr) ({col})"] =\
                    df_ppxf.loc[gal, f"Mass-weighted age vs. age cutoff ({col})"][age_idx - 1]
                df_thisgal.loc[gal, f"LW age (<{age/1e6:.2f} Myr) ({col})"] =\
                    df_ppxf.loc[gal, f"Light-weighted age vs. age cutoff ({col})"][age_idx - 1]
                # Also compute the cumulative light fraction
                df_thisgal.loc[gal, f"Cumulative light fraction (<{age/1e6:.2f} Myr) ({col})"] =\
                    df_ppxf.loc[gal, f"Cumulative light fraction vs. age cutoff ({col})"][age_idx - 1]

        # Append to main DataFrame
        df_all = df_all.append(df_thisgal)

    # Sort
    df_all = df_all.sort_index()
    df_all["Number"] = np.arange(len(gals_all))

    # Remove MC stellar kinematics so that they aren't accidentally used instead of the measurements from the VdS fit 
    df_all = df_all.drop(columns=[c for c in df_all if ("sigma_*" in c or "v_*" in c) and "VdS" not in c])

    # Save back to file
    if not overwrite_dataframe and os.path.exists(os.path.join(ppxf_output_path, df_fname)):
        k = input(f"File {os.path.join(ppxf_output_path, df_fname)} exists - are you sure you want to overwrite it? ")
        if k.lower().startswith("y"):
            df_all.to_hdf(os.path.join(ppxf_output_path, df_fname), key="S7")
        else:
            print("WARNING: not overwriting existing file!")
    else:
        df_all.to_hdf(os.path.join(ppxf_output_path, df_fname), key="S7")

    # Only save AGN contributions
    cols = ["x_AGN (total, regularised)", "x_AGN (total, MC mean)", "x_AGN (total, MC error)", "x_AGN (total, MC 16th percentile)", "x_AGN (total, MC 50th percentile)", "x_AGN (total, MC 84th percentile)",]
    df_agn = df_all[cols].copy()
    df_agn.to_csv(os.path.join(ppxf_output_path, f"s7_ppxf_{ap.name}_xAGN.csv"))

    # Only save A_V values
    cols = ["A_V (regularised)", "A_V (MC mean)", "A_V (MC error)", "A_V (MC 16th percentile)", "A_V (MC 50th percentile)", "A_V (MC 84th percentile)",]
    df_av = df_all[cols].copy()
    df_av.to_csv(os.path.join(ppxf_output_path, f"s7_ppxf_{ap.name}_A_V.csv"))

    # Only save ages
    cols = [c for c in df_all if ("1000.0" in c or "100.0" in c) and "age" in c]
    cols += [c for c in df_all if "17783.0" in c and "age" in c]
    df_ages = df_all[cols].copy()
    df_ages = df_ages.rename(columns=dict(zip(cols, [c.replace(" (<17783.00 Myr)", "") + " (log yr)" for c in cols])))
    df_ages.to_csv(os.path.join(ppxf_output_path, f"s7_ppxf_{ap.name}_ages.csv"))

    # Only save kinematics
    cols = [c for c in df_all if "VdS" in c]
    df_stekin = df_all[cols].copy()
    df_stekin.to_csv(os.path.join(ppxf_output_path, f"s7_ppxf_{ap.name}_kinematics.csv"))
