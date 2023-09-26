import os 
import pandas as pd

# Aperture type
from enum import Enum
class Aperture(Enum):
    RE1 = 0
    FOURAS = 1
    SDSS = 2
    ONEKPC = 3

"""
In run_ppxf_s7.py, we saved the results from each galaxy to their own DataFrame.
However, in a previous versino of the script, we saved 21 galaxies ino a single 
DataFrame, called s7_ppxf.hd5. 
Here, we append the results from those galaxies with individual .hd5 files
to s7_ppxf.hd5, so that all galaxies can be stored in a single file.
"""

s7_data_path = "/priv/meggs3/u5708159/S7/mar23/ppxf"
debug = False
overwrite_dataframe = True

for ap in [Aperture.FOURAS, Aperture.ONEKPC, Aperture.RE1]:

    # Open the DataFrame that will contain ALL galaxies
    df_fname = f"s7_ppxf_{ap.name}_kinematics"
    df_all = pd.DataFrame(columns=["Galaxy"])

    # One by one, open the DataFrames containing results for individual galaxies
    df_gal_fnames = [f for f in os.listdir(s7_data_path) 
                if f.endswith(".hd5")
                and f.startswith("s7_ppxf_")
                and ap.name in f
                and "kinematics" in f
                and f != f"{df_fname}.csv"
                and (("DEBUG" in f) if debug else ("DEBUG" not in f))]

    for df_gal_fname in df_gal_fnames:
        # Open the dataFrame
        df_gal = pd.read_hdf(os.path.join(s7_data_path, df_gal_fname))
        
        # Get the name of the galaxy
        gal = df_gal["Galaxy"].unique()[0]

        # Only add this DataFrame if the galaxy is not already present.
        if gal not in df_all["Galaxy"].unique():
            df_all = df_all.append(df_gal, ignore_index=True)

    # Save back to file
    df_all = df_all.set_index("Galaxy")
    if os.path.exists(os.path.join(s7_data_path, df_fname)):
        k = input(f"File {os.path.join(s7_data_path, df_fname)} exists - are you sure you want to overwrite it?")
        if k.lower().startswith("y"):
            df_all.to_hdf(os.path.join(s7_data_path, f"{df_fname}.hd5"), key="s7")
            df_all.to_csv(os.path.join(s7_data_path, f"{df_fname}.csv"))
        else:
            print("WARNING: not overwriting existing file!")
    else:
        df_all.to_hdf(os.path.join(s7_data_path, f"{df_fname}.hd5"), key="s7")
        df_all.to_csv(os.path.join(s7_data_path, f"{df_fname}.csv"))
