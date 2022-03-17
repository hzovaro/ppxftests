import os 
import pandas as pd

"""
In run_ppxf_s7.py, we saved the results from each galaxy to their own DataFrame.
However, in a previous versino of the script, we saved 21 galaxies ino a single 
DataFrame, called s7_ppxf.hd5. 
Here, we append the results from those galaxies with individual .hd5 files
to s7_ppxf.hd5, so that all galaxies can be stored in a single file.
"""

s7_data_path = "/priv/meggs3/u5708159/S7/ppxf"

# Open the DataFrame that will contain ALL galaxies
df_all = pd.read_hdf(os.path.join(s7_data_path, "s7_ppxf.hd5"))

# One by one, open the DataFrames containing results for individual galaxies
df_fnames = [f for f in os.listdir(s7_data_path) 
             if f.endswith(".hd5")
             and f.startswith("s7_ppxf_")
             and "DEBUG" not in f]

for df_fname in df_fnames:
    # Open the dataFrame
    df_gal = pd.read_hdf(os.path.join(s7_data_path, df_fname))
    df_all = df_all.append(df_gal, ignore_index=True)

# Save back to file
df_all.to_hdf(os.path.join(s7_data_path, "s7_ppxf.hd5"), key="s7")