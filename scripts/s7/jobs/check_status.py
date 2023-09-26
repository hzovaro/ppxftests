# Given an aperture, print list of galaxies from gal_list.txt that have NOT been run yet 
import os
ap = "RE1"
s7_data_path = "/priv/meggs3/u5708159/S7/mar23/"
gals = [g.strip("\n") for g in open(os.path.join(s7_data_path, "gal_list.txt")).readlines()]
gals_complete = [g.split("s7_ppxf_")[1].split(f"_{ap}.hd5")[0] for g in os.listdir(os.path.join(s7_data_path, "ppxf")) if ap in g]
gals_to_run = [g for g in gals if g not in gals_complete]
