#!/bin/bash
#PBS -N 1kpcset4
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py ONEKPC MCG-06-23-038
python run_ppxf_s7.py ONEKPC NGC1125
python run_ppxf_s7.py ONEKPC NGC1667
python run_ppxf_s7.py ONEKPC NGC3858
python run_ppxf_s7.py ONEKPC NGC4845 