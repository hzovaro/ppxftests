#!/bin/bash
#PBS -N 1kpcset4
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py ONEKPC MCG-06-23-038 NGC1125 NGC1667 NGC3858 NGC4845 