#!/bin/bash
#PBS -N ppxfset4
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py RE1 NGC7582 ESO362-G18 NGC1320 NGC1365 NGC3393 NGC4507 NGC613 NGC5664 NGC5128