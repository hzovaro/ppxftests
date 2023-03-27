#!/bin/bash
#PBS -N ppxfset2
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py RE1 MCG-03-34-064 MCG-06-23-038 NGC1125 NGC1667