#!/bin/bash
#PBS -N re1set2
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py RE1 NGC6860 
python run_ppxf_s7.py RE1 NGC5427 
python run_ppxf_s7.py RE1 ESO339-G11 
python run_ppxf_s7.py RE1 ESO460-G09 
python run_ppxf_s7.py RE1 ESO565-G19 