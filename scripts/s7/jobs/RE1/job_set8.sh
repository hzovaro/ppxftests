#!/bin/bash
#PBS -N re1set8
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py RE1 NGC5128
python run_ppxf_s7.py RE1 ESO420-G13
python run_ppxf_s7.py RE1 NGC1068
python run_ppxf_s7.py RE1 ESO103-G35
python run_ppxf_s7.py RE1 ESO138-G01