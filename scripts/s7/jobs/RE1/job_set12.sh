#!/bin/bash
#PBS -N re1set12
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py RE1 NGC5506
python run_ppxf_s7.py RE1 ESO500-G34
python run_ppxf_s7.py RE1 MCG-01-24-012
python run_ppxf_s7.py RE1 NGC1667