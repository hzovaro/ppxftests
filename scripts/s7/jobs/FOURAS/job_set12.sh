#!/bin/bash
#PBS -N 4asset12
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py FOURAS NGC5506
python run_ppxf_s7.py FOURAS ESO500-G34
python run_ppxf_s7.py FOURAS MCG-01-24-012
python run_ppxf_s7.py FOURAS NGC1667