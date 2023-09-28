#!/bin/bash
#PBS -N 4asset11
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py FOURAS NGC4968
python run_ppxf_s7.py FOURAS NGC6300
python run_ppxf_s7.py FOURAS NGC7682