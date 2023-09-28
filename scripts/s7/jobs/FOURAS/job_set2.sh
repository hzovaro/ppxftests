#!/bin/bash
#PBS -N 4asset2
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py FOURAS NGC6860
python run_ppxf_s7.py FOURAS NGC5427
python run_ppxf_s7.py FOURAS ESO339-G11
python run_ppxf_s7.py FOURAS ESO460-G09
python run_ppxf_s7.py FOURAS ESO565-G19 