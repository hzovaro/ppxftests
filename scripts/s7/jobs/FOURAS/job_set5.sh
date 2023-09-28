#!/bin/bash
#PBS -N 4asset5
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py FOURAS NGC424
python run_ppxf_s7.py FOURAS NGC7469
python run_ppxf_s7.py FOURAS MCG-02-51-008
python run_ppxf_s7.py FOURAS NGC5728 