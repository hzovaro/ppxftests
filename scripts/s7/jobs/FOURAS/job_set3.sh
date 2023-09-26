#!/bin/bash
#PBS -N 4asset3
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py FOURAS IC1657
python run_ppxf_s7.py FOURAS IC1816
python run_ppxf_s7.py FOURAS IC4777
python run_ppxf_s7.py FOURAS MCG-03-34-064 