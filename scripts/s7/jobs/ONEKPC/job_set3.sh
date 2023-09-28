#!/bin/bash
#PBS -N 1kpcset3
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py ONEKPC IC1657
python run_ppxf_s7.py ONEKPC IC1816
python run_ppxf_s7.py ONEKPC IC4777
python run_ppxf_s7.py ONEKPC MCG-03-34-064 