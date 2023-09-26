#!/bin/bash
#PBS -N 4asset6
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py FOURAS NGC7679
python run_ppxf_s7.py FOURAS IC5169
python run_ppxf_s7.py FOURAS NGC7582
python run_ppxf_s7.py FOURAS ESO362-G18
python run_ppxf_s7.py FOURAS NGC1320 