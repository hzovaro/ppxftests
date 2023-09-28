#!/bin/bash
#PBS -N re1set6
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py RE1 NGC7679
python run_ppxf_s7.py RE1 IC5169
python run_ppxf_s7.py RE1 NGC7582
python run_ppxf_s7.py RE1 ESO362-G18
python run_ppxf_s7.py RE1 NGC1320 