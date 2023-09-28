#!/bin/bash
#PBS -N re1leftover
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py RE1 MCG-03-34-064
python run_ppxf_s7.py RE1 FAIRALL49
python run_ppxf_s7.py RE1 NGC6860
python run_ppxf_s7.py RE1 NGC424
python run_ppxf_s7.py RE1 ESO362-G18
python run_ppxf_s7.py RE1 NGC1667