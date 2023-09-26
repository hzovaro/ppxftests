#!/bin/bash
#PBS -N re1set9
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py RE1 ESO362-G08
python run_ppxf_s7.py RE1 FAIRALL49
python run_ppxf_s7.py RE1 IC2560
python run_ppxf_s7.py RE1 IC4329A
python run_ppxf_s7.py RE1 IC4995