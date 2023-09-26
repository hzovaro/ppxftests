#!/bin/bash
#PBS -N re1set5
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py RE1 NGC424
python run_ppxf_s7.py RE1 NGC7469
python run_ppxf_s7.py RE1 MCG-02-51-008
python run_ppxf_s7.py RE1 NGC5728 