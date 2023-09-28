#!/bin/bash
#PBS -N re1set7
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py RE1 NGC1365
python run_ppxf_s7.py RE1 NGC3393
python run_ppxf_s7.py RE1 NGC4507
python run_ppxf_s7.py RE1 NGC613
python run_ppxf_s7.py RE1 NGC5664 