#!/bin/bash
#PBS -N re1set10
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py RE1 IC5063
python run_ppxf_s7.py RE1 IRAS11215-2806
python run_ppxf_s7.py RE1 MARK573
python run_ppxf_s7.py RE1 NGC2992
python run_ppxf_s7.py RE1 NGC4939