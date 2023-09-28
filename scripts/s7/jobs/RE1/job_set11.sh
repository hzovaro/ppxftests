#!/bin/bash
#PBS -N re1set11
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py RE1 NGC4968
python run_ppxf_s7.py RE1 NGC6300
python run_ppxf_s7.py RE1 NGC7682