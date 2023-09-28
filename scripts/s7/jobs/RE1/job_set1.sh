#!/bin/bash
#PBS -N re1set1
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py RE1 PKS1306-241 
python run_ppxf_s7.py RE1 NGC7130 
python run_ppxf_s7.py RE1 NGC7590 
python run_ppxf_s7.py RE1 NGC6915 
python run_ppxf_s7.py RE1 NGC6890 