#!/bin/bash
#PBS -N 1kpcset1
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py ONEKPC PKS1306-241 NGC7130 NGC7590 NGC6915 NGC6890 