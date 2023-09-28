#!/bin/bash
#PBS -N 1kpcset1
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py ONEKPC PKS1306-241
python run_ppxf_s7.py ONEKPC NGC7130
python run_ppxf_s7.py ONEKPC NGC7590
python run_ppxf_s7.py ONEKPC NGC6915
python run_ppxf_s7.py ONEKPC NGC6890 