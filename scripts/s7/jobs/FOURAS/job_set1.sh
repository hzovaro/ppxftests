#!/bin/bash
#PBS -N 4asset1
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py FOURAS PKS1306-241
python run_ppxf_s7.py FOURAS NGC7130
python run_ppxf_s7.py FOURAS NGC7590
python run_ppxf_s7.py FOURAS NGC6915
python run_ppxf_s7.py FOURAS NGC6890 