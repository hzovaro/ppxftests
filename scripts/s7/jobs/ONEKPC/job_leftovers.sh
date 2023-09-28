#!/bin/bash
#PBS -N 1kpcleftovers
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py ONEKPC MCG-03-34-064
python run_ppxf_s7.py ONEKPC FAIRALL49
python run_ppxf_s7.py ONEKPC NGC6860
python run_ppxf_s7.py ONEKPC NGC424
python run_ppxf_s7.py ONEKPC ESO362-G18
python run_ppxf_s7.py ONEKPC NGC1667