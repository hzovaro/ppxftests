#!/bin/bash
#PBS -N 1kpcset7
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py ONEKPC NGC1365
python run_ppxf_s7.py ONEKPC NGC3393
python run_ppxf_s7.py ONEKPC NGC4507
python run_ppxf_s7.py ONEKPC NGC613
python run_ppxf_s7.py ONEKPC NGC5664 