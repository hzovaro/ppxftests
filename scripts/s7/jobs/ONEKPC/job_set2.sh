#!/bin/bash
#PBS -N 1kpcset2
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py ONEKPC NGC6860
python run_ppxf_s7.py ONEKPC NGC5427
python run_ppxf_s7.py ONEKPC ESO339-G11
python run_ppxf_s7.py ONEKPC ESO460-G09
python run_ppxf_s7.py ONEKPC ESO565-G19 