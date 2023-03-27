#!/bin/bash
#PBS -N 1kpcset9
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py ONEKPC ESO362-G08 FAIRALL49 IC2560 IC4329A IC4995