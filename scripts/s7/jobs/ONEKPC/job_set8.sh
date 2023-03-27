#!/bin/bash
#PBS -N 1kpcset8
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py ONEKPC NGC5128 ESO420-G13 NGC1068 ESO103-G35 ESO138-G01