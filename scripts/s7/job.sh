#!/bin/bash
#PBS -N ppxfset5
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts
python run_ppxf_s7.py IC4995 IC5063 IRAS11215-2806