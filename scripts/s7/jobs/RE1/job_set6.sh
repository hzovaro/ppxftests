#!/bin/bash
#PBS -N ppxfset6
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py RE1 IC5063 IRAS11215-2806 MARK573 NGC2992 NGC4939 NGC4968