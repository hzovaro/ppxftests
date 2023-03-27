#!/bin/bash
#PBS -N ngc3100
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/NGC3100/python
python run_ppxf_s7.py