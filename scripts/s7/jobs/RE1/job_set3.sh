#!/bin/bash
#PBS -N ppxfset3
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7
python run_ppxf_s7.py RE1 NGC3858 NGC4845 NGC424 NGC7469 MCG-02-51-008 ESO500-G34 NGC5728 NGC7679 IC5169 