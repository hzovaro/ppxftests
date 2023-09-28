#!/bin/bash
#PBS -N 4asset10
#PBS -q smallmem
#PBS -M henry.zovaro@anu.edu.au
#PBS -m abe
#PBS -l ncpus=56
#PBS -k oe

cd /home/u5708159/python/Modules/ppxftests/scripts/s7

python run_ppxf_s7.py FOURAS IC5063
python run_ppxf_s7.py FOURAS IRAS11215-2806
python run_ppxf_s7.py FOURAS MARK573
python run_ppxf_s7.py FOURAS NGC2992
python run_ppxf_s7.py FOURAS NGC4939