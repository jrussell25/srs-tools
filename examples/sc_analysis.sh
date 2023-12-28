#!/bin/bash
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 30
#SBATCH -p serial_requeue,shared
#SBATCH --mem-per-cpu 4G
#SBATCH -o logs/sc_output_%j.txt
#SBATCH -e logs/sc_errors_%j.txt

python sc_analysis.py $zarrpath
