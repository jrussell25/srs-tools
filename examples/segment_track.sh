#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 15
#SBATCH -p gpu_requeue,gpu
#SBATCH --mem-per-cpu 4G
#SBATCH -o logs/st_output_%j_%a.txt
#SBATCH -e logs/st_errors_%j_%a.txt
#SBATCH --gres gpu:1
#SBATCH --constraint v100

#python preprocess.py /n/hekstra_lab/projects/microscopy/2023-12-06/Sequence/ /n/hekstra_lab/projects/microscopy/2023-12-06.zarr "${SLURM_ARRAY_TASK_ID}"
python segment_track.py $zarrpath "${SLURM_ARRAY_TASK_ID}"

# To run on a full dataset:
# sbatch --array=0-Number_of_scenes preprocess.sh
