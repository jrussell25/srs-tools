#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 15
#SBATCH -p serial_requeue,shared
#SBATCH --mem-per-cpu 4G
#SBATCH -o logs/output_zarr.txt
#SBATCH -e logs/errors_zarr.txt

tiffpath=/n/hekstra_lab/projects/microscopy/2023-12-12-timelapse/Sequence/

export zarrpath=/n/hekstra_lab/projects/microscopy/2023-12-12-timelapse.zarr

python setup_zarr.py $tiffpath $zarrpath

sbatch --array=0-4 segment_track.sh
