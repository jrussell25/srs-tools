#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 15
#SBATCH -p serial_requeue,shared
#SBATCH --mem-per-cpu 4G
#SBATCH -o logs/zarr_%j.txt
#SBATCH -e logs/zarr_%j.txt

python setup_zarr.py $tiffpath $zarrpath

sbr="$(sbatch --array=0-4 segment_track.sh)"

if [[ "$sbr" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    jid="${BASH_REMATCH[1]}"
else
    echo "segmentation sbatch failed"
    exit 1
fi

sbatch --dependency=afterany:$jid sc_analysis.sh
