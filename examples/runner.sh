#!/bin/bash

micro_path=/n/hekstra_lab/projects/microscopy

while read -ra arr f;
do
    export tiffpath=$micro_path/${arr[0]}
    export zarrpath=$micro_path/${arr[1]}

    echo tiff path: $tiffpath
    echo zarr path: $zarrpath
    echo "Found tiffs: $(ls -l $tiffpath*.tif | wc -l)"

    sbatch preprocess.sh

done < file_list.txt;
