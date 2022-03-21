#!/bin/bash

source /root/miniconda3/bin/activate stack_detection
echo "Conda environment activated"
python /root/ocrtoc_ws/src/stack_detection/stack_detect.py --object_dict_path=$1 --mesh_dir=$2 --save_path=$3
echo "Python script ran"
