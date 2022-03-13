#!/bin/bash

cd  /root/ocrtoc_ws/src/ensemble_matching/mmdetection

python3 ensemble_object_matching_final.py --scene_image=$1 --target_images_list=$2 --target_image_dir_path=$3