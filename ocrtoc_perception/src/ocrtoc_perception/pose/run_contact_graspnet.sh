#!/bin/bash

source /root/miniconda3/bin/activate contact_graspnet_env

cd  /root/ocrtoc_ws/src/contact_graspnet/

python contact_graspnet/inference.py --np_path=$1 --pcd_colors=$2 --ocrtoc_save=$3
