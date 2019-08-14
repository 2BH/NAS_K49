#!/bin/bash
source /home/zhangb/.bashrc
conda activate automl
# sbatch -p meta_gpu-black train_search.sh
# 2761719 with adamW
# python train_search.py --set K49 --batch_size 128 --weighted_sample --cutout --epochs 45
python train.py --cutout --auxiliary --set K49 --epochs 20 --weighted_sample 