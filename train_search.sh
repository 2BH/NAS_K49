#!/bin/bash
source /home/zhangb/.bashrc
conda activate automl
# sbatch -p meta_gpu-black train_search.sh
# 2761719 with adamW
# python train_search.py --set K49 --batch_size 128 --weighted_sample --cutout --epochs 20
# python train.py --cutout --set K49 --epochs 20 --weighted_sample --drop_path_prob 0 --seed 0 --optimizer Adam
# python train.py --cutout --set K49 --epochs 20 --weighted_sample --drop_path_prob 0 --seed 2 --optimizer Adam
# python train_search.py --cutout --set KMNIST --epochs 50 --seed 0 --batch_size 128
python warmstart_search.py --cutout --batch_size 128 --weighted_sample 

# python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.3422406656613465 --drop_path_prob 0.26054605565409283 --init_channel 23 --learning_rate 0.005786678417786406 --layers 10 --optimizer AdamW --weight_decay 0.00017014059877199961