#!/bin/bash

source /home/zhangb/.bashrc
conda activate automl
# python bo-train.py --weighted_sample --cutout --batch_size 256 --report_freq 100

# ARCH="PC_DARTS_K49_"

#python bo-train.py --weighted_sample --cutout --batch_size 256 --report_freq 100 --arch PC_DARTS_K49_SIMPLE_CNN
#python bo-train.py --weighted_sample --cutout --batch_size 256 --report_freq 100 --arch PC_DARTS_K49_TS_20
#python bo-train.py --weighted_sample --cutout --batch_size 256 --report_freq 100 --arch PC_DARTS_KMNIST_50

python bo-train.py --weighted_sample --cutout --seed 0 --batch_size 256 --report_freq 100 --arch K49_50_0
python bo-train.py --weighted_sample --cutout --seed 1 --batch_size 256 --report_freq 100 --arch K49_50_1
python bo-train.py --weighted_sample --cutout --seed 2 --batch_size 256 --report_freq 100 --arch K49_50_1
#if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
#then
    #echo 1
#    python bo-train.py --weighted_sample --cutout --seed 0 --batch_size 256 --report_freq 100 --arch KMNIST_50_0
#elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]
#then
    #echo 2
#    python bo-train.py --weighted_sample --cutout --seed 1 --batch_size 256 --report_freq 100 --arch KMNIST_50_1
#else
    #echo 3
#    python bo-train.py --weighted_sample --cutout --seed 2 --batch_size 256 --report_freq 100 --arch KMNIST_50_2
#fi
