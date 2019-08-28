#!/bin/bash

source /home/zhangb/.bashrc
conda activate automl
# python bo-train.py --weighted_sample --cutout --batch_size 256 --report_freq 100

# ARCH="PC_DARTS_K49_"

#python bo-train.py --weighted_sample --cutout --batch_size 256 --report_freq 100 --arch PC_DARTS_K49_SIMPLE_CNN
#python bo-train.py --weighted_sample --cutout --batch_size 256 --report_freq 100 --arch PC_DARTS_K49_TS_20
python bo-train.py --weighted_sample --cutout --batch_size 256 --report_freq 100 --arch PC_DARTS_KMNIST_50
