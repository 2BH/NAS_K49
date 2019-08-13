#!/bin/bash
source /home/zhangb/.bashrc
conda activate automl
python train_search.py --set K49 --batch_size 128
