#!/bin/bash
source /home/zhangb/.bashrc
conda activate automl
# sbatch -p meta_gpu-black train_search.sh
# 2761719 with adamW
# python train_search.py --set K49 --batch_size 128 --weighted_sample --cutout --epochs 20
# python train.py --cutout --set K49 --epochs 20 --weighted_sample --drop_path_prob 0 --seed 0 --optimizer Adam
# python train.py --cutout --set K49 --epochs 20 --weighted_sample --drop_path_prob 0 --seed 2 --optimizer Adam
#python train_search.py --cutout --set KMNIST --epochs 50 --seed 1 --batch_size 128
#python train_search.py --cutout --set KMNIST --epochs 50 --seed 1 --batch_size 128
#if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
#then
#    python train_search.py --cutout --set K49 --weighted_sample --epochs 50 --seed 0 --batch_size 128
#elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]
#then
#    python train_search.py --cutout --set K49 --weighted_sample --epochs 50 --seed 1 --batch_size 128
#else
#    #echo 3
#    python train_search.py --cutout --set K49 --weighted_sample --epochs 50 --seed 2 --batch_size 128
#fi

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
then
    python train_search_darts.py --cutout --set K49 --weighted_sample --epochs 20 --seed 0 --batch_size 32
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]
then
    python train_search_darts.py --cutout --set K49 --weighted_sample --epochs 20 --seed 1 --batch_size 32
else
    #echo 3
    python train_search_darts.py --cutout --set K49 --weighted_sample --epochs 20 --seed 2 --batch_size 32
fi

#DIR1=exp_2019-08-30T17:21:54.305142
#DIR2=exp_2019-08-30T17:21:54.305512
#DIR3=exp_2019-08-30T17:21:54.317886
#if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
#then
#    python warmstart_search.py --cutout --seed 0 --batch_size 128 --weighted_sample --model_dir $DIR1
#elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]
#then
#    python warmstart_search.py --cutout --seed 1 --batch_size 128 --weighted_sample --model_dir $DIR2
#else
    #echo 3
#    python warmstart_search.py --cutout --seed 2 --batch_size 128 --weighted_sample --model_dir $DIR3
#fi
#python warmstart_search.py --cutout --batch_size 128 --weighted_sample 

# python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.3422406656613465 --drop_path_prob 0.26054605565409283 --init_channel 23 --learning_rate 0.005786678417786406 --layers 10 --optimizer AdamW --weight_decay 0.00017014059877199961