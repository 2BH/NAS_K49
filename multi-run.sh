#!/bin/bash
source /home/zhangb/.bashrc
conda activate automl

# BASELINE
# python main_baseline.py --set K49 --cutout --seed 0 --optimizer adam
# python main_baseline.py --set K49 --cutout --seed 1 --optimizer adam
#python main_baseline.py --set K49 --cutout --seed 2 --optimizer adam

python main_baseline.py --set K49 --seed 0  --optimizer adam
python main_baseline.py --set K49 --seed 1  --optimizer adam
python main_baseline.py --set K49 --seed 2  --optimizer adam

#python main_baseline.py --set K49 --cutout --seed 0 --weighted_sample --optimizer adam
#python main_baseline.py --set K49 --cutout --seed 1 --weighted_sample --optimizer adam
#python main_baseline.py --set K49 --cutout --seed 2 --weighted_sample --optimizer adam

### Random Candidates

#python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.2526668075463563 --drop_path_prob 0.2144843306364046 --init_channel 17 --learning_rate 0.014355517707684132 --layers 6 --optimizer AdamW --weight_decay 0.0018265960314980333 --arch PC_DARTS_K49_9
#python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.3376894312121811 --drop_path_prob 0.2741225421278378 --init_channel 22 --learning_rate 0.008647567347478456 --layers 10 --optimizer AdamW --weight_decay 0.0004329502066179016 --arch PC_DARTS_K49_6
#python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.36268203611078165 --drop_path_prob 0.3141718650257615 --init_channel 23 --learning_rate 0.019446277337616755 --layers 7 --optimizer AdamW --weight_decay 0.0003991119201316201 --arch PC_DARTS_K49_2

#python train.py --cutout --set K49 --weighted_sample --arch PCDARTS --seed 0
#python train.py --cutout --set K49 --weighted_sample --arch PCDARTS --seed 1
#python train.py --cutout --set K49 --weighted_sample --arch PCDARTS --seed 2

# python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.4243634749901255 --drop_path_prob 0.3244774363108105 --init_channel 21 --learning_rate 0.006023611560175412 --layers 10 --optimizer AdamW --weight_decay 0.0006414577136514433 --arch PC_DARTS_K49_17
# python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.41573618371273047 --drop_path_prob 0.3438951137821758 --init_channel 20 --learning_rate 0.009570012712227077 --layers 9 --optimizer AdamW --weight_decay 0.00010923484407269452 --arch PC_DARTS_K49_18
# python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.39824050254011 --drop_path_prob 0.39835924879698514 --init_channel 20 --learning_rate 0.008085394472293925 --layers 8 --optimizer AdamW --weight_decay 0.00022220521551917044 --arch PC_DARTS_K49_19

