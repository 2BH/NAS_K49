#!/bin/bash
source /home/zhangb/.bashrc
conda activate automl

# BASELINE
# python main_baseline.py --set K49 --cutout --seed 0 --optimizer adam
# python main_baseline.py --set K49 --cutout --seed 1 --optimizer adam
#python main_baseline.py --set K49 --cutout --seed 2 --optimizer adam

#python main_baseline.py --set K49 --seed 0  --optimizer adam
#python main_baseline.py --set K49 --seed 1  --optimizer adam
#python main_baseline.py --set K49 --seed 2  --optimizer adam

#python main_baseline.py --set K49 --cutout --seed 0 --weighted_sample --optimizer adam
#python main_baseline.py --set K49 --cutout --seed 1 --weighted_sample --optimizer adam
#python main_baseline.py --set K49 --cutout --seed 2 --weighted_sample --optimizer adam

### Random Candidates

#python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.2526668075463563 --drop_path_prob 0.2144843306364046 --init_channel 17 --learning_rate 0.014355517707684132 --layers 6 --optimizer AdamW --weight_decay 0.0018265960314980333 --arch PC_DARTS_K49_9
#python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.3376894312121811 --drop_path_prob 0.2741225421278378 --init_channel 22 --learning_rate 0.008647567347478456 --layers 10 --optimizer AdamW --weight_decay 0.0004329502066179016 --arch PC_DARTS_K49_6
#python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.36268203611078165 --drop_path_prob 0.3141718650257615 --init_channel 23 --learning_rate 0.019446277337616755 --layers 7 --optimizer AdamW --weight_decay 0.0003991119201316201 --arch PC_DARTS_K49_2


#if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
#then
#python train.py --cutout --set K49 --weighted_sample --arch PC_DARTS_K49_SIMPLE_CNN --seed $SLURM_ARRAY_TASK_ID --epochs 100
#else
#   python bo-mbexp.py -run_id $SLURM_ARRAY_JOB_ID -worker -env $ENV -logdir $DIR -worker_id $SLURM_ARRAY_TASK_ID -interface $INTERFACE -o exp_cfg.log_cfg.neval 5 -o exp_cfg.bo_cfg.min_budget 6 -o exp_cfg.bo_cfg.max_budget 30 -o exp_cfg.bo_cfg.nopt_iter 100
#fi
#python train.py --cutout --set K49 --weighted_sample --arch PC_DARTS_K49_TS_20 --seed 0 --epochs 100
#python train.py --cutout --set K49 --weighted_sample --arch PC_DARTS_K49_TS_20 --seed 1 --epochs 100
#python train.py --cutout --set K49 --weighted_sample --arch PC_DARTS_K49_TS_20 --seed 2 --epochs 100

#python train.py --cutout --set K49 --weighted_sample --arch PC_DARTS_K49_SIMPLE_CNN --seed 0 --epochs 100
#python train.py --cutout --set K49 --weighted_sample --arch PC_DARTS_K49_SIMPLE_CNN --seed 1 --epochs 100
#python train.py --cutout --set K49 --weighted_sample --arch PC_DARTS_K49_SIMPLE_CNN --seed 2 --epochs 100

#python train.py --cutout --set K49 --seed 0 --auxiliary --weighted_sample --auxiliary_weight 0.36991891869362237 --drop_path_prob 0.20684068134688113 --init_channel 21 --learning_rate 0.02459323590273285 --layers 9 --optimizer AdamW --weight_decay 0.000154773933523847 --arch PC_DARTS_K49_SIMPLE_CNN
#python train.py --cutout --set K49 --seed 1 --auxiliary --weighted_sample --auxiliary_weight 0.36991891869362237 --drop_path_prob 0.20684068134688113 --init_channel 21 --learning_rate 0.02459323590273285 --layers 9 --optimizer AdamW --weight_decay 0.000154773933523847 --arch PC_DARTS_K49_SIMPLE_CNN
#python train.py --cutout --set K49 --seed 2 --auxiliary --weighted_sample --auxiliary_weight 0.36991891869362237 --drop_path_prob 0.20684068134688113 --init_channel 21 --learning_rate 0.02459323590273285 --layers 9 --optimizer AdamW --weight_decay 0.000154773933523847 --arch PC_DARTS_K49_SIMPLE_CNN

#python train.py --cutout --set K49 --seed 0 --auxiliary --weighted_sample --auxiliary_weight 0.2877401829392393 --drop_path_prob 0.20825218032050513 --init_channel 24 --learning_rate 0.008124271128030202 --layers 9 --optimizer AdamW --weight_decay 0.00019934124166045357 --arch PC_DARTS_K49_TS_20
#python train.py --cutout --set K49 --seed 1 --auxiliary --weighted_sample --auxiliary_weight 0.2877401829392393 --drop_path_prob 0.20825218032050513 --init_channel 24 --learning_rate 0.008124271128030202 --layers 9 --optimizer AdamW --weight_decay 0.00019934124166045357 --arch PC_DARTS_K49_TS_20
#python train.py --cutout --set K49 --seed 2 --auxiliary --weighted_sample --auxiliary_weight 0.2877401829392393 --drop_path_prob 0.20825218032050513 --init_channel 24 --learning_rate 0.008124271128030202 --layers 9 --optimizer AdamW --weight_decay 0.00019934124166045357 --arch PC_DARTS_K49_TS_20

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
then
    #python main_default.py -s 0
    #echo 1
    #python main_baseline.py --set K49 --cutout --seed 0 --weighted_sample --optimizer adam
    #python train.py --cutout --seed 0 --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.2699479152819686 --drop_path_prob 0.2201176335588656 --init_channel 20 --learning_rate 0.014808908571806835 --layers 10 --optimizer AdamW --weight_decay 0.0003645065872059875 --arch K49_50_0
    python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.2107590792039467 --drop_path_prob 0.2746180588295253 --init_channel 36 --learning_rate 0.02429753098617262 --layers 20 --optimizer AdamW --weight_decay 0.0007598690386213507 --arch K49_TS_2 --batch_size 128
#    python train.py --cutout --set K49 --seed 0 --auxiliary --weighted_sample --init_channel 24 --layers 10 --optimizer Adam --arch K49_50_WARM_0
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]
then
    #python main_default.py -s 1 
    #python main_baseline.py --set K49 --cutout --seed 1 --weighted_sample --optimizer adam
    echo 2
    #python train.py --cutout --seed 1 --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.2500674879573223 --drop_path_prob 0.21685877714166632 --init_channel 18 --learning_rate 0.01707180872639491 --layers 10 --optimizer AdamW --weight_decay 0.00021046756223061478 --arch K49_50_1
    #python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.3504770073473448 --drop_path_prob 0.2719526514456682 --init_channel 24 --learning_rate 0.007481753869613876 --layers 10 --optimizer AdamW --weight_decay 0.00197268431489239 --arch K49_TS_1
#    python train.py --cutout --set K49 --seed 1 --auxiliary --weighted_sample --init_channel 24 --layers 10 --optimizer Adam --arch K49_50_WARM_1
else
    echo 3
    #python main_default.py -s 2
    #python main_baseline.py --set K49 --cutout --seed 2 --weighted_sample --optimizer adam
    #python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.38434528090336906 --drop_path_prob 0.22175002567477953 --init_channel 24 --learning_rate 0.010959559764224188 --layers 9 --optimizer AdamW --weight_decay 0.0018691938287144958 --arch K49_50_2
    #python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.4125016947470608 --drop_path_prob 0.20961204405663425 --init_channel 23 --learning_rate 0.011645124655282177 --layers 10 --optimizer AdamW --weight_decay 0.00032645523838299495 --arch K49_TS_0
#    python train.py --cutout --set K49 --seed 2  --auxiliary --weighted_sample --init_channel 24 --layers 10 --optimizer Adam --arch K49_50_WARM_2
fi

#if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
#then
    #echo 1
#    python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.3851897598096238 --drop_path_prob 0.2464329061426443 --init_channel 23 --learning_rate 0.022637736874299628 --layers 9 --optimizer AdamW --weight_decay 0.00158689453920829 --arch PC_DARTS_K49_11
    #python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.22983473880372648 --drop_path_prob 0.22476479203555422 --init_channel 23 --learning_rate 0.012010288921625147 --layers 9 --optimizer AdamW --weight_decay 0.0003158766677225857 --arch KMNIST_50_0
#elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]
#then
    #echo 2
#    python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.30420662698282874 --drop_path_prob 0.23443128708267458 --init_channel 17 --learning_rate 0.0175028951759942 --layers 10 --optimizer AdamW --weight_decay 0.0015043720328872033 --arch PC_DARTS_K49_12
    #python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.3972004166714155 --drop_path_prob 0.23809376906164603 --init_channel 24 --learning_rate 0.00958267769550307 --layers 10 --optimizer AdamW --weight_decay 0.00010553920364216671 --arch KMNIST_50_1
#else
    #echo 3
#    python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.4640723736557679 --drop_path_prob 0.23657417620941099 --init_channel 24 --learning_rate 0.006449340695736157 --layers 7 --optimizer AdamW --weight_decay 0.0015171979609508858 --arch PC_DARTS_K49_13
    #python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.2603280673215272 --drop_path_prob 0.2015879772110147 --init_channel 20 --learning_rate 0.008663923038072537 --layers 9 --optimizer AdamW --weight_decay 0.001199865567527739 --arch KMNIST_50_2
#fi

#python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.3851897598096238 --drop_path_prob 0.2464329061426443 --init_channel 23 --learning_rate 0.022637736874299628 --layers 9 --optimizer AdamW --weight_decay 0.00158689453920829 --arch PC_DARTS_K49_11
#python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.30420662698282874 --drop_path_prob 0.23443128708267458 --init_channel 17 --learning_rate 0.0175028951759942 --layers 10 --optimizer AdamW --weight_decay 0.0015043720328872033 --arch PC_DARTS_K49_12
#python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.4640723736557679 --drop_path_prob 0.23657417620941099 --init_channel 24 --learning_rate 0.006449340695736157 --layers 7 --optimizer AdamW --weight_decay 0.0015171979609508858 --arch PC_DARTS_K49_13

#python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.41573618371273047 --drop_path_prob 0.3438951137821758 --init_channel 20 --learning_rate 0.009570012712227077 --layers 9 --optimizer AdamW --weight_decay 0.00010923484407269452 --arch PC_DARTS_K49_18
#python train.py --cutout --set K49 --auxiliary --weighted_sample --auxiliary_weight 0.39824050254011 --drop_path_prob 0.39835924879698514 --init_channel 20 --learning_rate 0.008085394472293925 --layers 8 --optimizer AdamW --weight_decay 0.00022220521551917044 --arch PC_DARTS_K49_19

