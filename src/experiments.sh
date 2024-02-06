#!/bin/bash 

CUDA_VISIBLE_DEVICES=3 python3 baseline_multilabel.py --experiment_name SciBERT_twowayloss_25K_bs64 \
--model_name_or_path 'allenai/scibert_scivocab_uncased' \
--output_dir '../results' \
--seed 42 \
--evaluation_strategy steps \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 25000 \
--logging_strategy steps \
--logging_steps 0.05 \
--save_steps 0.2 \
--eval_steps 0.2 
--Tp 4.0 \
--Tn 1.0 \
--criterion 'TwoWayLoss' \

'''
## server - higher batch size
CUDA_VISIBLE_DEVICES=3 python3 baseline_multilabel.py --experiment_name SciBERT_25K_steps_bs64 \
--model_name_or_path 'allenai/scibert_scivocab_uncased' \
--output_dir '../results' \
--seed 42 \
--evaluation_strategy steps \
--per_device_train_batch_size 64 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--max_steps 25000 \
--logging_strategy steps \
--logging_steps 0.05 \
--save_steps 0.2 \
--eval_steps 0.2 
'''

#0.05*max_steps \ #20 implicit epochs 
#--metric_for_best_model  \
#--label_smoothing_factor 0 \
#--eval_accumulation_steps  \
#baseline_BERT_10K_steps \