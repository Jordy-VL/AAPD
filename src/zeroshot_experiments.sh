#!/bin/bash 

CUDA_VISIBLE_DEVICES=1 python3 baseline_zeroshot.py --experiment_name setfit_mpnet_20K_steps \
--output_dir '../results' \
--seed 42 \
--evaluation_strategy steps \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 20000 \
--logging_strategy steps \
--logging_steps 0.05 \
--save_steps 0.05 \
--eval_steps 0.05 \

'''
--sentence_transformer 'jordyvl/scibert_scivocab_uncased_sentence_transformer' \
'''

"""
CUDA_VISIBLE_DEVICES=0 python3 baseline_zeroshot.py --experiment_name setfit_mpnet_20K_steps \
--sentence_transformer 'jordyvl/scibert_scivocab_uncased_sentence_transformer' \
--output_dir '../results' \
--seed 42 \
--evaluation_strategy steps \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 20000 \
--logging_strategy steps \
--logging_steps 0.05 \
--save_steps 0.05 \
--eval_steps 0.05 \
"""