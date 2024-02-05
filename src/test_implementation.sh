#!/bin/bash 
python3 baseline_multilabel.py --experiment_name test_implementation \
--model_name_or_path 'bert-base-uncased' \
--output_dir '../results' \
--seed 42 \
--evaluation_strategy steps \
--per_device_train_batch_size 8  \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 10 \
--logging_strategy steps \
--logging_steps 0.5 \
--save_steps 0.5 \
--eval_steps 0.5 \

"""
python3 baseline_zeroshot.py --experiment_name test_implementation \
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