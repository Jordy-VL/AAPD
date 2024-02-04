python3 baseline_multilabel --experiment_name baseline_BERT_10K_steps \ 
--model_name_or_path 'bert-base-uncased' \
--output_dir '../results' \
--seed 42 \
--evaluation_strategy steps \
--per_device_train_batch_size 16 \ 
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 10000 \
--logging_strategy steps \
--logging_steps 10000 \
--save_steps 500 \
--eval_steps 500 
#0.05*max_steps \ #20 implicit epochs 
#--metric_for_best_model  \
#--label_smoothing_factor 0 \
#--eval_accumulation_steps  \
#baseline_BERT_10K_steps \