CUDA_VISIBLE_DEVICES=2 python3 baseline_generative.py --experiment_name Llama2_10K_instructions \
--LLM 'NousResearch/Llama-2-7b-hf' \
--output_dir '../results' \
--seed 42 \
--evaluation_strategy steps \
--per_device_train_batch_size 8  \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--num_train_epochs 1 \
--max_steps 10000 \
--logging_strategy steps \
--logging_steps 0.1 \
--save_steps 0.1 \
--eval_steps 0.1 \