#!/bin/sh

python train.py \
	--train_data_path ./data/squad_train_bart.pt \
	--test_data_path ./data/squad_test_bart.pt \
	--model_name facebook/bart-base \
	--wandb_project msc_question_generation \
	--seed 42 \
	--do_train \
	--logging_first_step \
	--do_eval \
	--is_dryrun \
	--save_total_limit 2 \
	--evaluate_during_training \
	--eval_steps 300 \
	--logging_steps 50 \
	--num_train_epochs 3 \
	--output_dir models/squad_bart \
	--dataloader_drop_last \
	--per_device_train_batch_size 8 \
