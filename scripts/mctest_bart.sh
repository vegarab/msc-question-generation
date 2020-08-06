#!/bin/sh

python train.py \
	--train_data_path ./data/mc_test_train_bart.pt \
	--test_data_path ./data/mc_test_test_bart.pt \
	--model_name facebook/bart-large \
	--wandb_project msc_question_generation \
	--seed 42 \
	--do_train \
	--do_eval \
	--is_dryrun \
	--evaluate_during_training \
	--eval_steps 500 \
	--logging_steps 100 \
	--num_train_epochs 50 \
	--output_dir models/mc_test_bart \
	--dataloader_drop_last \
	--per_device_train_batch_size 2 \
