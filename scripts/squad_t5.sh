#!/bin/sh

python train.py \
	--train_data_path ./data/squad_train_t5.pt \
	--test_data_path ./data/squad_test_t5.pt \
	--model_name t5-base \
	--wandb_project msc_question_generation \
	--seed 42 \
	--do_train \
	--do_eval \
	--is_dryrun \
	--save_total_limit 2 \
	--evaluate_during_training \
	--eval_steps 2000 \
	--logging_steps 100 \
	--num_train_epochs 5 \
	--output_dir models/squad_t5 \
	--dataloader_drop_last \
	--per_device_train_batch_size 4 \
