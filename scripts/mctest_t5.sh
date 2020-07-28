#!/bin/sh

python train.py \
	--train_data_path ./data/mc_test_train_t5.pt \
	--test_data_path ./data/mc_test_test_t5.pt \
	--model_name t5-base \
	--wandb_project msc_question_generation \
	--model_path mc_test_t5 \
	--seed 42 \
	--do_train \
	--logging_steps 100 \
	--num_train_epochs 1 \
	--output_dir mc_test \
	--dataloader_drop_last \
	--per_device_train_batch_size 1 \
