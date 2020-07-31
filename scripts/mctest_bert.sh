#!/bin/sh

python train.py \
	--train_data_path ./data/mc_test_train_bert.pt \
	--test_data_path ./data/mc_test_test_bert.pt \
	--model_name bert-base-cased \
	--wandb_project msc_question_generation \
	--seed 42 \
	--do_train \
	--do_eval \
	--eval_steps 100 \
	--logging_steps 100 \
	--num_train_epochs 1 \
	--output_dir ./models/mc_test_bert \
	--dataloader_drop_last \
	--per_device_train_batch_size 8 \
