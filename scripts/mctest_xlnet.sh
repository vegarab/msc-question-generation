#!/bin/sh

python train.py \
	--train_data_path ./data/mc_test_train_xlnet.pt \
	--test_data_path ./data/mc_test_test_xlnet.pt \
	--model_name xlnet-base-cased \
	--wandb_project msc_question_generation \
	--seed 42 \
	--do_train \
	--logging_steps 100 \
	--num_train_epochs 1 \
	--output_dir models/mc_test_xlnet \
	--dataloader_drop_last \
	--per_device_train_batch_size 1 \
