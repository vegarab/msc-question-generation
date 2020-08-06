#!/bin/sh

python train.py \
	--train_data_path ./data/squad_train_bert.pt \
	--test_data_path ./data/squad_test_bert.pt \
	--model_name models/squad_bert/checkpoint-3500 \
	--tokenizer_name bert-base-cased \
	--wandb_project msc_question_generation \
	--seed 42 \
	--do_train \
	--do_eval \
	--evaluate_during_training \
	--eval_steps 500 \
	--logging_steps 100 \
	--num_train_epochs 1 \
	--output_dir models/squad_bert_1 \
	--dataloader_drop_last \
	--per_device_train_batch_size 6 \
