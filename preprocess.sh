#!/bin/sh

mkdir -p ./data
python preprocess.py \
	--train_save_path ./data/squad_train_100.pt \
	--test_save_path ./data/squad_test_100.pt \
	--tokenizer_name t5-base \
	--max_source_length 512 \
	--max_target_length 64 \
	--dataset squad
