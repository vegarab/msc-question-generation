#!/bin/sh

mkdir -p ./data
python preprocess.py \
	--train_save_path ./data/squad_train_bart.pt \
	--test_save_path ./data/squad_test_bart.pt \
	--tokenizer_name facebook/bart-large \
	--max_source_length 512 \
	--max_target_length 64 \
	--dataset squad
