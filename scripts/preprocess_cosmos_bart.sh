#!/bin/sh

mkdir -p ./data
python preprocess.py \
	--train_save_path ./data/cosmos_train_bart.pt \
	--test_save_path ./data/cosmos_test_bart.pt \
	--tokenizer_name facebook/bart-base \
	--max_source_length 512 \
	--max_target_length 64 \
	--dataset cosmos_qa
