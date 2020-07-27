#!/bin/sh

mkdir -p ./data
python preprocess.py \
	--train_save_path ./data/mc_test_train_t5.pt \
	--test_save_path ./data/mc_test_test_t5.pt \
	--tokenizer_name t5-base \
	--max_source_length 512 \
	--max_target_length 64 \
	--dataset mc_test
