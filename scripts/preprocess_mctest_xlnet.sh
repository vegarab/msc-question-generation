#!/bin/sh

mkdir -p ./data
python preprocess.py \
	--train_save_path ./data/mc_test_train_xlnet.pt \
	--test_save_path ./data/mc_test_test_xlnet.pt \
	--tokenizer_name xlnet-base-cased \
	--max_source_length 512 \
	--max_target_length 64 \
	--dataset mc_test
