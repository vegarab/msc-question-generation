#!/bin/sh
mkdir -p data
python preprocess.py --tokenizer_name t5-base --dataset squad
