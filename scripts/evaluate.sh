#!/bin/sh
mkdir -p eval/
python evaluate.py --model_name t5-base --model_path models/squad_t5_100 --test_sets squad
