#!/bin/sh
mkdir -p models
python trian.py --model_name t5-base --dataset squad --output_dir models/squad_t5_100  --data_size 100
