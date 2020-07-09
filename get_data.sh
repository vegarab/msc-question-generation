#!/bin/sh

cd datasets/
curl -LO https://multiqa.s3.amazonaws.com/squad2-0_format_data/NewsQA_train.json.gz 
curl -LO https://multiqa.s3.amazonaws.com/squad2-0_format_data/NewsQA_dev.json.gz

cd ..

python prepare_gzip_data.py datasets/NewsQA_train.json.gz
mv output.json datasets/newsqa_train.json
python prepare_gzip_data.py datasets/NewsQA_dev.json.gz
mv output.json datasets/newsqa_dev.json
