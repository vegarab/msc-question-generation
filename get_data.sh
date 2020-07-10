#!/bin/sh

echo "Downloading NewsQA"
mkdir -p datasets/news_qa/raw
cd datasets/news_qa/raw
curl -LO https://multiqa.s3.amazonaws.com/squad2-0_format_data/NewsQA_train.json.gz 
curl -LO https://multiqa.s3.amazonaws.com/squad2-0_format_data/NewsQA_dev.json.gz

echo ""
echo "Processing NewsQA"
cd ../..
python prepare_gzip_data.py \
	--input news_qa/raw/NewsQA_train.json.gz \
	--output news_qa/raw/newsqa_train.json

python prepare_gzip_data.py \
	--input news_qa/raw/NewsQA_dev.json.gz \
	--output news_qa/raw/newsqa_dev.json

python squad_to_csv.py \
	--input news_qa/raw/newsqa_train.json \
	--output news_qa/newsqa_train.csv

python squad_to_csv.py \
	--input news_qa/raw/newsqa_dev.json \
	--output news_qa/newsqa_dev.csv


echo ""
echo "Downloading MCTest"
cd ..
mkdir -p datasets/mctest/raw
cd datasets/mctest/raw
git clone https://github.com/mcobzarenco/mctest 

echo "Processing MCTest"
cd mctest/data/MCTest
mv *.tsv ../../..
mv *.ans ../../..
cd ../MCTestAnswers
mv *.ans ../../..
cd ../../..
rm -rf mctest/
cd ../..

pwd
python mctest.py

echo ""
echo "Finished"
