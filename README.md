# msc-question-generation
This is the accompanying code to my thesis for the degree of MSc Data Science
at the University of Southampton (2020), titled *Generalization and Transfer
Performance of Neural Question Generation Models*. 

The final thesis report will be made available as soon as it has been marked
and assessed.

### Abstract
Asking questions is important for testing both machine and human reading
comprehension. The ability to automatically generated high-quality questions
based on textual stimulus and context can have great value for many
applications, including pedagogically-motivated learning and assessment. Modern
deep-learning techniques for natural language processing and generation, such
as the Transformer, are demonstrating increasing capabilities of generated
comprehensive and relevant questions. 

This thesis includes a study of how different Transformer based models
generalize across multiple domains of data when observing a single domain
during fine-tuning. We train three different pre-trained architectures on four
different data domains and evaluate their zero-shot performance on
out-of-domain test sets. We provide empirical data that illustrates how
fine-tuning on the correct data domain is essential for generating
comprehensive questions in a domain-specific setting. 

Further, we investigate how the generalization performance scales with the
amount of fine-tuning data and report findings that indicate an early
diminishing returns on performance. As well as further evidence that the choice
of domain for fine-tuning is more impactful than the amount of data available.

Lastly, we fine-tune T5-Large on the SQuAD training set and provide a 7.3\%
improvement in METEOR performance over the current state-of-the-art on the
SQuAD question generation benchmark. We also provide a novel reporting of
embedding-based evaluation metrics on this task.

## Contents
This repository includes all the necessary code to download, preprocess and
store the datasets used (SQuAD1.1, NewsQA, MCTest, CosmosQA): `./datasets/`,
`preprocess.py`, using [HuggingFace nlp](https://github.com/huggingface/datasets) 
framework. A training procedure using the 
[HuggingFace Transformer](https://github.com/huggingface/transformers/) 
framework: `train.py`. Lastly, a script for performing inference on the test
sets: `evaluation.py`. 

The implementation is made to be general and easy to use for the thesis
project. There are some clearly hard-coded logic, but the implementation as a
whole can be converted to a general framework with little effort. 

## How to use
Download NewsQA and MCTest through a custom script `scripts/get_data.sh`:
```
./scripts/get_data.sh
```
These needs to be downloaded seperatly due to licensing terms and necessary
preprocessing. 

Preprocess a dataset, using a specific tokenizer:
```
mkdir -p data/
python preprocess.py --tokenizer_name facebook/bart-base --dataset mc_test
python preprocess.py --tokenizer_name facebook/bart-base --dataset squad
```

Train your model on the dataset:
```
mkdir -p models/
python train.py --model_name facebook/bart-base --dataset mc_test --output_dir models/mc_test_bart_100 --data_size 100
```

Finally, perform inference on your trained model, using multiple test sets:
```
mkdir -p eval/
python evaluate.py --model_name facebook/bart-base --model_path models/mc_test_bart_100 --test_sets mc_test squad
```

The `./scripts/` contains some basic examples to get started. Further, use the
`--help` parameter to get an overview of the available parameters for each
python-script. 

**Note**: The evaluation script requires the following model path format:
`models/{dataset}_{single-token-model-name}_{digit}`

## Dependencies
See `requirements.txt`. Additionally, we use the 
[HuggingFace Transformer](https://github.com/huggingface/transformers/)
framework and [HuggingFace nlp](https://github.com/huggingface/datasets) (now
called datasets) framework. These are evolving rapidly, and the pre-release
versions of huggingface/nlp were used. To ensure compatibility, the followin
specific commits should be used: <br>

Transformers: 05810cd <br>
NLP: 9f62388
