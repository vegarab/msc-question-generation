import os
from dataclasses import dataclass, field

import nlp
import torch

from transformers import (
    HfArgumentParser,
    T5Tokenizer,
    XLNetTokenizer,
    BertTokenizer,
    BartTokenizer,
)

from args import DataArguments


NAME_TO_TOK = {
    "t5-base": T5Tokenizer,
    "t5-small": T5Tokenizer,
    "bert-base-cased": BertTokenizer,
    "xlnet-base-cased": XLNetTokenizer,
    "facebook/bart-large": BartTokenizer,
    "facebook/bart-base": BartTokenizer,
}


class DataProcessor:
    def __init__(self, tokenizer, max_source_length=512, max_target_length=32, model_type="bert"):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        # Set it here to suppress the warning when == False
        self.bos_token = bool(self.tokenizer.bos_token)

    def __call__(self, dataset):
        dataset = dataset.map(self._format_text)
        dataset = dataset.map(self._add_bos_tokens)
        dataset = dataset.map(self._add_eos_tokens)
        dataset = dataset.map(self._create_features, batched=True)

        return dataset

    def _format_text(self, sample):
        if self.model_type == "bert":
            source_text = f"{sample['answer']} [SEP] {sample['context']}"
        else:
            source_text = f"answer: {sample['answer']} context: {sample['context']}"

        target_text = sample["question"]

        features = {
            "source_text": source_text,
            "target_text": target_text
        }

        return features

    def _add_bos_tokens(self, sample):
        if not self.model_type == "bert":
            # BART and T5 ignores the first token during decoding. We therefore
            # add a <pad> token to the beginning to ensure correct decoding.
            sample["target_text"] = f"<pad> {sample['target_text']}"
        return sample

    def _add_eos_tokens(self, sample):
        if self.model_type == "t5":
            # T5 does not add EOS itself
            sample["source_text"] = f"{sample['source_text']} </s>"
            sample["target_text"] = f"{sample['target_text']} </s>"
        return sample

    def _create_features(self, batch):
        source_text_encoding = self.tokenizer.batch_encode_plus(
            batch["source_text"],
            max_length=self.max_source_length,
            pad_to_max_length=True,
            truncation=True)

        target_text_encoding = self.tokenizer.batch_encode_plus(
            batch["target_text"],
            max_length=self.max_target_length,
            pad_to_max_length=True,
            truncation=True)

        features = {
            "source_ids": source_text_encoding["input_ids"],
            "target_ids": target_text_encoding["input_ids"],
            "attention_mask": source_text_encoding["attention_mask"]
        }

        return features


class DataCollator:
    def __init__(self, tokenizer, model_t="t5-base", is_training=True, tpu=False):
        self.tokenizer = tokenizer
        self.model_t = model_t
        self.is_training = is_training
        self.using_tpu = tpu

    def __call__(self, batch):
        target_ids = torch.stack([sample["target_ids"] for sample in batch])
        source_ids = torch.stack([sample["source_ids"] for sample in batch])
        attention_mask = torch.stack(
            [sample["attention_mask"] for sample in batch])

        decoder_input_ids = target_ids[:, :-1].contiguous()
        lm_labels = target_ids[:, 1:].clone().detach()

        if self.is_training:
            lm_labels[target_ids[:, 1:] == self.tokenizer.pad_token_id] = -100

        batch_params = {
            "input_ids": source_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_input_ids": decoder_input_ids
        }

        return batch_params


def preprocess():
    parser = HfArgumentParser((DataArguments))
    data_args = parser.parse_args_into_dataclasses()[0]

    tok_name = data_args.tokenizer_name
    tokenizer = NAME_TO_TOK[tok_name].from_pretrained(tok_name)

    # TODO: Fix this hardcoded shit
    if tok_name == "t5-base":
        model_type = "t5"
    elif tok_name == "facebook/bart-base":
        model_type = "bart"
    else:
        model_type = "bert"

    processor = DataProcessor(
        tokenizer, data_args.max_source_length, data_args.max_target_length, model_type=model_type)

    # CosmosQA has train, test and validation splits. Since this project only
    # wants a single split for testing, we merge the train and validation
    # splits.
    if data_args.dataset == "cosmos_qa":
        train_data = nlp.load_dataset(
            f"./datasets/{data_args.dataset}.py", split="train+validation")
    else:
        train_data = nlp.load_dataset(
            f"./datasets/{data_args.dataset}.py", split=nlp.Split.TRAIN)

    test_data = nlp.load_dataset(
        f"./datasets/{data_args.dataset}.py", split=nlp.Split.TEST)

    train_data = processor(train_data)
    test_data = processor(test_data)

    # Setup metadata for saving to .pt file
    fields = ["source_ids", "target_ids", "attention_mask"]
    train_data.set_format(type="torch", columns=fields)
    test_data.set_format(type="torch", columns=fields)

    torch.save(train_data, data_args.train_save_path)
    torch.save(test_data, data_args.test_save_path)


if __name__ == "__main__":
    preprocess()
