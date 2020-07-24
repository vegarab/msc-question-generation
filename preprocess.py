import os
from dataclasses import dataclass, field

import nlp
import torch

from transformers import (
    HfArgumentParser,
    T5Tokenizer
)

from args import DataArguments


MODEL_TO_TOK = {
    "t5-base": T5Tokenizer,
}


class DataProcessor:
    def __init__(self, tokenizer, max_source_length=512, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, dataset):
        dataset = dataset.map(self._add_eos_tokens)
        dataset = dataset.map(self._create_features, batched=True)
        return dataset

    def _add_eos_tokens(self, sample):
        sample["source_text"] = sample["source_text"] + " </s>"
        sample["target_text"] = sample["target_text"] + " </s>"
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


def DataCollator():
    def __init__(self, tokenizer, model_t="t5", is_training=True, tpu=False):
        self.tokenizer = tokenizer
        self.model_t = model_t
        self.is_training = is_training
        self.using_tpu = tpu

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        target_ids = torch.stack([sample["target_ids"] for sample in batch])
        source_ids = torch.stack([sample["source_ids"] for sample in batch])
        attention_mask = torch.stack(
            [sample["attention_mask"] for sample in batch])

        labels = target_ids[:, :-1].contiguous()
        lm_labels = target_ids[:, 1:].clone().detach()

        if self.is_training:
            lm_labels[lm_labels[:, 1:] == self.tokenizer.pad_token_id] = -100

        batch_params = {
            "input_ids": source_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_input_ids": labels
        }

        return batch_params


def preprocess():
    parser = HfArgumentParser((DataArguments))
    data_args = parser.parse_args_into_dataclasses()[0]

    tok_name = data_args.tokenizer_name
    tokenizer = MODEL_TO_TOK[tok_name].from_pretrained(tok_name)

    processor = DataProcessor(
        tokenizer, data_args.max_source_length, data_args.max_target_length)

    train_data = nlp.load_dataset(
        f"./datasets/{data_args.dataset}.py", split=nlp.Split.TRAIN)
    test_data = nlp.load_dataset(
        f"./datasets/{data_args.dataset}.py", split=nlp.Split.VALIDATION)

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
