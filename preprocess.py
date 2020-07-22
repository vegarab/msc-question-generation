import nlp
import torch
from transformers import T5Tokenizer


class DataProcessor:
    def __init__(self, tokenizer, model_t="t5", max_source_length=512,
                 max_target_length=64):
        self.tokenizer = tokenizer
        self.model_t = model_t
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, dataset):
        dataset = dataset.map(self._add_eos_tokens)
        dataset = dataset.map(self._create_features)
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
            return_tensors="pt")

        target_text_encoding = self.tokenizer.batch_encode_plus(
            batch["target_text"],
            max_length=self.max_target_length,
            pad_to_max_length=True,
            return_tensors="pt")

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
            lm_labels[lm_labels[:, 1:] == self.tokenizer.pad_token_id]] = -100

        batch_params = {
            "input_ids": source_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "decoder_input_ids": labels
        }

        return batch_Params
