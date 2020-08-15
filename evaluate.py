import os
import re

import torch
from torch.utils.data import DataLoader

import nlp

from transformers import (
    HfArgumentParser,
)

from tqdm.auto import tqdm

from preprocess import DataCollator, NAME_TO_TOK
from train import NAME_TO_MODEL
from args import EvalScriptArguments


device = "cuda" if torch.cuda.is_available() else "cpu"


DATASETS = ["mc_test", "squad", "cosmos", "news"]


# These two functions are a quick fix to not having to pass extra
# arguments alongside the relative model_path. All extra files can be derived
# from this with these two functions.
def _get_model_name_from_model_path(model_path):
    return model_path.split("/")[-1]


def _get_data_file_from_model_name_path(model_name_path, dataset):
    # Remove indicator of trained size, e.g. cosmos_t5_100 -> cosmos_t5
    data_file = re.sub("(_)(\d+)", "", model_name_path)

    # Add "test" before model indicator, e.g. cosmos_t5 -> cosmos_test_t5
    split_d = data_file.split("_")
    split_d.insert(-1, "test")

    # Update dataset annotation
    idx = 1
    # mc_test is split into two due to '_' in dataset annotation
    # we therefore need to add a special case here..
    if split_d[0] == "mc":
        idx = 2
    split_d[:idx] = [dataset]

    data_file = "_".join(split_d)

    return data_file


def get_predictions(model, model_name, tokenizer, loader, max_length=32, num_beams=4, rep_penalty=2.5):
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader):
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=rep_penalty,
                early_stopping=True
            )

            # Hack to fix the ignoring of first token during decoding
            if "bart" in model_name:
                prediction = [tokenizer.decode(output, skip_special_tokens=True)[1:]
                              for output in outputs]
            else:
                prediction = [tokenizer.decode(output, skip_special_tokens=True)
                              for output in outputs]
            predictions.extend(prediction)

    return predictions


def get_true_targets(tokenizer, loader):
    true_targets = []
    for batch in tqdm(loader):
        target = [tokenizer.decode(output, skip_special_tokens=True)
                  for output in batch["labels"]]
        true_targets.extend(target)

    return true_targets


def evaluate(model_name, model_path, tokenizer_name, batch_size, test_sets):
    tokenizer = NAME_TO_TOK[tokenizer_name].from_pretrained(tokenizer_name)
    model = NAME_TO_MODEL[model_name].from_pretrained(model_path)

    collator = DataCollator(tokenizer, is_training=False)

    # Derive file name from model_path
    model_path_name = _get_model_name_from_model_path(model_path)
    for set_ in test_sets:
        print(f"*** Creating {model_path_name} {set_} hypothesis file ***")
        predictions_file = f"./eval/{model_path_name}_{set_}.txt"
        # No need to run this costly procedure if the output already exists
        if os.path.isfile(predictions_file):
            print("Skipping... Already exists")
            continue

        data_file = _get_data_file_from_model_name_path(model_path_name, set_)
        test_set = torch.load(f"./data/{data_file}.pt")
        test_loader = DataLoader(
            test_set, collate_fn=collator, batch_size=batch_size)

        predictions = get_predictions(model,
                                      model_name,
                                      tokenizer,
                                      test_loader,
                                      32)

        with open(predictions_file, "w") as f:
            f.write("\n".join(predictions))


def create_reference_file(dataset, batch_size):
    # Choose BART as it is the smallest and fastest tokenizer out of the three
    tok_name = "facebook/bart-base"
    tokenizer = NAME_TO_TOK[tok_name].from_pretrained(tok_name)

    collator = DataCollator(tokenizer, is_training=False)
    data = torch.load(f"./data/{dataset}_test_bart.pt")
    loader = DataLoader(data, collate_fn=collator, batch_size=batch_size)

    true_targets = get_true_targets(tokenizer, loader)
    with open(f"./eval/{dataset}.txt", "w") as f:
        f.write("\n".join(true_targets))


if __name__ == "__main__":
    parser = HfArgumentParser((EvalScriptArguments))
    args = parser.parse_args_into_dataclasses()[0]

    batch_size = 4
    if "bart" in args.model_name:
        batch_size = 8

    eval_args = {
        "model_name": args.model_name,
        "model_path": args.model_name if args.model_path is None else args.model_path,
        "tokenizer_name": args.model_name if args.tokenizer_name is None else args.tokenizer_name,
        "batch_size": batch_size,
    }

    # Make sure that all the reference files exist
    for dset in DATASETS:
        print(f"*** Creating {dset} reference file ***")
        if os.path.isfile("./eval/" + dset + ".txt"):
            print("Skipping... Already exists")
            continue
        else:
            create_reference_file(dset, batch_size=8)

    # Test on all datasets if args.test_sets is empty
    test_sets = args.test_sets
    if len(args.test_sets) < 1:
        test_sets = DATASETS

    eval_args["test_sets"] = test_sets

    evaluate(**eval_args)
