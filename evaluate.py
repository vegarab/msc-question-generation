import os

import torch
from torch.utils.data import DataLoader

import nlp
import nlgeval

from transformers import (
    HfArgumentParser,
)

from tqdm.auto import tqdm

from preprocess import DataCollator, NAME_TO_TOK
from train import NAME_TO_MODEL
from args import EvalScriptArguments


device = "cuda" if torch.cuda.is_available() else "cpu"


DATASETS = ["mc_test", "squad", "cosmos", "news"]


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

    for set_ in test_sets:
        test_set = torch.load(f"./data/{args.model_path}.pt")
        test_loader = DataLoader(
            test_set, collate_fn=collator, batch_size=batch_size)

        predictions = get_predictions(model,
                                      model_name,
                                      tokenizer,
                                      test_loader,
                                      32)

        with open(f"./data/{model_path}_{set_}.txt", "w") as f:
            f.write("\n".join(predictions))


def create_reference_file(dataset, batch_size):
    # Choose BART as it is the smallest and fastest tokenizer out of the three
    tok_name = "facebook/bart-base"
    tokenizer = NAME_TO_TOK[tok_name].from_pretrained(tok_name)

    collator = DataCollator(tokenizer, is_training=False)
    data = torch.load(f"./data/{dataset}_test_bart.pt")
    loader = DataLoader(data, collate_fn=collator, batch_size=batch_size)

    true_targets = get_true_targets(tokenizer, loader)
    with open(f"./data/{dataset}.txt", "w") as f:
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
        if not os.path.isfile(dset + ".txt"):
            create_reference_file(dset, batch_size=8)

    # Test on all datasets if args.test_sets is empty
    if len(args.test_sets) < 1:
        test_sets = DATASETS

    eval_args["test_sets"] = test_sets

    evaluate(**eval_args)
