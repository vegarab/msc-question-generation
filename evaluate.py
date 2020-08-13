import torch
from torch.utils.data import DataLoader

import nlp

from transformers import (
    HfArgumentParser,
)

from tqdm.auto import tqdm

from preprocess import DataCollator, NAME_TO_TOK
from train import NAME_TO_MODEL
from args import EvalArguments


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_predictions(model, tokenizer, loader, max_length=32, num_beams=4, rep_penalty=2.5):
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


def create_evaluation_files():
    parser = HfArgumentParser((EvalArguments))
    args = parser.parse_args_into_dataclasses()[0]

    tok_name = args.model_name if args.tokenizer_name is None else args.tokenizer_name
    tokenizer = NAME_TO_TOK[tok_name].from_pretrained(tok_name)
    args.model_path = args.model_name if args.model_path is None else args.model_path
    model = NAME_TO_MODEL[args.model_name].from_pretrained(args.model_path)

    test_set = torch.load(args.test_data_path)

    collator = DataCollator(tokenizer, is_training=False)
    test_loader = DataLoader(
        test_set, collate_fn=collator, batch_size=args.batch_size)

    predictions = get_predictions(model,
                                  tokenizer,
                                  test_loader,
                                  args.max_target_length)

    with open(args.hypothesis_path, "w") as f:
        f.write("\n".join(predictions))

    # We only create a reference file if it is specified. These are shared
    # across models
    if args.reference_path is not None:
        true_targets = get_true_targets(tokenizer, test_loader)
        with open(args.reference_path, "w") as f:
            f.write("\n".join(true_targets))


if __name__ == "__main__":
    create_evaluation_files()
