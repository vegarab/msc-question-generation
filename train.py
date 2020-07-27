import logging
import os
from dataclasses import dataclass, field

import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed
)


from preprocess import MODEL_TO_TOK
from args import Arguments, DataArguments


logger = logging.getLogger(__name__)


def run(args):
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [
            -1, 0] else logging.WARN,
    )

    set_seed(training_args.seed)

    # Setup wandb
    os.environ["WANDB_PROJECT"] = args.wandb_project
    logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)

    tokenizer_name = args.model_name if args.tokenizer_name is None else args.tokenizer_name
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = MODEL_TO_TOK[tokenizer_name].from_pretrained(tokenizer_name)

    train_data = torch.load(args.train_data_path)
    test_data = torch.load(args.test_data_path)
