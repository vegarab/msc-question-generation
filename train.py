import logging
import os
from dataclasses import dataclass, field

import torch

from transformers import (
    XLNetLMHeadModel,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    EncoderDecoderModel,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)


from preprocess import NAME_TO_TOK, DataCollator
from args import TrainScriptArguments, get_data_paths


logger = logging.getLogger(__name__)


NAME_TO_MODEL = {
    "t5-base": T5ForConditionalGeneration,
    "t5-small": T5ForConditionalGeneration,
    "xlnet-base-cased": XLNetLMHeadModel,
    "bert-base-cased": EncoderDecoderModel,
    "facebook/bart-large": BartForConditionalGeneration,
    "facebook/bart-base": BartForConditionalGeneration,
}


def run(args=None, training_args=None):
    if args is not None and training_args is not None:
        parser = HfArgumentParser((TrainScriptArguments))
        args = parser.parse_dict(args)[0]
        parser = HfArgumentParser((TrainingArguments))
        training_args = parser.parse_dict(training_args)[0]
    else:
        parser = HfArgumentParser((TrainScriptArguments, TrainingArguments))
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
    if args.is_dryrun:
        os.environ["WANDB_MODE"] = "dryrun"

    logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)

    tokenizer_name = args.model_name if args.tokenizer_name is None else args.tokenizer_name

    # TODO: Fix this hard-coded shit
    if args.model_name == "bert-base-cased":
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            args.model_name, args.model_name)
    else:
        model = NAME_TO_MODEL[args.model_name].from_pretrained(args.model_name)
    tokenizer = NAME_TO_TOK[tokenizer_name].from_pretrained(tokenizer_name)

    logger.info(f"Path {args.train_data_path}")
    train_data = torch.load(
        args.train_data_path) if training_args.do_train else None
    test_data = torch.load(
        args.test_data_path) if training_args.do_eval else None

    # Resizes the train set to args.data_size percentage
    logger.info(f"Train-data pre-size: {train_data.num_rows}")
    if args.absolute_data_size:
        num_rows = args.absolute_data_size
    else:
        num_rows = int(train_data.num_rows * (args.data_size / 100))

    # The select takes time even when selecting all rows. Do this check first.
    if args.absolute_data_size or args.data_size < 100:
        # Shuffle train_data before re-sizing. Controlling with seed
        # Make sure to override the cache file as it does not care about data size
        train_data = train_data.shuffle(seed=training_args.seed,
                                        keep_in_memory=True,
                                        load_from_cache_file=False)
        train_data = train_data.select(torch.arange(0, num_rows),
                                       keep_in_memory=True,
                                       load_from_cache_file=False)

        # For some reason, Dataset.select() and Dataset.shuffle() resets format
        fields = ["source_ids", "target_ids", "attention_mask"]
        train_data.set_format(type="torch", columns=fields)
        test_data.set_format(type="torch", columns=fields)

    logger.info(f"Train-data size: {train_data.num_rows}")

    collator = DataCollator(tokenizer=tokenizer,
                            is_training=training_args.do_train,
                            tpu=training_args.tpu_num_cores is not None)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_data,
                      eval_dataset=test_data,
                      data_collator=collator,
                      prediction_loss_only=True)

    if training_args.do_train:
        trainer.train()
        trainer.save_model()


DEFAULT_T5 = {
    "per_device_train_batch_size": 4,
    "eval_steps": 800,
    "logging_steps": 50,
}

DEFAULT_BART = {
    "per_device_train_batch_size": 8,
    "eval_steps": 300,
    "logging_steps": 50,
}

DEFAULT_BERT = {
    "per_device_train_batch_size": 8,
    "eval_steps": 600,
    "logging_steps": 50,
}

DEFAULT_ARGS = {
    "seed": 42,
    "do_train": True,
    "logging_first_step": True,
    "do_eval": True,
    "save_total_limit": 2,
    "evaluate_during_training": True,
    "num_train_epochs": 3,
    "dataloader_drop_last": True,
}


if __name__ == "__main__":
    parser = HfArgumentParser((TrainScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    model_args = {}
    if "bart" in script_args.model_name:
        model_args = DEFAULT_BART
        model_tag = "bart"
    elif "t5" in script_args.model_name:
        model_args = DEFAULT_T5
        model_tag = "t5"
    elif "bert" in script_args.model_name:
        model_args = DEFAULT_BERT
        model_tag = "bert"

    train_data_path, test_data_path = get_data_paths(script_args.dataset,
                                                     model_tag)
    args = script_args.__dict__
    args["train_data_path"] = train_data_path
    args["test_data_path"] = test_data_path

    training_args = training_args.__dict__
    if script_args.use_defaults:
        for key, value in DEFAULT_ARGS.items():
            training_args[key] = value
        for key, value in model_args.items():
            training_args[key] = value

    run(args, training_args)
