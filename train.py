import logging
import os
from dataclasses import dataclass, field

import torch

from transformers import (
    XLNetLMHeadModel,
    T5ForConditionalGeneration,
    EncoderDecoderModel,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)


from preprocess import NAME_TO_TOK, DataCollator
from args import Arguments, DataArguments


logger = logging.getLogger(__name__)


NAME_TO_MODEL = {
    "t5-base": T5ForConditionalGeneration,
    "t5-small": T5ForConditionalGeneration,
    "xlnet-base-cased": XLNetLMHeadModel,
    "bert-base-cased": EncoderDecoderModel,
}


def run(args=None):
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses(args)

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

    # TODO: Fix this hard-coded shit
    if args.model_name == "bert-base-cased":
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            args.model_name, args.model_name)
    else:
        model = NAME_TO_MODEL[args.model_name].from_pretrained(args.model_name)
    tokenizer = NAME_TO_TOK[tokenizer_name].from_pretrained(tokenizer_name)

    train_data = torch.load(
        args.train_data_path) if training_args.do_train else None
    test_data = torch.load(
        args.test_data_path) if training_args.do_eval else None

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


if __name__ == "__main__":
    run()
