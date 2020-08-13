from dataclasses import dataclass, field

from typing import List


@dataclass
class Arguments:
    train_data_path: str = field(
        metadata={"help": "Path to training dataset"}
    )
    test_data_path: str = field(
        metadata={"help": "Path to test dataset"}
    )
    model_name: str = field(
        metadata={
            "help": "Model identifier for models in huggingface/transformers."}
    )
    wandb_project: str = field(
        metadata={"help": "Name of wandb project for loggin"}
    )
    tokenizer_name: str = field(
        metadata={"help": "Tokenizer identifier, defaults to model_name"},
        default=None
    )
    max_source_length: int = field(
        metadata={"help": "Max input length for the source context + answer"},
        default=512
    )
    max_target_length: int = field(
        metadata={"help": "Max input length for the target question to generate"},
        default=32
    )
    is_dryrun: bool = field(
        metadata={"help": "Set True to notify wandb that we are offline"},
        default=False
    )


@dataclass
class DataArguments:
    train_save_path: str = field(
        metadata={"help": "Path to save training dataset"}
    )
    test_save_path: str = field(
        metadata={"help": "Path to save test dataset"}
    )
    tokenizer_name: str = field(
        metadata={"help": "Tokenizer identifier, from huggingface/transformers"}
    )
    dataset: str = field(
        metadata={"help": "Dataset identifier from custom huggingface/nlp scripts"}
    )
    max_source_length: int = field(
        metadata={"help": "Max input length for the source context + answer"},
        default=512
    )
    max_target_length: int = field(
        metadata={"help": "Max input length for the target question to generate"},
        default=32
    )


@dataclass
class EvalScriptArguments:
    model_name: str = field(
        metadata={
            "help": "Model identifier for models in huggingface/transformers."}
    )
    test_sets: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Which sets to test on. If empty, test against all. Defaults to []"},
    )
    tokenizer_name: str = field(
        metadata={"help": "Tokenizer identifier, defaults to model_name"},
        default=None
    )
    model_path: str = field(
        metadata={
            "help": "Path to local model checkpoint or hf/transformers. Defaults to model_name"},
        default=None
    )
