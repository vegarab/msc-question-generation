from dataclasses import dataclass, field


@dataclass
class Arguments:
    train_data_path: str = field(
        metadata={"help": "Path to training dataset"}
    )
    test_data_path: str = field(
        metadata={"help": "Path to test dataset"}
    )
    model_name: str = field(
        metadata={"help": "Model identifier for models in huggingface/transformers"}
    )
    wandb_project: str = field(
        metadata={"help": "Name of wandb project for loggin"}
    )
    model_path: str = field(
        metadata={"help": "Path to save/load the model"}
    )
    tokenizer_name: str = field(
        metadata={"help": "Tokenizer identifier, defaults to model_name"},
        default=None
    )
    max_source_length: int = field(
        metadata={"help": "Max input length for the source context + answer"},
        default=512
    )
    max_target_lenght: int = field(
        metadata={"help": "Max input length for the target question to generate"},
        default=32
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
