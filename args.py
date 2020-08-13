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
class EvalArguments:
    model_name: str = field(
        metadata={
            "help": "Model identifier for models in huggingface/transformers."}
    )
    hypothesis_path: str = field(
        metadata={"help": "Path to save hypothesis."}
    )
    test_data_path: str = field(
        metadata={"help": "Path to saved test dataset."}
    )
    reference_path: str = field(
        metadata={"help": "Path to save references. If none, not creating one."},
        default=None
    )
    model_path: str = field(
        metadata={
            "help": """Path to pre-trained model, either from hf/transformers or local. 
                       Defaults to model_name."""},
        default=None
    )
    tokenizer_name: str = field(
        metadata={"help": "Tokenizer identifier. Defaults to model_name."},
        default=None
    )
    max_source_length: int = field(
        metadata={"help": "Max input length for the source context + answer."},
        default=512
    )
    max_target_length: int = field(
        metadata={"help": "Max input length for the target question to generate."},
        default=32
    )
    batch_size: int = field(
        metadata={"help": "Evaluation batch size. Defaults to 8"},
        default=8
    )
