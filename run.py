from train import run

args = {
    "model_name": "t5-small",
    "train_data_path": "data/squad_train.pt",
    "test_data_path": "data/squad_test.pt",
    "max_source_length": 512,
    "max_target_lenght": 32,
    "seed": 42,
    "do_train": True,
    "logging_steps": 100
}

run(args)
