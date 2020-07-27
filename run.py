from train import run

args = {
    "train_data_path": "data/squad_train.pt",
    "test_data_path": "data/squad_test.pt",
    "model_name": "t5-base",
    "wandb_project": "msc_question_generation",
    "model_path": "mc_test_t5",
    "max_source_length": 512,
    "max_target_length": 32,

}
training_args = {
    "seed": 42,
    "do_train": True,
    "logging_steps": 100,
    "num_training_epochs": 3,
    "output_dir": "mc_test",
    "dataloader_drop_last": True,
    "n_gpu": 1,
}

run({**args, **training_args})
