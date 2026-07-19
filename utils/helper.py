import time
import os
from datetime import datetime

import yaml
import logging
import torch

def load_config(config_path: str):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def setup_logger(run_dir):
    logger = logging.getLogger("Football_Sim_Play_Encoder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(run_dir / "train.log")
    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def setup_run_dir(root):
    """Creates a timestamped run directory and returns its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "runs" / f"{timestamp}_spleen_unetr_pp"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_checkpoint(state, is_best, checkpoint_dir, filename="latest_checkpoint.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, save_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)


def print_model_summary(model):
    print("\n" + "=" * 40)
    print(f"{'MODEL SUMMARY':^40}")
    print("=" * 40)

    total_params = 0
    trainable_params = 0

    for name, parameter in model.named_parameters():
        params_count = parameter.numel()
        total_params += params_count
        if parameter.requires_grad:
            trainable_params += params_count

    non_trainable_params = total_params - trainable_params

    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Frozen Parameters:     {non_trainable_params:,}")

    if total_params > 0:
        percent_trainable = (trainable_params / total_params) * 100
        print(f"% Trainable:           {percent_trainable:.2f}%")
    print("=" * 40 + "\n")