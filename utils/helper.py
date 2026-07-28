import os
import logging
from datetime import datetime
from pathlib import Path

import torch


def setup_logger(run_dir):
    logger = logging.getLogger("Play_Encoder_Baseline")
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
    run_dir = root / "runs" / f"{timestamp}_simple_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(state, is_best, checkpoint_dir, filename="latest_checkpoint.pth"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, save_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)


def print_model_summary(model, logger=None):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    msg = (
        "\n" + "=" * 40 + "\n" +
        f"{'MODEL SUMMARY':^40}\n" + "=" * 40 + "\n" +
        f"Total Parameters:      {total_params:,}\n" +
        f"Trainable Parameters:  {trainable_params:,}\n" +
        "=" * 40
    )
    (logger.info if logger else print)(msg)
