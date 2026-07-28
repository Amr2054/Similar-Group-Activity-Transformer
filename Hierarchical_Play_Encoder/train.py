import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning import losses, samplers
from utils import setup_run_dir, setup_logger, print_model_summary, save_checkpoint
from data import FIFASequenceDataset
from model import PlayEncoder

# =============================================================================
# PARSER.
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="Configs/hier_model_config.yaml",
    help="Path to the yaml config file",
)
args = parser.parse_args()

# =============================================================================
# Load Config.
# =============================================================================
def load_config(config_path):
    """Flattens the nested hier_model_config.yaml into the single-level
    dict the rest of this script expects."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    data_cfg = raw["data"]
    model_cfg = raw["model"]
    train_cfg = raw["training"]
    log_cfg = raw.get("logging", {})

    config = {
        "data_dir": data_cfg["data_dir"],
        "train_matches": data_cfg["train_matches"],
        "val_matches": data_cfg["val_matches"],
        "target_frames": data_cfg["target_frames"],

        "d_model": model_cfg["embed_dim"],
        "n_heads": model_cfg["n_heads"],
        "frame_layers": model_cfg["frame_layers"],
        "play_layers": model_cfg["play_layers"],
        "dropout": model_cfg.get("dropout", 0.3),
        "output_dim": model_cfg.get("output_dim", 64),

        "batch_size": train_cfg["batch_size"],
        "m_per_class": train_cfg["m_per_class"],
        "num_workers": train_cfg["num_workers"],
        "epochs": train_cfg["epochs"],
        "learning_rate": float(train_cfg["learning_rate"]),
        "weight_decay": train_cfg["weight_decay"],
        "temperature": train_cfg["temperature"],
        "clip_max_norm": train_cfg["clip_max_norm"],
        "seed": train_cfg["seed"],
        "early_stopping_patience": train_cfg.get("early_stopping_patience", 60),
        "save_every": train_cfg.get("save_every", 20),

        "use_tensorboard": log_cfg.get("tensorboard", True),
        "tensorboard_subdir": log_cfg.get("log_dir", "tensorboard"),
    }
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def augment_play(features, p_flip_y=0.5, p_mask_player=0.25, jitter_std=0.04):
    """Simple augmentations for the two SupCon views:
    features: (B, Frames, 23, 9) -> [x, y, z, speed, visibility,
                                      is_attacking, dist_to_goal, is_home, is_away]
    1. Tactical mirroring on the y-axis (index 1).
    2. Small Gaussian jitter on x/y.
    3. Random agent dropout (never the ball, index 0).
    """
    aug = features.clone()
    B = aug.shape[0]

    flip_mask = torch.rand(B, device=features.device) < p_flip_y
    aug[flip_mask, :, :, 1] = aug[flip_mask, :, :, 1] * -1.0

    noise = torch.randn_like(aug[..., 0:2]) * jitter_std
    aug[..., 0:2] = torch.clamp(aug[..., 0:2] + noise, min=-1.0, max=1.0)

    drop_mask = torch.rand(B, 1, 23, 1, device=features.device) > p_mask_player
    drop_mask[:, :, 0, :] = True  # never drop the ball
    aug = aug * drop_mask

    return aug


def main():
    config = load_config(args.config)
    ROOT = Path.cwd()
    run_dir = setup_run_dir(ROOT)
    logger = setup_logger(run_dir)
    logger.info(f"Loaded config from {args.config}")

    set_seed(config["seed"])
    logger.info(f"Global seed fixed at {config['seed']}")

    writer = None
    if config["use_tensorboard"]:
        tb_dir = run_dir / config["tensorboard_subdir"]
        writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"TensorBoard logging to {tb_dir}")

    g = torch.Generator()
    g.manual_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    use_auto_cast = True if torch.amp.autocast_mode.is_autocast_available(device_type="cuda") else False
    
    # ---- Data ----
    logger.info("Loading dataset")
    train_dataset = FIFASequenceDataset(
        data_dir=config["data_dir"],
        match_files=config["train_matches"],
        target_frames=config["target_frames"],
    )
    train_sampler = samplers.MPerClassSampler(
        labels=train_dataset.coarse_label_ids,
        m=config["m_per_class"],
        batch_size=config["batch_size"],
        length_before_new_iter=len(train_dataset),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        drop_last=True,
        num_workers=config["num_workers"],
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_dataset = FIFASequenceDataset(
        data_dir=config["data_dir"],
        match_files=config["val_matches"],
        target_frames=config["target_frames"],
    )
    val_sampler = samplers.MPerClassSampler(
        labels=val_dataset.coarse_label_ids,
        m=config["m_per_class"],
        batch_size=config["batch_size"],
        length_before_new_iter=len(val_dataset),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        sampler=val_sampler,
        drop_last=True,
        num_workers=config["num_workers"],
        worker_init_fn=seed_worker,
        generator=g,
    )

    # ---- Model ----
    logger.info("Initializing SimplePlayEncoder")
    model = PlayEncoder(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        frame_layers=config["frame_layers"],
        play_layers=config["play_layers"],
        output_dim=config["output_dim"],
        max_frames=config["target_frames"],
        dropout=config["dropout"],
        pos_encoding="sinusoidal",
    ).to(device)
    print_model_summary(model, logger)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config["epochs"] * len(train_loader), eta_min=1e-6
    )

    loss_func = losses.SupConLoss(temperature=config["temperature"])

    logger.info("Starting training")
    best_val_loss = float("inf")

    
    scaler = torch.amp.GradScaler(device=str(device), enabled=use_auto_cast)
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0

        for step_idx, batch in enumerate(train_loader):
            features = batch["features"].to(device)
            weak_labels = batch["label"].to(device)

            optimizer.zero_grad()
            if use_auto_cast:
                with torch.autocast(device_type=device.type, enabled=use_auto_cast):
                    view_1 = augment_play(features)
                    view_2 = augment_play(features)

                    _, proj_1 = model(view_1)
                    _, proj_2 = model(view_2)

                    embeddings = torch.cat([proj_1, proj_2], dim=0)
                    embeddings = F.normalize(embeddings, dim=1)
                    batch_labels = torch.cat([weak_labels, weak_labels], dim=0)

                    loss = loss_func(embeddings, batch_labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["clip_max_norm"])
                    scaler.step(optimizer)
                    scaler.update()
            else:
                view_1 = augment_play(features)
                view_2 = augment_play(features)

                _, proj_1 = model(view_1)
                _, proj_2 = model(view_2)

                embeddings = torch.cat([proj_1, proj_2], dim=0)
                embeddings = F.normalize(embeddings, dim=1)
                batch_labels = torch.cat([weak_labels, weak_labels], dim=0)

                loss = loss_func(embeddings, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["clip_max_norm"])
                optimizer.step()
                
            scheduler.step()
            total_loss += loss.item()

            if writer is not None:
                global_step = epoch * len(train_loader) + step_idx
                writer.add_scalar("train/step_loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        avg_train_loss = total_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                weak_labels = batch["label"].to(device)

                view_1 = augment_play(features)
                view_2 = augment_play(features)

                _, proj_1 = model(view_1)
                _, proj_2 = model(view_2)

                embeddings = torch.cat([proj_1, proj_2], dim=0)
                embeddings = F.normalize(embeddings, dim=1)
                batch_labels = torch.cat([weak_labels, weak_labels], dim=0)

                loss = loss_func(embeddings, batch_labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.info(
            f"Epoch [{epoch + 1}/{config['epochs']}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Best Val: {best_val_loss:.4f}"
        )

        if writer is not None:
            writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
            writer.add_scalar("epoch/val_loss", avg_val_loss, epoch)
            writer.add_scalar("epoch/best_val_loss", best_val_loss, epoch)
        is_save_epoch = (epoch + 1) % config["save_every"] == 0 or (epoch + 1) == config["epochs"]
        if is_best or is_save_epoch:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            save_checkpoint(checkpoint, is_best, run_dir)
        
        if epochs_without_improvement >= config["early_stopping_patience"]:
            logger.info(f"Early stopping: no val improvement for {config['early_stopping_patience']} epochs (best: {best_val_loss:.4f})")
            break
    
    if writer is not None:
        writer.close()

    logger.info(f"Model saved at {run_dir}")


if __name__ == "__main__":
    main()