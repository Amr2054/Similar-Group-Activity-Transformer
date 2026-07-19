from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses
from pytorch_metric_learning import samplers

from data import FIFASequenceDataset
from Hierarchical_Play_Encoder.model import HierarchicalPlayEncoder
from utils import setup_logger, save_checkpoint, load_config,print_model_summary,setup_run_dir
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def augment_play(features, p_flip_y=0.5, p_mask_player=0.15, jitter_std=0.02):
    """
    features: (B, 100, 23, 9) ->
    [x, y, z, speed, visibility, is_attacking, dist_to_goal, is_home, is_away]
    """
    aug = features.clone()
    B = aug.shape[0]

    # 1. Tactical mirroring on Y (index 1). dist_to_goal (index 6) is
    # symmetric under y -> -y, so it does not need recomputing.
    flip_mask = torch.rand(B, device=features.device) < p_flip_y
    aug[flip_mask, :, :, 1] = aug[flip_mask, :, :, 1] * -1.0

    # 2. Spatial jitter — only on the continuous x/y channels. Never add
    # noise to visibility/is_attacking/is_home/is_away/dist_to_goal; they're
    # categorical/derived and noise would corrupt their meaning.
    noise = torch.randn_like(aug[..., 0:2]) * jitter_std
    aug[..., 0:2] = torch.clamp(aug[..., 0:2] + noise, min=-1.0, max=1.0)

    # 3. Agent dropout — zero the whole 9-dim token (never the ball, index 0).
    # This reuses the same "missing player" convention as dataGenerator, so
    # the key_padding_mask in HierarchicalPlayEncoder picks it up for free.
    drop_mask = torch.rand(B, 1, 23, 1, device=features.device) > p_mask_player
    drop_mask[:, :, 0, :] = True
    aug = aug * drop_mask

    return aug

def main():
    # Load configuration
    config = load_config("../Configs/hier_model_config.yaml")
    ROOT = Path(__file__).resolve().parent
    run_dir = setup_run_dir(ROOT)
    # Setup components using config values
    logger = setup_logger(run_dir)

    SEED = config['training'].get('seed', 42)
    set_seed(SEED)
    logger.info(f"Global seed fixed at {SEED}")

    g = torch.Generator()
    g.manual_seed(SEED)

    logger.info("Initializing Training Pipeline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")


    writer = SummaryWriter(log_dir=run_dir)
    global_step = 0


    # Dataset & DataLoader
    logger.info("Loading Dataset")
    train_dataset = FIFASequenceDataset(
        data_dir=config['data']['data_dir'],
        match_files=config['data']['train_matches'],  # Pass training split
        target_frames=config['data']['target_frames']
    )
    # print(sorted(list(train_dataset.coarse_labels)))
    train_sampler = samplers.MPerClassSampler(
        labels=train_dataset.coarse_label_ids,
        m=config['training']['m_per_class'],
        batch_size=config['training']['batch_size'],
        length_before_new_iter=len(train_dataset)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,  # replaces shuffle=True — can't use both
        drop_last=True,
        num_workers=config['training']['num_workers'],
        worker_init_fn=seed_worker,
        generator=g
    )

    val_dataset = FIFASequenceDataset(
        data_dir=config['data']['data_dir'],
        match_files=config['data']['val_matches'],  # Pass validation split
        target_frames=config['data']['target_frames']
    )

    val_sampler = samplers.MPerClassSampler(
        labels=val_dataset.coarse_label_ids,
        m=config['training']['m_per_class'],
        batch_size=config['training']['batch_size'],
        length_before_new_iter=len(val_dataset)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,  # replaces shuffle=False
        drop_last=True,
        num_workers=config['training']['num_workers'],
        worker_init_fn=seed_worker,
        generator=g
    )
    # Model & Optimizer
    logger.info("Initializing Hierarchical Model")
    model = HierarchicalPlayEncoder(
        d_model=config['model']['embed_dim'],
        n_heads=config['model']['n_heads'],
        frame_layers=config['model']['frame_layers'],
        play_layers=config['model']['play_layers']
    ).to(device)
    print_model_summary(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'] * len(train_loader),  # Total number of steps
        eta_min=1e-6
    )

    logger.info("\nStarting Training\n")
    best_val_loss = float('inf')

    contrastive_loss_func = losses.SupConLoss(temperature=config['training']['temperature'])

    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)  # (B, 100, 23, 9)
            weak_labels = batch['label'].to(device)  # (B,) coarse label ids

            optimizer.zero_grad()

            view_1 = augment_play(features)
            view_2 = augment_play(features)

            try:
                _, proj_1 = model(view_1)
                _, proj_2 = model(view_2)
            except Exception as e:
                logger.error(f"Forward pass failed at batch {batch_idx}: {str(e)}")
                break

            embeddings = torch.cat([proj_1, proj_2], dim=0)
            batch_labels = torch.cat([weak_labels, weak_labels], dim=0)

            loss = contrastive_loss_func(embeddings, batch_labels)

            # Backward and Optimize
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config['training'].get('grad_clip', 5.0)
            )
            optimizer.step()

            with torch.no_grad():
                embedding_std = embeddings.std(dim=0).mean().item()

            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            writer.add_scalar('Diagnostics/grad_norm', grad_norm.item(), global_step)
            writer.add_scalar('Diagnostics/embedding_std', embedding_std, global_step) # TODO: Watch
            global_step += 1

            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # EVALUATION PHASE
        model.eval()  # Freeze model layers (like BatchNorm/Dropout)
        total_val_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                features = batch['features'].to(device)  # (B, 100, 23, 9)
                weak_labels = batch['label'].to(device)  # (B,) coarse label ids

                view_1 = augment_play(features)
                view_2 = augment_play(features)

                try:
                    _, proj_1 = model(view_1)
                    _, proj_2 = model(view_2)
                except Exception as e:
                    logger.error(f"Forward pass failed at batch {batch_idx}: {str(e)}")
                    break

                embeddings = torch.cat([proj_1, proj_2], dim=0)
                batch_labels = torch.cat([weak_labels, weak_labels], dim=0)

                loss = contrastive_loss_func(embeddings, batch_labels)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Check if this is our best epoch so far
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        # Log epoch results
        logger.info(
            f"Epoch [{epoch + 1}/{config['training']['epochs']}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Best Val: {best_val_loss:.4f}"
        )

        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss,
            'config': config
        }
        save_checkpoint(checkpoint, is_best, run_dir)
    logger.info(f"Model Saved at {run_dir}")

    writer.close() # tensorboard --logdir runs


if __name__ == "__main__":
    main()