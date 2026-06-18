# =============================================================================
# training/train.py — Industrial-Grade Training Engine
# =============================================================================
# Reads everything from YAML configs — zero hardcoding.
# Features: AMP, AdamW, Warmup+Cosine LR, Early Stopping,
#           Gradient Clipping, CSV+TensorBoard logging, LR Finder.
#
# Usage (from project root):
#   python training/train.py                     # full training run
#   python training/train.py --skip-lr-finder    # skip LR range test
#   python training/train.py --resume            # resume from latest.pth
#   python training/train.py --overfit-test      # 128-sample overfit check
#
# Interview note:
#   "My training engine reads every hyperparameter from YAML. LR finder
#    runs before training to find optimal LR. AMP gives 2x GPU speedup.
#    Best checkpoint auto-saved with full training state for reproducibility."
# =============================================================================

import sys
import math
import logging
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

# Project imports
from datasets.dataset_loader  import get_dataloaders, load_config as load_data_cfg
from datasets.augmentation    import AugmentationManager, SoftCrossEntropyLoss
from models.resnet             import build_model
from training.lr_finder        import LRFinder
from training.monitor          import GradientMonitor
from utils.seed                import set_seed
from utils.logger              import TrainingLogger
from utils.metrics             import accuracy, compute_metrics
from utils.checkpoint          import save_checkpoint, load_checkpoint_with_hooks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG LOADERS
# =============================================================================

def load_configs() -> tuple[dict, dict, dict]:
    """Load and return (data_cfg, model_cfg, train_cfg)."""
    data_cfg  = load_data_cfg("configs/data_config.yaml")
    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/train_config.yaml") as f:
        train_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg, train_cfg


# =============================================================================
# LOSS
# =============================================================================

def build_criterion(train_cfg: dict) -> nn.Module:
    """
    Build loss function from config.
    Uses SoftCrossEntropyLoss — accepts both hard and soft labels.
    Required for Mixup / CutMix compatibility (Day 4).
    """
    loss_cfg = train_cfg["loss"]
    if loss_cfg["name"] == "cross_entropy":
        return SoftCrossEntropyLoss(
            smoothing=loss_cfg.get("label_smoothing", 0.1)
        )
    raise ValueError(f"Unknown loss: {loss_cfg['name']}")


# =============================================================================
# OPTIMIZER
# =============================================================================

def build_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    """Build AdamW optimizer from config."""
    opt_cfg = train_cfg["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr           = opt_cfg["lr"],
        weight_decay = opt_cfg["weight_decay"],
        betas        = tuple(opt_cfg["betas"]),
        eps          = opt_cfg["eps"],
    )


# =============================================================================
# SCHEDULER — Linear Warmup + Cosine Annealing
# =============================================================================

def build_scheduler(
    optimizer:      torch.optim.Optimizer,
    train_cfg:      dict,
    steps_per_epoch: int,
) -> LambdaLR:
    """
    Linear warmup for first warmup_epochs, then cosine annealing.

    Step-level scheduler (called every batch, not every epoch)
    for smooth LR curves.
    """
    sched_cfg      = train_cfg["scheduler"]
    total_epochs   = train_cfg["training"]["epochs"]
    warmup_epochs  = sched_cfg["warmup_epochs"]
    min_lr         = sched_cfg["min_lr"]
    base_lr        = train_cfg["optimizer"]["lr"]

    warmup_steps = warmup_epochs  * steps_per_epoch
    total_steps  = total_epochs   * steps_per_epoch
    min_ratio    = min_lr / base_lr

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup: 0 → 1
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing: 1 → min_ratio
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Patience-based early stopping on val_acc or val_loss."""

    def __init__(self, patience: int = 8, mode: str = "max"):
        self.patience  = patience
        self.mode      = mode
        self.counter   = 0
        self.best      = float("-inf") if mode == "max" else float("inf")
        self.triggered = False

    def step(self, metric: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            metric > self.best if self.mode == "max" else metric < self.best
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"  EarlyStopping: {self.counter}/{self.patience} "
                f"(best={self.best:.4f})"
            )
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# =============================================================================
# ONE EPOCH TRAIN
# =============================================================================

def train_one_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    scaler:     GradScaler,
    scheduler:  LambdaLR,
    monitor:    GradientMonitor,
    aug_manager: AugmentationManager,
    device:     torch.device,
    train_cfg:  dict,
    epoch:      int,
) -> tuple[float, float]:
    """
    Single training epoch with AMP + Mixup/CutMix + gradient clipping.

    Returns:
        (mean_loss, mean_accuracy) for the epoch
    """
    model.train()
    amp_enabled   = train_cfg["amp"]["enabled"]
    clip_max_norm = train_cfg["amp"]["grad_clip_max_norm"]
    log_every     = train_cfg["logging"]["log_every_n_batches"]

    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Mixup / CutMix ──
        images, soft_labels = aug_manager.apply(images, labels)

        optimizer.zero_grad()

        # ── AMP forward pass ──
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            logits = model(images)
            loss   = criterion(logits, soft_labels)   # soft labels here

        # ── AMP backward ──
        scaler.scale(loss).backward()

        # ── Gradient norm monitoring ──
        scaler.unscale_(optimizer)
        monitor.record(batch_idx)

        # ── Gradient clipping ──
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)

        # ── Optimizer + scaler step ──
        scaler.step(optimizer)
        scaler.update()

        # ── Scheduler step (per batch) ──
        scheduler.step()

        batch_loss = loss.item()
        batch_acc  = accuracy(logits.detach(), labels)   # hard labels for acc
        total_loss += batch_loss
        total_acc  += batch_acc
        n_batches  += 1

        if batch_idx % log_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"  Epoch {epoch} [{batch_idx:4d}/{len(loader)}] "
                f"loss={batch_loss:.4f} acc={batch_acc:.4f} lr={current_lr:.2e}"
            )

    # Log augmentation usage for epoch
    aug_manager.log_usage(epoch)

    return total_loss / n_batches, total_acc / n_batches


# =============================================================================
# ONE EPOCH VALIDATE
# =============================================================================

@torch.no_grad()
def validate(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    num_classes: int,
) -> tuple[float, float, dict]:
    """
    Validation pass — no gradients, compute full metrics.

    Returns:
        (mean_loss, mean_accuracy, metrics_dict)
    """
    model.eval()

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        soft_labels = F.one_hot(labels, num_classes).float()
        logits = model(images)
        loss   = criterion(logits, soft_labels)

        # ── Day 5: NaN / Inf guard ──
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                f"NaN/Inf loss detected during validation! "
                f"logit range: [{logits.min():.2f}, {logits.max():.2f}]. "
                f"Check AMP settings and LR magnitude."
            )

        total_loss += loss.item()
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mean_loss = total_loss / len(loader)
    mean_acc  = (all_preds == all_labels).mean()
    metrics   = compute_metrics(all_preds, all_labels, num_classes)

    return mean_loss, float(mean_acc), metrics


# =============================================================================
# OVERFIT TEST — confirms model has enough capacity
# =============================================================================

def overfit_test(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    train_cfg: dict,
    n_samples: int = 128,
    max_epochs: int = 10,
) -> None:
    """
    Train on n_samples for max_epochs — expect ~100% train accuracy.
    If not reached: model capacity issue or data pipeline bug.
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"OVERFIT TEST — {n_samples} samples, {max_epochs} epochs")
    logger.info(f"Expected: ~100% train acc within {max_epochs} epochs")
    logger.info(f"{'='*50}")

    # Grab first n_samples
    images_list, labels_list = [], []
    for images, labels in loader:
        images_list.append(images)
        labels_list.append(labels)
        if sum(x.shape[0] for x in images_list) >= n_samples:
            break

    images = torch.cat(images_list)[:n_samples].to(device)
    labels = torch.cat(labels_list)[:n_samples].to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler    = GradScaler(enabled=train_cfg["amp"]["enabled"])
    amp_on    = train_cfg["amp"]["enabled"]

    model.train()
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        with autocast(enabled=amp_on):
            logits = model(images)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc = accuracy(logits.detach(), labels)
        logger.info(f"  Epoch {epoch:2d} | loss={loss.item():.4f} | acc={acc:.4f}")

        if acc >= 0.99:
            logger.info(f"  ✓ Overfit test passed at epoch {epoch}")
            return

    logger.warning(
        "  ⚠ Overfit test FAILED — model did not reach ~100% on 128 samples.\n"
        "  Check: model capacity, data pipeline, label correctness."
    )


# =============================================================================
# DAY 5 — LOSS CURVE SAVER
# =============================================================================

def _save_loss_curves(csv_path: str) -> None:
    """
    Read training CSV log and save loss + accuracy curve plots.
    Called automatically at end of training.
    Saved to: logs/loss_curves/training_curves.png
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(csv_path)
        if len(df) < 2:
            return

        Path("logs/loss_curves").mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(df["epoch"], df["train_loss"], label="Train", color="#3498db", lw=2)
        axes[0].plot(df["epoch"], df["val_loss"],   label="Val",   color="#e74c3c", lw=2)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

        # Accuracy
        axes[1].plot(df["epoch"], df["train_acc"], label="Train", color="#3498db", lw=2)
        axes[1].plot(df["epoch"], df["val_acc"],   label="Val",   color="#e74c3c", lw=2)
        best_row = df.loc[df["val_acc"].idxmax()]
        axes[1].axvline(best_row["epoch"], color="green", linestyle="--", lw=1.5,
                        label=f'Best val: {best_row["val_acc"]:.4f}')
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.suptitle("Training Curves — Vision System Capstone v3",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = "logs/loss_curves/training_curves.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Loss curves saved → {out_path}")

    except Exception as e:
        logger.warning(f"Could not save loss curves: {e}")


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train(args: argparse.Namespace) -> None:
    """Full training run."""

    # ── Load configs ──
    data_cfg, model_cfg, train_cfg = load_configs()
    t_cfg = train_cfg["training"]

    # ── Seed ──
    set_seed(t_cfg["seed"])

    # ── Device ──
    device = torch.device(
        "cuda" if torch.cuda.is_available() and t_cfg["device"] == "cuda"
        else "cpu"
    )
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──
    loaders     = get_dataloaders(data_cfg)
    num_classes = data_cfg["dataset"]["num_classes"]

    # ── Model ──
    model = build_model().to(device)
    logger.info(f"Model: ResNet-{model_cfg['arch']['depth']} | params={model.count_params():,}")

    # ── Loss / Optimizer ──
    criterion = build_criterion(train_cfg)
    optimizer = build_optimizer(model, train_cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=train_cfg["amp"]["enabled"])

    # ── Overfit test (quick capacity check) ──
    if args.overfit_test:
        overfit_test(model, loaders["train"], criterion, device, train_cfg)
        return

    # ── LR Finder ──
    if not args.skip_lr_finder:
        logger.info("\nRunning LR Finder before training...")
        finder     = LRFinder(model, optimizer, criterion, device)
        best_lr    = finder.run(loaders["train"], train_cfg)
        # Update optimizer LR with finder result
        for pg in optimizer.param_groups:
            pg["lr"] = best_lr
        logger.info(f"LR set to {best_lr:.2e} from finder\n")

    # ── Scheduler ──
    scheduler = build_scheduler(optimizer, train_cfg, len(loaders["train"]))

    # ── Resume ──
    start_epoch = 1
    best_acc    = 0.0
    if args.resume:
        ckpt_path = Path(train_cfg["checkpoint"]["save_dir"]) / "latest.pth"
        if ckpt_path.exists():
            ckpt        = load_checkpoint_with_hooks(str(ckpt_path), model, optimizer, scaler, device)
            start_epoch = ckpt["epoch"] + 1
            best_acc    = ckpt["best_acc"]
            logger.info(f"Resumed from epoch {start_epoch}")
        else:
            logger.warning(f"No checkpoint at {ckpt_path} — starting fresh")

    # ── Gradient Monitor ──
    monitor = GradientMonitor(model, log_every_n=train_cfg["logging"]["log_every_n_batches"])

    # ── Augmentation Manager (Mixup + CutMix) ──
    aug_manager = AugmentationManager.from_config(data_cfg)
    logger.info(
        f"Augmentation: Mixup p={aug_manager.p_mixup} | "
        f"CutMix p={aug_manager.p_cutmix}"
    )

    # ── Logger ──
    tlogger = TrainingLogger(
        csv_path        = train_cfg["logging"]["csv_path"],
        tensorboard_dir = train_cfg["logging"]["tensorboard_dir"],
    )

    # ── Early Stopping ──
    es_cfg = train_cfg["early_stopping"]
    early_stop = EarlyStopping(
        patience = es_cfg["patience"],
        mode     = es_cfg["mode"],
    ) if es_cfg["enabled"] else None

    # ── Log config snapshot ──
    Path("logs").mkdir(exist_ok=True)
    with open("logs/train_config_snapshot.yaml", "w") as f:
        yaml.dump(train_cfg, f)
    logger.info("Config snapshot saved → logs/train_config_snapshot.yaml\n")

    # ══════════════════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════════
    logger.info(f"Training: epochs={t_cfg['epochs']} | device={device}")
    logger.info("=" * 60)

    for epoch in range(start_epoch, t_cfg["epochs"] + 1):
        logger.info(f"\nEpoch {epoch}/{t_cfg['epochs']}")
        monitor.reset()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion,
            scaler, scheduler, monitor, aug_manager, device, train_cfg, epoch,
        )

        # Gradient summary
        monitor.summarize(epoch)

        # Validate
        val_loss, val_acc, metrics = validate(
            model, loaders["val"], criterion, device, num_classes
        )

        # Current LR
        current_lr = scheduler.get_last_lr()[0]

        # Log
        tlogger.log(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

        logger.info(
            f"Epoch {epoch} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"F1={metrics['f1_macro']:.4f} | lr={current_lr:.2e}"
        )

        # Checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        save_checkpoint(
            model, optimizer, scaler, epoch, best_acc,
            {"data": data_cfg, "model": model_cfg, "train": train_cfg},
            train_cfg["checkpoint"]["save_dir"],
            is_best=is_best,
        )

        # Early stopping
        if early_stop and early_stop.step(val_acc):
            logger.info(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # ══════════════════════════════════════════════════════════════════════
    # TRAINING COMPLETE
    # ══════════════════════════════════════════════════════════════════════
    tlogger.close()
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete | Best val_acc: {best_acc:.4f}")
    logger.info(f"Best checkpoint → {train_cfg['checkpoint']['save_dir']}best.pth")

    # ── Day 5: Save loss curves ──
    _save_loss_curves(train_cfg["logging"]["csv_path"])


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet on Indian Food dataset")
    parser.add_argument("--skip-lr-finder", action="store_true",
                        help="Skip LR range test (use lr from train_config.yaml)")
    parser.add_argument("--resume",         action="store_true",
                        help="Resume from logs/checkpoints/latest.pth")
    parser.add_argument("--overfit-test",   action="store_true",
                        help="Run 128-sample overfit check only (no full training)")
    args = parser.parse_args()
    train(args)
