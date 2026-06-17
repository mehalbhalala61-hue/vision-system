# =============================================================================
# training/stability.py — Training Stability Helpers (Day 5)
# =============================================================================
# Contains:
#   check_nan_inf()        — NaN/Inf detection in logits + loss
#   run_overfit_test()     — 128-sample capacity check
#   save_loss_curves()     — plot train/val curves from CSV
#   check_weight_init()    — verify Kaiming + zero-init BN
#   StabilityTracker       — tracks stability metrics across epochs
#
# Used by:
#   training/train.py              (loss curve save at end)
#   notebooks/day5_stability.ipynb (interactive analysis)
#
# Interview note:
#   "I maintained a stability tracker throughout training — gradient
#    norms, NaN detection, overfit test. Debug log has 8 entries.
#    Debugging process: hypothesis → minimal reproduction → fix → document."
# =============================================================================

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


# =============================================================================
# NaN / Inf DETECTION
# =============================================================================

def check_nan_inf(
    logits:    torch.Tensor,
    loss:      torch.Tensor,
    batch_idx: int,
    epoch:     int,
) -> dict:
    """
    Check logits and loss for NaN/Inf values.
    Call after every forward pass during training.

    Returns dict: has_nan, has_inf, logit_min, logit_max, loss_val
    """
    result = {
        "has_nan":   False,
        "has_inf":   False,
        "logit_min": float(logits.min().item()),
        "logit_max": float(logits.max().item()),
        "loss_val":  float(loss.item()),
        "epoch":     epoch,
        "batch":     batch_idx,
    }

    if torch.isnan(logits).any():
        result["has_nan"] = True
        logger.error(
            f"NaN in LOGITS | epoch={epoch} batch={batch_idx} | "
            f"range=[{result['logit_min']:.2f}, {result['logit_max']:.2f}]\n"
            f"  Causes: LR too high | AMP overflow | bad weight init\n"
            f"  Fix: run LR finder, reduce LR 10x and restart"
        )

    if torch.isinf(logits).any():
        result["has_inf"] = True
        logger.error(
            f"Inf in LOGITS | epoch={epoch} batch={batch_idx}\n"
            f"  AMP float16 overflow — logit magnitude too large\n"
            f"  Fix: lower LR + gradient clipping max_norm=1.0"
        )

    if torch.isnan(loss):
        result["has_nan"] = True
        logger.error(
            f"NaN in LOSS | epoch={epoch} batch={batch_idx}\n"
            f"  debug_log Bug #2: AMP + high LR = float16 overflow\n"
            f"  Fix: restart training with LR from lr_finder_plot.png"
        )

    return result


# =============================================================================
# OVERFIT TEST
# =============================================================================

def run_overfit_test(
    model:       nn.Module,
    loader:      torch.utils.data.DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    num_classes: int,
    n_samples:   int  = 128,
    max_epochs:  int  = 10,
    amp_enabled: bool = True,
) -> dict:
    """
    Train on n_samples for max_epochs — expect ~100% accuracy.

    If model cannot overfit 128 samples:
        model capacity too small | wrong labels | LR too low

    Returns dict: passed, final_acc, final_loss, epochs_taken, history
    """
    logger.info(f"\n{'='*55}")
    logger.info(f"OVERFIT TEST — {n_samples} samples, max {max_epochs} epochs")
    logger.info(f"Expected: train acc >= 0.99 within {max_epochs} epochs")
    logger.info(f"{'='*55}")

    # Collect first n_samples
    images_list, labels_list = [], []
    collected = 0
    for imgs, lbls in loader:
        take = min(imgs.size(0), n_samples - collected)
        images_list.append(imgs[:take])
        labels_list.append(lbls[:take])
        collected += take
        if collected >= n_samples:
            break

    images = torch.cat(images_list).to(device)
    labels = torch.cat(labels_list).to(device)

    # High LR intentional — we WANT to overfit fast
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    scaler    = GradScaler(enabled=amp_enabled)

    results = {
        "passed":       False,
        "final_acc":    0.0,
        "final_loss":   float("inf"),
        "epochs_taken": max_epochs,
        "history":      [],
    }

    model.train()
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()

        with autocast(enabled=amp_enabled):
            logits      = model(images)
            soft_labels = F.one_hot(labels, num_classes).float()
            loss        = criterion(logits, soft_labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean().item()

        results["history"].append({
            "epoch": epoch,
            "loss":  round(loss.item(), 4),
            "acc":   round(acc, 4),
        })
        logger.info(f"  Epoch {epoch:2d} | loss={loss.item():.4f} | acc={acc:.4f}")

        if acc >= 0.99:
            results["passed"]       = True
            results["epochs_taken"] = epoch
            results["final_acc"]    = acc
            results["final_loss"]   = loss.item()
            logger.info(f"  PASSED at epoch {epoch} — pipeline is correct.")
            break

    if not results["passed"]:
        results["final_acc"]  = results["history"][-1]["acc"]
        results["final_loss"] = results["history"][-1]["loss"]
        logger.warning(
            f"  FAILED — acc={results['final_acc']:.4f} after {max_epochs} epochs\n"
            f"  Check: classes.txt, label indices, model output shape={num_classes}"
        )

    return results


# =============================================================================
# LOSS CURVE SAVER
# =============================================================================

def save_loss_curves(
    csv_path:  str = "logs/training_log.csv",
    save_path: str = "logs/loss_curves/training_curves.png",
) -> bool:
    """
    Read training CSV and save loss + accuracy + LR plots.

    Returns True if saved, False if CSV not found.
    Called automatically at end of train.py.
    Can also be called manually from notebook.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not installed")
        return False

    if not Path(csv_path).exists():
        logger.warning(f"CSV not found: {csv_path} — run training first")
        return False

    df = pd.read_csv(csv_path)
    if len(df) < 2:
        logger.warning("Less than 2 epochs — skipping plot")
        return False

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train", color="#3498db", lw=2)
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val",   color="#e74c3c", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(df["epoch"], df["train_acc"], label="Train", color="#3498db", lw=2)
    axes[1].plot(df["epoch"], df["val_acc"],   label="Val",   color="#e74c3c", lw=2)
    best_row = df.loc[df["val_acc"].idxmax()]
    axes[1].axvline(
        best_row["epoch"], color="#27ae60", linestyle="--", lw=1.5,
        label=f"Best: {best_row['val_acc']:.4f} (ep {int(best_row['epoch'])})"
    )
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve"); axes[1].legend(); axes[1].grid(alpha=0.3)

    # LR
    axes[2].plot(df["epoch"], df["lr"].astype(float), color="#2ecc71", lw=2)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
    axes[2].set_title("LR Schedule"); axes[2].set_yscale("log"); axes[2].grid(alpha=0.3)

    # Overfit diagnosis
    if len(df) >= 5:
        gap = (df.tail(5)["train_acc"] - df.tail(5)["val_acc"]).mean()
        diag = ("Overfitting" if gap > 0.15 else
                "Mild overfit" if gap > 0.08 else "Healthy")
        fig.suptitle(
            f"Training Curves | Best val={best_row['val_acc']:.4f} | {diag} (gap={gap:.3f})",
            fontsize=12, fontweight="bold"
        )
    else:
        fig.suptitle("Training Curves — Vision System Capstone v3",
                     fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Loss curves saved -> {save_path}")
    return True


# =============================================================================
# WEIGHT INIT CHECK
# =============================================================================

def check_weight_init(model: nn.Module) -> dict:
    """
    Verify Kaiming He init on Conv + zero-init on last BN of residual blocks.
    Returns dict with all checks.
    """
    from models.blocks import BasicBlock, Bottleneck

    conv_stds, bn_weights, bn_biases = [], [], []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_stds.append(m.weight.data.std().item())
        elif isinstance(m, nn.BatchNorm2d):
            bn_weights.append(m.weight.data.mean().item())
            bn_biases.append(m.bias.data.mean().item())

    zero_init_count = 0
    for m in model.modules():
        if isinstance(m, BasicBlock) and hasattr(m, "bn2"):
            if m.bn2.weight.data.abs().max().item() < 1e-6:
                zero_init_count += 1
        elif isinstance(m, Bottleneck) and hasattr(m, "bn3"):
            if m.bn3.weight.data.abs().max().item() < 1e-6:
                zero_init_count += 1

    bn_gamma_ok = abs(np.mean(bn_weights) - 1.0) < 0.1
    bn_beta_ok  = abs(np.mean(bn_biases))         < 0.1
    zero_ok     = zero_init_count > 0

    results = {
        "conv_std_mean":   round(float(np.mean(conv_stds)),  4),
        "bn_gamma_mean":   round(float(np.mean(bn_weights)), 4),
        "bn_beta_mean":    round(float(np.mean(bn_biases)),  4),
        "zero_init_count": zero_init_count,
        "bn_gamma_ok":     bn_gamma_ok,
        "bn_beta_ok":      bn_beta_ok,
        "zero_init_ok":    zero_ok,
        "all_passed":      bn_gamma_ok and bn_beta_ok and zero_ok,
    }

    logger.info("Weight init check:")
    logger.info(f"  Conv std    : {results['conv_std_mean']:.4f} (Kaiming He)")
    logger.info(f"  BN gamma    : {results['bn_gamma_mean']:.4f} (want ~1.0) {'OK' if bn_gamma_ok else 'FAIL'}")
    logger.info(f"  BN beta     : {results['bn_beta_mean']:.4f} (want ~0.0) {'OK' if bn_beta_ok else 'FAIL'}")
    logger.info(f"  Zero-init BN: {zero_init_count} blocks {'OK' if zero_ok else 'FAIL'}")
    return results


# =============================================================================
# STABILITY TRACKER
# =============================================================================

class StabilityTracker:
    """
    Collects per-epoch stability metrics.

    Usage in train loop:
        tracker = StabilityTracker()
        tracker.record_epoch(epoch, train_loss, val_loss, grad_summary, nan_count)
        tracker.report()   # at end of training
    """

    def __init__(self):
        self.epochs:         list = []
        self.train_losses:   list = []
        self.val_losses:     list = []
        self.nan_counts:     list = []
        self.grad_summaries: list = []

    def record_epoch(
        self,
        epoch:        int,
        train_loss:   float,
        val_loss:     float,
        grad_summary: dict = None,
        nan_count:    int  = 0,
    ) -> None:
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.nan_counts.append(nan_count)
        self.grad_summaries.append(grad_summary or {})

    def report(self) -> dict:
        if not self.epochs:
            return {}

        total_nans = sum(self.nan_counts)
        loss_gaps  = [t - v for t, v in zip(self.train_losses, self.val_losses)]
        final_gap  = loss_gaps[-1] if loss_gaps else 0.0
        best_val   = min(self.val_losses)

        logger.info("\n" + "="*55)
        logger.info("STABILITY REPORT")
        logger.info("="*55)
        logger.info(f"  Epochs trained : {len(self.epochs)}")
        logger.info(f"  NaN events     : {total_nans} {'OK' if total_nans == 0 else 'WARN'}")
        logger.info(f"  Best val loss  : {best_val:.4f}")
        logger.info(f"  Final loss gap : {final_gap:.4f} "
                    f"({'healthy' if final_gap < 0.15 else 'possible overfit'})")

        return {
            "epochs":     len(self.epochs),
            "total_nans": total_nans,
            "best_val":   round(best_val,  4),
            "final_gap":  round(final_gap, 4),
            "stable":     total_nans == 0 and final_gap < 0.20,
        }
