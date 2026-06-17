# =============================================================================
# utils/checkpoint.py — Checkpoint Save / Load
# =============================================================================
# Key function: load_checkpoint_with_hooks()
# Re-registers Grad-CAM hook on layer4 after every torch.load() call.
# v3 fix: hooks silently detach after checkpoint reload → blank heatmaps.
#
# Interview note:
#   "I noticed Grad-CAM heatmaps were blank after loading a checkpoint.
#    Debugged: forward hooks detach silently after torch.load(). Fix:
#    load_checkpoint_with_hooks() re-registers the hook every time."
# =============================================================================

import logging
from pathlib import Path
import torch
from models.resnet import ResNet

logger = logging.getLogger(__name__)


def save_checkpoint(
    model:      ResNet,
    optimizer:  torch.optim.Optimizer,
    scaler:     torch.cuda.amp.GradScaler,
    epoch:      int,
    best_acc:   float,
    cfg:        dict,
    save_dir:   str,
    is_best:    bool = False,
) -> None:
    """
    Save full training state.
    Saves latest.pth always; best.pth only when is_best=True.

    Checkpoint keys:
        epoch, model_state_dict, optimizer_state_dict,
        scaler_state_dict, best_acc, config
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    state = {
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":   scaler.state_dict(),
        "best_acc":            best_acc,
        "config":              cfg,
    }

    latest_path = Path(save_dir) / "latest.pth"
    torch.save(state, latest_path)

    if is_best:
        best_path = Path(save_dir) / "best.pth"
        torch.save(state, best_path)
        logger.info(f"  ★ New best ({best_acc:.4f}) → {best_path}")

    logger.debug(f"Checkpoint saved → {latest_path}")


def load_checkpoint_with_hooks(
    checkpoint_path: str,
    model:           ResNet,
    optimizer:       torch.optim.Optimizer = None,
    scaler:          torch.cuda.amp.GradScaler = None,
    device:          torch.device = torch.device("cpu"),
) -> dict:
    """
    Load checkpoint AND re-register Grad-CAM hook on layer4.

    Always use this instead of model.load_state_dict() directly.
    Hooks detach silently after torch.load() — this fixes blank heatmaps.

    Args:
        checkpoint_path : path to .pth file
        model           : ResNet instance (must be on device already)
        optimizer       : optional — restored if provided
        scaler          : optional — restored if provided
        device          : target device

    Returns:
        checkpoint dict (epoch, best_acc, config, ...)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model.load_state_dict(ckpt["model_state_dict"])

    # Re-register hook — this is the v3 fix
    model._register_gradcam_hook()
    logger.info(f"Loaded checkpoint: epoch={ckpt['epoch']} | best_acc={ckpt['best_acc']:.4f}")
    logger.info("Grad-CAM hook re-registered on layer4")

    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return ckpt
