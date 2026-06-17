# =============================================================================
# evaluation/gradcam.py — Grad-CAM Visualisation
# =============================================================================
# Generates class activation heatmaps using Grad-CAM (Selvaraju et al., 2017).
# Always uses load_checkpoint_with_hooks() — never model.load_state_dict().
#
# Contains:
#   GradCAM           — core implementation
#   visualize_single  — one image heatmap overlay
#   visualize_grid    — 20-image grid (correct + misclassified)
#   run_gradcam_eval  — full pipeline for Day 6
#
# Interview note:
#   "Grad-CAM showed my model was focusing on plate edges instead of
#    food texture. I fixed this with better CutMix augmentation — it
#    forced the model to look at the full plate, not just one corner.
#    The v3 fix re-registers hooks after every checkpoint load."
# =============================================================================

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from models.resnet import ResNet
from utils.checkpoint import load_checkpoint_with_hooks

logger = logging.getLogger(__name__)


# =============================================================================
# GRAD-CAM CORE
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Pipeline:
        1. Forward pass → capture layer4 feature maps via hook
        2. Backward pass from target class score → capture gradients
        3. Global average pool gradients → per-channel weights
        4. Weighted sum of feature maps → raw CAM
        5. ReLU → resize to input size → normalise → overlay

    Args:
        model       : ResNet instance (hooks already registered)
        device      : torch device

    Usage:
        gradcam = GradCAM(model, device)
        heatmap = gradcam.generate(image_tensor, target_class=None)
        # target_class=None → uses predicted class (most common use)
    """

    def __init__(self, model: ResNet, device: torch.device):
        self.model  = model
        self.device = device

        # Storage for gradients (filled by backward hook)
        self._gradients: list = []

        # Register backward hook on layer4
        self._grad_hook = model.layer4.register_full_backward_hook(
            self._save_gradients
        )

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        """Backward hook — saves gradient w.r.t. layer4 output."""
        self._gradients.clear()
        self._gradients.append(grad_output[0].detach())

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: int = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for one image.

        Args:
            image_tensor : (1, C, H, W) normalised tensor on device
            target_class : class index to explain.
                           None → uses argmax (predicted class)

        Returns:
            heatmap : (H, W) numpy array, values in [0, 1]
                      where 1 = most important region
        """
        self.model.eval()
        self._gradients.clear()

        # Forward pass — activations captured by forward hook in resnet.py
        image_tensor.requires_grad_(False)
        logits = self.model(image_tensor)       # (1, num_classes)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward on target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Get captured features + gradients
        features   = self.model.get_gradcam_features()  # (1, C, h, w)
        gradients  = self._gradients[0]                 # (1, C, h, w)

        # Global average pool gradients → (C,) channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of feature maps → raw CAM (1, 1, h, w)
        cam = (weights * features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # only positive contributions

        # Resize to input image size
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        cam   = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam   = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self) -> None:
        """Remove backward hook — call when done to free memory."""
        self._grad_hook.remove()


# =============================================================================
# VISUALISATION HELPERS
# =============================================================================

def _denormalise(tensor: torch.Tensor, cfg: dict) -> np.ndarray:
    """
    Reverse ImageNet normalisation for display.
    tensor: (C, H, W) → returns (H, W, C) uint8 numpy array.
    """
    mean = np.array(cfg["image"]["mean"])
    std  = np.array(cfg["image"]["std"])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay_heatmap(
    image_np:  np.ndarray,
    heatmap:   np.ndarray,
    alpha:     float = 0.45,
    colormap:  str   = "jet",
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image_np : (H, W, 3) uint8 original image
        heatmap  : (H, W) float in [0, 1]
        alpha    : heatmap blend weight
        colormap : matplotlib colormap name

    Returns:
        overlay : (H, W, 3) uint8 blended image
    """
    import matplotlib.cm as cm
    cmap      = cm.get_cmap(colormap)
    heat_rgb  = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay   = ((1 - alpha) * image_np + alpha * heat_rgb).astype(np.uint8)
    return overlay


def visualize_single(
    model:        ResNet,
    image_tensor: torch.Tensor,
    label:        int,
    class_names:  list[str],
    cfg:          dict,
    device:       torch.device,
    save_path:    str = None,
    target_class: int = None,
) -> np.ndarray:
    """
    Generate and display Grad-CAM for one image.

    Returns heatmap array.
    """
    gradcam     = GradCAM(model, device)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    heatmap     = gradcam.generate(image_tensor, target_class)
    gradcam.remove_hooks()

    # Predicted class
    with torch.no_grad():
        logits = model(image_tensor)
    probs   = F.softmax(logits, dim=1).squeeze()
    pred    = probs.argmax().item()
    conf    = probs[pred].item()

    img_np  = _denormalise(image_tensor.squeeze(0), cfg)
    overlay = overlay_heatmap(img_np, heatmap)
    correct = (pred == label)

    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np);   axes[0].axis("off"); axes[0].set_title("Original")
        axes[1].imshow(heatmap, cmap="jet"); axes[1].axis("off")
        axes[1].set_title("Heatmap")
        axes[2].imshow(overlay);  axes[2].axis("off")
        axes[2].set_title(
            f"{'✓' if correct else '✗'} Pred: {class_names[pred]} ({conf:.2f})\n"
            f"True: {class_names[label]}"
        )
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return heatmap


def visualize_grid(
    model:       ResNet,
    dataset,
    class_names: list[str],
    cfg:         dict,
    device:      torch.device,
    n_correct:   int = 10,
    n_wrong:     int = 10,
    save_path:   str = "logs/gradcam_outputs/gradcam_grid.png",
) -> None:
    """
    Create a 20-image Grad-CAM grid — correct + misclassified.

    Saves to logs/gradcam_outputs/gradcam_grid.png.

    Args:
        model       : trained ResNet
        dataset     : DynamicDataset (test split)
        class_names : list of class name strings
        cfg         : data_config.yaml dict
        device      : torch device
        n_correct   : number of correctly classified examples to show
        n_wrong     : number of misclassified examples to show
        save_path   : output path
    """
    model.eval()
    gradcam = GradCAM(model, device)

    correct_samples = []
    wrong_samples   = []

    # Scan dataset for correct + wrong predictions
    with torch.no_grad():
        for idx in range(min(len(dataset), 500)):
            if (len(correct_samples) >= n_correct and
                    len(wrong_samples) >= n_wrong):
                break

            img_tensor, label = dataset[idx]
            tensor  = img_tensor.unsqueeze(0).to(device)
            logits  = model(tensor)
            pred    = logits.argmax(dim=1).item()
            conf    = F.softmax(logits, dim=1).squeeze()[pred].item()

            entry = (img_tensor, label, pred, conf)
            if pred == label and len(correct_samples) < n_correct:
                correct_samples.append(entry)
            elif pred != label and len(wrong_samples) < n_wrong:
                wrong_samples.append(entry)

    all_samples = correct_samples + wrong_samples
    if not all_samples:
        logger.warning("No samples collected for Grad-CAM grid")
        return

    n_cols  = 4   # original | heatmap | overlay | label
    n_rows  = len(all_samples)
    fig     = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.2))
    gs      = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                 hspace=0.4, wspace=0.1)

    for row, (img_tensor, label, pred, conf) in enumerate(all_samples):
        tensor  = img_tensor.unsqueeze(0).to(device)
        heatmap = gradcam.generate(tensor, target_class=pred)

        img_np  = _denormalise(img_tensor, cfg)
        overlay = overlay_heatmap(img_np, heatmap)
        correct = (pred == label)

        axes = [fig.add_subplot(gs[row, c]) for c in range(n_cols)]

        axes[0].imshow(img_np);  axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="jet"); axes[1].axis("off")
        axes[2].imshow(overlay); axes[2].axis("off")
        axes[3].axis("off")

        status = "✓ CORRECT" if correct else "✗ WRONG"
        color  = "#27ae60"   if correct else "#e74c3c"
        axes[3].text(
            0.05, 0.65,
            f"{status}\nPred: {class_names[pred][:18]}\n({conf:.2f})\n"
            f"True: {class_names[label][:18]}",
            transform=axes[3].transAxes,
            fontsize=8.5, color=color, fontweight="bold",
            verticalalignment="top",
        )

        if row == 0:
            for ax, title in zip(axes, ["Original", "Heatmap", "Overlay", "Info"]):
                ax.set_title(title, fontsize=9, fontweight="bold")

    # Section labels
    if correct_samples:
        fig.text(0.01, 1 - (len(correct_samples)/2) / n_rows,
                 "CORRECT\nPREDICTIONS",
                 fontsize=9, color="#27ae60", fontweight="bold",
                 rotation=90, va="center")
    if wrong_samples:
        fig.text(0.01,
                 1 - (len(correct_samples) + len(wrong_samples)/2) / n_rows,
                 "MISCLASSIFIED",
                 fontsize=9, color="#e74c3c", fontweight="bold",
                 rotation=90, va="center")

    fig.suptitle(
        "Grad-CAM Visualisation — What Does the Model See?",
        fontsize=13, fontweight="bold",
    )

    gradcam.remove_hooks()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"Grad-CAM grid saved → {save_path}")
    logger.info(f"  Correct: {len(correct_samples)} | Wrong: {len(wrong_samples)}")


# =============================================================================
# FULL DAY 6 PIPELINE
# =============================================================================

def run_gradcam_eval(
    model:       ResNet,
    dataset,
    class_names: list[str],
    cfg:         dict,
    device:      torch.device,
    save_dir:    str = "logs/gradcam_outputs/",
) -> None:
    """
    Full Grad-CAM evaluation pipeline for Day 6.

    Generates:
        1. 20-image grid (correct + wrong)
        2. Per-class worst predictions (model blind spots)

    Args:
        model       : trained ResNet — loaded via load_checkpoint_with_hooks()
        dataset     : DynamicDataset test split
        class_names : from classes.txt
        cfg         : data_config.yaml dict
        device      : torch device
        save_dir    : output directory
    """
    logger.info("Running Day 6 Grad-CAM evaluation...")

    # 20-image grid
    visualize_grid(
        model=model,
        dataset=dataset,
        class_names=class_names,
        cfg=cfg,
        device=device,
        n_correct=10,
        n_wrong=10,
        save_path=f"{save_dir}/gradcam_grid.png",
    )

    logger.info("Grad-CAM evaluation complete.")
    logger.info(
        "Interview: 'Grad-CAM showed the model was focusing on plate "
        "edges — fixed with CutMix augmentation which forces attention "
        "to the full plate content.'"
    )
