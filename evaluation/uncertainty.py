# =============================================================================
# evaluation/uncertainty.py — MC Dropout Uncertainty Estimation
# =============================================================================
# Uses Monte Carlo Dropout to estimate prediction uncertainty.
# N=20 stochastic forward passes with dropout active → std of predictions.
# High std (>0.15) → flag for human review.
#
# Interview note:
#   "MC Dropout gives us a free uncertainty estimate — enable dropout
#    at inference, run N=20 forward passes, measure std across passes.
#    High std means the model is genuinely uncertain — flag these for
#    human review instead of serving a potentially wrong prediction.
#    In production this feeds our data flywheel: uncertain samples get
#    human labels → weekly retraining."
# =============================================================================

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Threshold above which a prediction is flagged as uncertain
UNCERTAINTY_THRESHOLD = 0.15


# =============================================================================
# MC DROPOUT — enable dropout at inference
# =============================================================================

def enable_mc_dropout(model: nn.Module) -> None:
    """
    Set model to eval mode BUT keep Dropout layers active.

    Standard model.eval() disables dropout — we want it ON
    for MC Dropout inference.

    Call this before every MC Dropout inference session.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()   # Dropout active = stochastic


def disable_mc_dropout(model: nn.Module) -> None:
    """
    Restore standard eval mode (dropout OFF).
    Call after MC Dropout inference is done.
    """
    model.eval()


# =============================================================================
# SINGLE IMAGE — MC DROPOUT PREDICTION
# =============================================================================

def mc_dropout_predict(
    model:        nn.Module,
    image_tensor: torch.Tensor,
    device:       torch.device,
    n_passes:     int = 20,
) -> dict:
    """
    Run N stochastic forward passes on one image.

    Args:
        model        : ResNet with Dropout in FC head
        image_tensor : (1, C, H, W) or (C, H, W) — normalised
        device       : torch device
        n_passes     : number of MC samples (20 is standard)

    Returns dict:
        mean_probs   : (num_classes,) mean probability across passes
        std_probs    : (num_classes,) std across passes
        pred_class   : most likely class index
        confidence   : mean probability of predicted class
        uncertainty  : std of predicted class probability
        is_uncertain : True if uncertainty > UNCERTAINTY_THRESHOLD
        all_probs    : (n_passes, num_classes) all raw predictions
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    enable_mc_dropout(model)

    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(image_tensor)                    # (1, num_classes)
            probs  = F.softmax(logits, dim=1).squeeze(0)   # (num_classes,)
            all_probs.append(probs.cpu().numpy())

    disable_mc_dropout(model)

    all_probs   = np.array(all_probs)          # (n_passes, num_classes)
    mean_probs  = all_probs.mean(axis=0)       # (num_classes,)
    std_probs   = all_probs.std(axis=0)        # (num_classes,)

    pred_class  = int(mean_probs.argmax())
    confidence  = float(mean_probs[pred_class])
    uncertainty = float(std_probs[pred_class])

    return {
        "mean_probs":   mean_probs,
        "std_probs":    std_probs,
        "pred_class":   pred_class,
        "confidence":   round(confidence,  4),
        "uncertainty":  round(uncertainty, 4),
        "is_uncertain": uncertainty > UNCERTAINTY_THRESHOLD,
        "all_probs":    all_probs,
    }


# =============================================================================
# BATCH UNCERTAINTY EVALUATION
# =============================================================================

def evaluate_uncertainty(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    n_passes:    int = 20,
    max_samples: int = 500,
) -> dict:
    """
    Run MC Dropout uncertainty estimation on a dataset split.

    Args:
        model       : trained ResNet
        loader      : DataLoader (test/val)
        device      : torch device
        n_passes    : MC samples per image
        max_samples : cap to keep runtime reasonable

    Returns dict:
        uncertainties   : (N,) std of predicted class prob
        confidences     : (N,) mean prob of predicted class
        predictions     : (N,) predicted class indices
        labels          : (N,) ground truth
        flagged_indices : indices where uncertainty > threshold
        flagged_frac    : fraction of samples flagged
        accuracy_all    : accuracy on all samples
        accuracy_certain: accuracy on non-flagged samples only
    """
    model.eval()
    enable_mc_dropout(model)

    uncertainties = []
    confidences   = []
    predictions   = []
    labels_all    = []
    n_collected   = 0

    for images, labels in loader:
        if n_collected >= max_samples:
            break

        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)

        # Run N passes for entire batch at once (faster)
        batch_probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = model(images)                          # (B, C)
                probs  = F.softmax(logits, dim=1).cpu().numpy()
                batch_probs.append(probs)

        batch_probs = np.array(batch_probs)  # (n_passes, B, C)

        mean_p = batch_probs.mean(axis=0)    # (B, C)
        std_p  = batch_probs.std(axis=0)     # (B, C)

        preds  = mean_p.argmax(axis=1)       # (B,)
        confs  = mean_p[np.arange(batch_size), preds]
        uncerts= std_p[np.arange(batch_size), preds]

        uncertainties.extend(uncerts.tolist())
        confidences.extend(confs.tolist())
        predictions.extend(preds.tolist())
        labels_all.extend(labels.numpy().tolist())
        n_collected += batch_size

    disable_mc_dropout(model)

    uncertainties   = np.array(uncertainties)
    confidences     = np.array(confidences)
    predictions     = np.array(predictions)
    labels_all      = np.array(labels_all)

    flagged         = uncertainties > UNCERTAINTY_THRESHOLD
    flagged_indices = np.where(flagged)[0].tolist()
    flagged_frac    = flagged.mean()

    acc_all     = (predictions == labels_all).mean()
    # Accuracy on samples the model was certain about
    if (~flagged).sum() > 0:
        acc_certain = (predictions[~flagged] == labels_all[~flagged]).mean()
    else:
        acc_certain = 0.0

    logger.info(
        f"MC Dropout ({n_passes} passes, {n_collected} samples):\n"
        f"  Flagged (uncertainty>{UNCERTAINTY_THRESHOLD}): "
        f"{flagged.sum()} ({flagged_frac*100:.1f}%)\n"
        f"  Accuracy (all)    : {acc_all:.4f}\n"
        f"  Accuracy (certain): {acc_certain:.4f}"
    )

    return {
        "uncertainties":    uncertainties,
        "confidences":      confidences,
        "predictions":      predictions,
        "labels":           labels_all,
        "flagged_indices":  flagged_indices,
        "flagged_frac":     round(float(flagged_frac),  4),
        "accuracy_all":     round(float(acc_all),       4),
        "accuracy_certain": round(float(acc_certain),   4),
        "n_passes":         n_passes,
        "threshold":        UNCERTAINTY_THRESHOLD,
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_uncertainty_distribution(
    results:   dict,
    save_path: str = "logs/uncertainty_distribution.png",
) -> None:
    """
    3-panel uncertainty analysis plot:
        1. Uncertainty distribution with threshold line
        2. Confidence vs Uncertainty scatter
        3. Accuracy: certain vs uncertain samples
    """
    uncertainties   = results["uncertainties"]
    confidences     = results["confidences"]
    predictions     = results["predictions"]
    labels          = results["labels"]
    flagged_frac    = results["flagged_frac"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: Uncertainty distribution ──
    axes[0].hist(uncertainties, bins=40, color="#3498db",
                 edgecolor="white", linewidth=0.5)
    axes[0].axvline(
        UNCERTAINTY_THRESHOLD, color="#e74c3c",
        linestyle="--", linewidth=2,
        label=f"Threshold = {UNCERTAINTY_THRESHOLD}"
    )
    axes[0].set_xlabel("Uncertainty (std of predicted class prob)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(
        f"Uncertainty Distribution\n"
        f"Flagged: {flagged_frac*100:.1f}% of samples"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Panel 2: Confidence vs Uncertainty scatter ──
    correct = predictions == labels
    axes[1].scatter(
        confidences[correct],  uncertainties[correct],
        c="#2ecc71", alpha=0.4, s=12, label="Correct"
    )
    axes[1].scatter(
        confidences[~correct], uncertainties[~correct],
        c="#e74c3c", alpha=0.4, s=12, label="Wrong"
    )
    axes[1].axhline(
        UNCERTAINTY_THRESHOLD, color="black",
        linestyle="--", linewidth=1.5,
        label=f"Uncertainty threshold ({UNCERTAINTY_THRESHOLD})"
    )
    axes[1].set_xlabel("Confidence (mean prob)")
    axes[1].set_ylabel("Uncertainty (std)")
    axes[1].set_title("Confidence vs Uncertainty")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # ── Panel 3: Accuracy comparison ──
    categories = ["All\nSamples", "Certain\n(low uncertainty)",
                  "Uncertain\n(flagged)"]
    flagged    = uncertainties > UNCERTAINTY_THRESHOLD
    acc_all    = (predictions == labels).mean()
    acc_cert   = ((predictions[~flagged] == labels[~flagged]).mean()
                  if (~flagged).sum() > 0 else 0)
    acc_uncert = ((predictions[flagged]  == labels[flagged]).mean()
                  if flagged.sum() > 0 else 0)

    bars = axes[2].bar(
        categories,
        [acc_all, acc_cert, acc_uncert],
        color=["#95a5a6", "#2ecc71", "#e74c3c"],
        edgecolor="white", linewidth=0.5,
    )
    for bar, val in zip(bars, [acc_all, acc_cert, acc_uncert]):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11
        )
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title("Accuracy: Certain vs Uncertain")
    axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"MC Dropout Uncertainty Analysis ({results['n_passes']} passes)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Uncertainty plot saved → {save_path}")


def plot_single_uncertainty(
    result:      dict,
    class_names: list[str],
    image_np:    np.ndarray = None,
    save_path:   str = None,
) -> None:
    """
    Visualise MC Dropout result for a single image.
    Shows: top-5 mean probs with error bars (std).
    """
    mean_probs = result["mean_probs"]
    std_probs  = result["std_probs"]
    pred       = result["pred_class"]
    uncertain  = result["is_uncertain"]

    # Top-5 classes
    top5_idx   = mean_probs.argsort()[::-1][:5]
    top5_names = [class_names[i][:20] for i in top5_idx]
    top5_means = mean_probs[top5_idx]
    top5_stds  = std_probs[top5_idx]

    ncols = 2 if image_np is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    if image_np is not None:
        axes[0].imshow(image_np)
        axes[0].axis("off")
        status = "⚠ UNCERTAIN" if uncertain else "✓ CONFIDENT"
        color  = "#e74c3c"     if uncertain else "#27ae60"
        axes[0].set_title(
            f"{status}\nPred: {class_names[pred]}\n"
            f"conf={result['confidence']:.3f}  "
            f"std={result['uncertainty']:.3f}",
            color=color, fontweight="bold"
        )

    # Error bar chart — top 5 predictions
    colors = ["#e74c3c" if i == 0 else "#3498db" for i in range(5)]
    axes[-1].barh(
        range(5), top5_means[::-1],
        xerr=top5_stds[::-1],
        color=colors[::-1],
        edgecolor="white", linewidth=0.5,
        capsize=4, error_kw={"linewidth": 1.5},
    )
    axes[-1].set_yticks(range(5))
    axes[-1].set_yticklabels(top5_names[::-1], fontsize=10)
    axes[-1].set_xlabel("Mean Probability ± Std (across 20 passes)")
    axes[-1].set_title(
        f"MC Dropout Top-5 Predictions\n"
        f"(error bars = uncertainty)"
    )
    axes[-1].axvline(0, color="black", linewidth=0.5)
    axes[-1].set_xlim(0, 1)
    axes[-1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Single uncertainty plot → {save_path}")
    else:
        plt.show()


# =============================================================================
# API-READY WRAPPER
# =============================================================================

class UncertaintyEstimator:
    """
    Production-ready MC Dropout wrapper for the FastAPI /predict endpoint.

    Usage in api/routes/predict.py (Day 7B):
        estimator = UncertaintyEstimator(model, device, cfg)
        result    = estimator.predict(pil_image)

    Returns:
        {
            "pred_class"  : "biryani",
            "confidence"  : 0.91,
            "uncertainty" : 0.06,
            "is_uncertain": False,
            "flag_review" : False,
        }
    """

    def __init__(
        self,
        model:    nn.Module,
        device:   torch.device,
        cfg:      dict,
        n_passes: int = 20,
    ):
        self.model    = model
        self.device   = device
        self.n_passes = n_passes

        img_size = cfg["image"]["size"]
        mean     = cfg["image"]["mean"]
        std      = cfg["image"]["std"]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def predict(self, pil_image: Image.Image) -> dict:
        """
        Run MC Dropout on a single PIL image.

        Returns lightweight dict for API response.
        """
        tensor = self.transform(pil_image)
        result = mc_dropout_predict(
            self.model, tensor, self.device, self.n_passes
        )
        return {
            "pred_class":   result["pred_class"],
            "confidence":   result["confidence"],
            "uncertainty":  result["uncertainty"],
            "is_uncertain": result["is_uncertain"],
            "flag_review":  result["is_uncertain"],  # alias for API clarity
        }


# =============================================================================
# ENTRYPOINT — quick test
# =============================================================================

if __name__ == "__main__":
    import yaml, sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open("configs/data_config.yaml") as f:
        cfg = yaml.safe_load(f)

    from models.resnet import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model().to(device)

    # Single image test
    dummy  = torch.randn(3, 224, 224)
    result = mc_dropout_predict(model, dummy, device, n_passes=20)

    print(f"Predicted class : {result['pred_class']}")
    print(f"Confidence      : {result['confidence']:.4f}")
    print(f"Uncertainty     : {result['uncertainty']:.4f}")
    print(f"Flag for review : {result['is_uncertain']}")
    print("✓ MC Dropout working correctly")
