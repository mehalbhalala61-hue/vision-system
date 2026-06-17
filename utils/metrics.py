# =============================================================================
# utils/metrics.py — Classification Metrics
# =============================================================================
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy over a batch."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Top-K accuracy over a batch."""
    _, top_k = logits.topk(k, dim=1)
    correct  = top_k.eq(labels.unsqueeze(1).expand_as(top_k))
    return correct.any(dim=1).float().mean().item()


def compute_metrics(
    all_preds:  np.ndarray,
    all_labels: np.ndarray,
    num_classes: int,
) -> dict:
    """
    Compute macro precision, recall, F1 over the full val/test set.

    Args:
        all_preds  : (N,) predicted class indices
        all_labels : (N,) ground truth class indices
        num_classes: total number of classes

    Returns:
        dict with precision, recall, f1 (macro) and per_class_f1
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds,
        average="macro",
        zero_division=0,
    )
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    return {
        "precision_macro": round(float(precision), 4),
        "recall_macro":    round(float(recall),    4),
        "f1_macro":        round(float(f1),        4),
        "per_class_f1":    per_class_f1.tolist(),
    }
