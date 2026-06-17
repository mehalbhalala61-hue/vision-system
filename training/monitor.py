# =============================================================================
# training/monitor.py — Gradient Norm Monitor
# =============================================================================
# Tracks gradient norms per layer every N batches.
# Alerts on exploding (>10) or vanishing (<1e-4) gradients.
#
# Interview note:
#   "I tracked gradient norms per layer during training. This caught an
#    exploding gradient at epoch 3 — caused by AMP + high LR interaction.
#    Documented in debug_log.md."
# =============================================================================

import logging
import torch
import torch.nn as nn
from collections import defaultdict

logger = logging.getLogger(__name__)

EXPLODE_THRESHOLD = 10.0
VANISH_THRESHOLD  = 1e-4


class GradientMonitor:
    """
    Tracks gradient norms per named layer group.

    Usage in train loop:
        monitor = GradientMonitor(model)
        # after loss.backward(), before optimizer.step():
        monitor.record(batch_idx)
        # at epoch end:
        monitor.summarize(epoch)
        monitor.reset()
    """

    def __init__(self, model: nn.Module, log_every_n: int = 50):
        self.model       = model
        self.log_every_n = log_every_n
        self.history: dict[str, list[float]] = defaultdict(list)

    def record(self, batch_idx: int) -> dict[str, float]:
        """
        Compute gradient norm per layer group and store.
        Call after loss.backward(), before optimizer.step().

        Returns dict of {layer_name: grad_norm} for this batch.
        """
        norms: dict[str, float] = {}

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            # Group by top-level module (stem / layer1 / layer2 / ...)
            group = name.split(".")[0]
            norm  = param.grad.data.norm(2).item()
            self.history[group].append(norm)
            norms[group] = norms.get(group, 0) + norm

        # Alert on every batch — these are serious
        for group, norm in norms.items():
            if norm > EXPLODE_THRESHOLD:
                logger.warning(
                    f"  ⚠ EXPLODING gradient | {group} | norm={norm:.2f} "
                    f"(batch {batch_idx}) — check LR or AMP settings"
                )
            elif norm < VANISH_THRESHOLD and norm > 0:
                logger.warning(
                    f"  ⚠ VANISHING gradient | {group} | norm={norm:.2e} "
                    f"(batch {batch_idx})"
                )

        if batch_idx % self.log_every_n == 0:
            parts = " | ".join(f"{g}={v:.3f}" for g, v in sorted(norms.items()))
            logger.debug(f"  GradNorms [{batch_idx}]: {parts}")

        return norms

    def summarize(self, epoch: int) -> dict[str, float]:
        """
        Log mean gradient norm per layer group for the epoch.
        Returns dict of {layer_group: mean_norm}.
        """
        if not self.history:
            return {}

        summary: dict[str, float] = {}
        for group, values in sorted(self.history.items()):
            mean_norm = sum(values) / len(values)
            summary[group] = round(mean_norm, 6)

        logger.info(f"Epoch {epoch} — Gradient norm summary (mean per group):")
        for group, norm in summary.items():
            status = ""
            if norm > EXPLODE_THRESHOLD:
                status = " ← EXPLODING ⚠"
            elif norm < VANISH_THRESHOLD:
                status = " ← VANISHING ⚠"
            logger.info(f"  {group:<12}: {norm:.6f}{status}")

        return summary

    def reset(self) -> None:
        """Clear history at start of each epoch."""
        self.history.clear()
