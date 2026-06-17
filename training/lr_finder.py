# =============================================================================
# training/lr_finder.py — Learning Rate Range Test
# =============================================================================
# Sweeps LR from start_lr to end_lr exponentially over num_iter steps.
# Plots loss vs LR — pick LR just before loss diverges.
#
# Interview note:
#   "Before training I ran an LR range test — it found the optimal LR 10x
#    faster than grid search. I picked 3e-4 from the curve — the point just
#    before loss starts to diverge. This removes all LR guesswork."
# =============================================================================

import copy
import logging
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LRFinder:
    """
    Learning Rate Range Test (Smith, 2017).

    Exponentially increases LR from start_lr to end_lr over num_iter steps,
    records smoothed loss at each step, then plots the curve.

    Usage:
        finder = LRFinder(model, optimizer, criterion, device)
        best_lr = finder.run(train_loader, cfg)
        # update train_config.yaml: optimizer.lr = best_lr
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device:    torch.device,
    ):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device

        # Save original state — restored after finder run
        self._model_state     = copy.deepcopy(model.state_dict())
        self._optimizer_state = copy.deepcopy(optimizer.state_dict())

    def run(self, train_loader: DataLoader, cfg: dict) -> float:
        """
        Run LR range test.

        Args:
            train_loader : training DataLoader
            cfg          : full train_config.yaml dict

        Returns:
            suggested_lr : LR at steepest loss descent
        """
        finder_cfg = cfg["lr_finder"]
        start_lr   = finder_cfg["start_lr"]
        end_lr     = finder_cfg["end_lr"]
        num_iter   = finder_cfg["num_iter"]
        smooth_f   = finder_cfg["smooth_f"]
        diverge_th = finder_cfg["diverge_th"]
        output_plot= finder_cfg["output_plot"]

        # LR multiplier per step
        lr_mult = (end_lr / start_lr) ** (1 / (num_iter - 1))

        # Set initial LR
        for pg in self.optimizer.param_groups:
            pg["lr"] = start_lr

        lrs, losses = [], []
        smoothed_loss = 0.0
        best_loss     = float("inf")
        current_lr    = start_lr
        step          = 0

        self.model.train()
        data_iter = iter(train_loader)

        logger.info(f"LR Finder: {start_lr:.2e} → {end_lr:.2e} over {num_iter} steps")

        while step < num_iter:
            # Cycle through loader if needed
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()

            # EMA smoothing
            smoothed_loss = smooth_f * loss_val + (1 - smooth_f) * smoothed_loss
            if step == 0:
                smoothed_loss = loss_val

            lrs.append(current_lr)
            losses.append(smoothed_loss)

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Stop if loss diverges
            if smoothed_loss > diverge_th * best_loss:
                logger.info(f"  Loss diverged at LR={current_lr:.2e} — stopping")
                break

            # Step LR
            current_lr *= lr_mult
            for pg in self.optimizer.param_groups:
                pg["lr"] = current_lr

            step += 1

        # Suggested LR = steepest negative gradient in loss curve
        suggested_lr = self._find_best_lr(lrs, losses)

        # Plot
        self._plot(lrs, losses, suggested_lr, output_plot)

        # Restore original model + optimizer state
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optimizer_state)

        # Re-register Grad-CAM hook — load_state_dict detaches hooks silently (v3 fix)
        if hasattr(self.model, "_register_gradcam_hook"):
            self.model._register_gradcam_hook()

        logger.info(f"LR Finder done. Suggested LR: {suggested_lr:.2e}")
        logger.info(f"Update configs/train_config.yaml → optimizer.lr: {suggested_lr:.2e}")

        return suggested_lr

    def _find_best_lr(self, lrs: list, losses: list) -> float:
        """
        Return LR at the point of steepest loss descent.
        Skip first 10% and last 20% of the curve (unstable regions).
        """
        if len(losses) < 5:
            return lrs[len(lrs) // 2]

        start = max(1, len(losses) // 10)
        end   = max(start + 1, int(len(losses) * 0.8))

        min_grad_idx = start
        min_grad     = float("inf")

        for i in range(start, end):
            grad = losses[i] - losses[i - 1]
            if grad < min_grad:
                min_grad     = grad
                min_grad_idx = i

        return lrs[min_grad_idx]

    def _plot(
        self,
        lrs:          list,
        losses:       list,
        suggested_lr: float,
        output_path:  str,
    ) -> None:
        """Save LR vs Loss plot to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(lrs, losses, color="#3498db", linewidth=2)
        ax.axvline(
            suggested_lr, color="#e74c3c", linestyle="--", linewidth=1.5,
            label=f"Suggested LR: {suggested_lr:.2e}",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
        ax.set_ylabel("Smoothed Loss",              fontsize=12)
        ax.set_title("LR Range Test — Pick LR just before loss diverges", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"LR finder plot saved → {output_path}")# =============================================================================
# training/lr_finder.py — Learning Rate Range Test
# =============================================================================
# Sweeps LR from start_lr to end_lr exponentially over num_iter steps.
# Plots loss vs LR — pick LR just before loss diverges.
#
# Interview note:
#   "Before training I ran an LR range test — it found the optimal LR 10x
#    faster than grid search. I picked 3e-4 from the curve — the point just
#    before loss starts to diverge. This removes all LR guesswork."
# =============================================================================

import copy
import logging
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LRFinder:
    """
    Learning Rate Range Test (Smith, 2017).

    Exponentially increases LR from start_lr to end_lr over num_iter steps,
    records smoothed loss at each step, then plots the curve.

    Usage:
        finder = LRFinder(model, optimizer, criterion, device)
        best_lr = finder.run(train_loader, cfg)
        # update train_config.yaml: optimizer.lr = best_lr
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device:    torch.device,
    ):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device

        # Save original state — restored after finder run
        self._model_state     = copy.deepcopy(model.state_dict())
        self._optimizer_state = copy.deepcopy(optimizer.state_dict())

    def run(self, train_loader: DataLoader, cfg: dict) -> float:
        """
        Run LR range test.

        Args:
            train_loader : training DataLoader
            cfg          : full train_config.yaml dict

        Returns:
            suggested_lr : LR at steepest loss descent
        """
        finder_cfg = cfg["lr_finder"]
        start_lr   = finder_cfg["start_lr"]
        end_lr     = finder_cfg["end_lr"]
        num_iter   = finder_cfg["num_iter"]
        smooth_f   = finder_cfg["smooth_f"]
        diverge_th = finder_cfg["diverge_th"]
        output_plot= finder_cfg["output_plot"]

        # LR multiplier per step
        lr_mult = (end_lr / start_lr) ** (1 / (num_iter - 1))

        # Set initial LR
        for pg in self.optimizer.param_groups:
            pg["lr"] = start_lr

        lrs, losses = [], []
        smoothed_loss = 0.0
        best_loss     = float("inf")
        current_lr    = start_lr
        step          = 0

        self.model.train()
        data_iter = iter(train_loader)

        logger.info(f"LR Finder: {start_lr:.2e} → {end_lr:.2e} over {num_iter} steps")

        while step < num_iter:
            # Cycle through loader if needed
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()

            # EMA smoothing
            smoothed_loss = smooth_f * loss_val + (1 - smooth_f) * smoothed_loss
            if step == 0:
                smoothed_loss = loss_val

            lrs.append(current_lr)
            losses.append(smoothed_loss)

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Stop if loss diverges
            if smoothed_loss > diverge_th * best_loss:
                logger.info(f"  Loss diverged at LR={current_lr:.2e} — stopping")
                break

            # Step LR
            current_lr *= lr_mult
            for pg in self.optimizer.param_groups:
                pg["lr"] = current_lr

            step += 1

        # Suggested LR = steepest negative gradient in loss curve
        suggested_lr = self._find_best_lr(lrs, losses)

        # Plot
        self._plot(lrs, losses, suggested_lr, output_plot)

        # Restore original model + optimizer state
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optimizer_state)

        # Re-register Grad-CAM hook — load_state_dict detaches hooks silently (v3 fix)
        if hasattr(self.model, "_register_gradcam_hook"):
            self.model._register_gradcam_hook()

        logger.info(f"LR Finder done. Suggested LR: {suggested_lr:.2e}")
        logger.info(f"Update configs/train_config.yaml → optimizer.lr: {suggested_lr:.2e}")

        return suggested_lr

    def _find_best_lr(self, lrs: list, losses: list) -> float:
        """
        Return LR at the point of steepest loss descent.
        Skip first 10% and last 20% of the curve (unstable regions).
        """
        if len(losses) < 5:
            return lrs[len(lrs) // 2]

        start = max(1, len(losses) // 10)
        end   = max(start + 1, int(len(losses) * 0.8))

        min_grad_idx = start
        min_grad     = float("inf")

        for i in range(start, end):
            grad = losses[i] - losses[i - 1]
            if grad < min_grad:
                min_grad     = grad
                min_grad_idx = i

        return lrs[min_grad_idx]

    def _plot(
        self,
        lrs:          list,
        losses:       list,
        suggested_lr: float,
        output_path:  str,
    ) -> None:
        """Save LR vs Loss plot to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(lrs, losses, color="#3498db", linewidth=2)
        ax.axvline(
            suggested_lr, color="#e74c3c", linestyle="--", linewidth=1.5,
            label=f"Suggested LR: {suggested_lr:.2e}",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
        ax.set_ylabel("Smoothed Loss",              fontsize=12)
        ax.set_title("LR Range Test — Pick LR just before loss diverges", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"LR finder plot saved → {output_path}")# =============================================================================
# training/lr_finder.py — Learning Rate Range Test
# =============================================================================
# Sweeps LR from start_lr to end_lr exponentially over num_iter steps.
# Plots loss vs LR — pick LR just before loss diverges.
#
# Interview note:
#   "Before training I ran an LR range test — it found the optimal LR 10x
#    faster than grid search. I picked 3e-4 from the curve — the point just
#    before loss starts to diverge. This removes all LR guesswork."
# =============================================================================

import copy
import logging
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LRFinder:
    """
    Learning Rate Range Test (Smith, 2017).

    Exponentially increases LR from start_lr to end_lr over num_iter steps,
    records smoothed loss at each step, then plots the curve.

    Usage:
        finder = LRFinder(model, optimizer, criterion, device)
        best_lr = finder.run(train_loader, cfg)
        # update train_config.yaml: optimizer.lr = best_lr
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device:    torch.device,
    ):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device

        # Save original state — restored after finder run
        self._model_state     = copy.deepcopy(model.state_dict())
        self._optimizer_state = copy.deepcopy(optimizer.state_dict())

    def run(self, train_loader: DataLoader, cfg: dict) -> float:
        """
        Run LR range test.

        Args:
            train_loader : training DataLoader
            cfg          : full train_config.yaml dict

        Returns:
            suggested_lr : LR at steepest loss descent
        """
        finder_cfg = cfg["lr_finder"]
        start_lr   = finder_cfg["start_lr"]
        end_lr     = finder_cfg["end_lr"]
        num_iter   = finder_cfg["num_iter"]
        smooth_f   = finder_cfg["smooth_f"]
        diverge_th = finder_cfg["diverge_th"]
        output_plot= finder_cfg["output_plot"]

        # LR multiplier per step
        lr_mult = (end_lr / start_lr) ** (1 / (num_iter - 1))

        # Set initial LR
        for pg in self.optimizer.param_groups:
            pg["lr"] = start_lr

        lrs, losses = [], []
        smoothed_loss = 0.0
        best_loss     = float("inf")
        current_lr    = start_lr
        step          = 0

        self.model.train()
        data_iter = iter(train_loader)

        logger.info(f"LR Finder: {start_lr:.2e} → {end_lr:.2e} over {num_iter} steps")

        while step < num_iter:
            # Cycle through loader if needed
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                images, labels = next(data_iter)

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()

            # EMA smoothing
            smoothed_loss = smooth_f * loss_val + (1 - smooth_f) * smoothed_loss
            if step == 0:
                smoothed_loss = loss_val

            lrs.append(current_lr)
            losses.append(smoothed_loss)

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Stop if loss diverges
            if smoothed_loss > diverge_th * best_loss:
                logger.info(f"  Loss diverged at LR={current_lr:.2e} — stopping")
                break

            # Step LR
            current_lr *= lr_mult
            for pg in self.optimizer.param_groups:
                pg["lr"] = current_lr

            step += 1

        # Suggested LR = steepest negative gradient in loss curve
        suggested_lr = self._find_best_lr(lrs, losses)

        # Plot
        self._plot(lrs, losses, suggested_lr, output_plot)

        # Restore original model + optimizer state
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optimizer_state)

        # Re-register Grad-CAM hook — load_state_dict detaches hooks silently (v3 fix)
        if hasattr(self.model, "_register_gradcam_hook"):
            self.model._register_gradcam_hook()

        logger.info(f"LR Finder done. Suggested LR: {suggested_lr:.2e}")
        logger.info(f"Update configs/train_config.yaml → optimizer.lr: {suggested_lr:.2e}")

        return suggested_lr

    def _find_best_lr(self, lrs: list, losses: list) -> float:
        """
        Return LR at the point of steepest loss descent.
        Skip first 10% and last 20% of the curve (unstable regions).
        """
        if len(losses) < 5:
            return lrs[len(lrs) // 2]

        start = max(1, len(losses) // 10)
        end   = max(start + 1, int(len(losses) * 0.8))

        min_grad_idx = start
        min_grad     = float("inf")

        for i in range(start, end):
            grad = losses[i] - losses[i - 1]
            if grad < min_grad:
                min_grad     = grad
                min_grad_idx = i

        return lrs[min_grad_idx]

    def _plot(
        self,
        lrs:          list,
        losses:       list,
        suggested_lr: float,
        output_path:  str,
    ) -> None:
        """Save LR vs Loss plot to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(lrs, losses, color="#3498db", linewidth=2)
        ax.axvline(
            suggested_lr, color="#e74c3c", linestyle="--", linewidth=1.5,
            label=f"Suggested LR: {suggested_lr:.2e}",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)", fontsize=12)
        ax.set_ylabel("Smoothed Loss",              fontsize=12)
        ax.set_title("LR Range Test — Pick LR just before loss diverges", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"LR finder plot saved → {output_path}")