# =============================================================================
# datasets/augmentation.py — Advanced Augmentation
# =============================================================================
# Contains:
#   Mixup           — soft label blending between two images
#   CutMix          — random patch replacement with proportional labels
#   TrivialAugment  — zero-hyperparameter augmentation policy
#   AugmentationManager — single class that plugs into train_one_epoch()
#   TTA             — Test-Time Augmentation (used at inference)
#
# How it plugs into train.py (Day 3):
#   manager = AugmentationManager(data_config)
#   # Inside train_one_epoch(), after loading batch:
#   images, labels, soft_labels = manager.apply(images, labels)
#   loss = criterion(logits, soft_labels)   # soft labels for Mixup/CutMix
#
# Interview notes:
#   Mixup   : "Soft label smoothing between two images — reduces
#              overconfident predictions and improves calibration."
#   CutMix  : "Replaces a random patch with another image's patch.
#              Forces model to attend to the FULL plate, not just
#              one corner. Proportional label mixing."
#   TrivialAug: "Zero hyperparameter policy — randomly picks one
#               transform and one magnitude each batch. No tuning needed."
#   TTA     : "At inference: predict on original + flipped + crops,
#              average probabilities. Zero training cost, +0.8% accuracy."
# =============================================================================

import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import TrivialAugmentWide
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# MIXUP
# =============================================================================

class Mixup:
    """
    Mixup augmentation (Zhang et al., 2018).

    Blends two images and their labels:
        image = λ * img_A + (1-λ) * img_B
        label = λ * label_A + (1-λ) * label_B   (soft one-hot)

    λ ~ Beta(alpha, alpha). alpha=0.4 gives mostly pure images with
    occasional strong blends — sweet spot for food classification.

    Args:
        alpha      : Beta distribution concentration (0.4 recommended)
        num_classes: needed to create one-hot soft labels
        p          : probability of applying Mixup to a given batch
    """

    def __init__(self, alpha: float = 0.4, num_classes: int = 80, p: float = 0.5):
        self.alpha       = alpha
        self.num_classes = num_classes
        self.p           = p

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Mixup to a batch.

        Args:
            images : (B, C, H, W) float tensor
            labels : (B,) int64 hard labels

        Returns:
            mixed_images : (B, C, H, W)
            soft_labels  : (B, num_classes) float — use with CrossEntropy
        """
        if np.random.rand() > self.p:
            # No mixup — return one-hot of original labels
            return images, self._to_onehot(labels)

        B = images.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)   # ensure lam >= 0.5 → primary image dominates

        # Shuffled indices for second image
        idx = torch.randperm(B, device=images.device)

        # Blend images
        mixed = lam * images + (1 - lam) * images[idx]

        # Soft labels: weighted sum of one-hot vectors
        labels_a = self._to_onehot(labels)
        labels_b = self._to_onehot(labels[idx])
        soft     = lam * labels_a + (1 - lam) * labels_b

        return mixed, soft

    def _to_onehot(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert (B,) int labels to (B, num_classes) float one-hot."""
        return F.one_hot(labels, self.num_classes).float()


# =============================================================================
# CUTMIX
# =============================================================================

class CutMix:
    """
    CutMix augmentation (Yun et al., 2019).

    Replaces a random rectangular patch in image_A with the same
    patch from image_B. Labels mixed proportionally to patch area.

    Why better than Mixup for food:
        Food images have strong spatial structure — biryani's
        characteristic texture is localised. CutMix forces the model
        to learn from partial views, improving occlusion robustness.

    Args:
        alpha      : Beta distribution for patch size (1.0 recommended)
        num_classes: for soft label creation
        p          : probability of applying CutMix to a given batch
    """

    def __init__(self, alpha: float = 1.0, num_classes: int = 80, p: float = 0.5):
        self.alpha       = alpha
        self.num_classes = num_classes
        self.p           = p

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply CutMix to a batch.

        Returns:
            mixed_images : (B, C, H, W)
            soft_labels  : (B, num_classes) float
        """
        if np.random.rand() > self.p:
            return images, self._to_onehot(labels)

        B, C, H, W = images.shape
        lam = np.random.beta(self.alpha, self.alpha)

        # Compute patch box
        cut_ratio = math.sqrt(1 - lam)
        cut_h     = int(H * cut_ratio)
        cut_w     = int(W * cut_ratio)

        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # Clamp to image bounds
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)

        # Actual lambda after clamping
        lam_actual = 1 - ((x2 - x1) * (y2 - y1)) / (H * W)

        # Shuffled pair
        idx    = torch.randperm(B, device=images.device)
        mixed  = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

        # Soft labels proportional to pixel area
        labels_a = self._to_onehot(labels)
        labels_b = self._to_onehot(labels[idx])
        soft     = lam_actual * labels_a + (1 - lam_actual) * labels_b

        return mixed, soft

    def _to_onehot(self, labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(labels, self.num_classes).float()


# =============================================================================
# AUGMENTATION MANAGER
# =============================================================================

class AugmentationManager:
    """
    Manages all batch-level augmentations for the training loop.

    Policy per batch (randomly picks ONE):
        1. No augmentation    (prob = 1 - p_mixup - p_cutmix)
        2. Mixup only         (prob = p_mixup)
        3. CutMix only        (prob = p_cutmix)

    TrivialAugment is image-level (applied in DataLoader transforms),
    NOT here — see get_train_transforms_with_trivial().

    Usage in train_one_epoch():
        manager = AugmentationManager.from_config(data_cfg)
        ...
        for images, labels in loader:
            images, soft_labels = manager.apply(images, labels)
            with autocast():
                logits = model(images)
                loss   = criterion(logits, soft_labels)
    """

    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        p_mixup:     float = 0.3,
        p_cutmix:    float = 0.3,
    ):
        self.num_classes = num_classes
        self.p_mixup     = p_mixup
        self.p_cutmix    = p_cutmix

        self.mixup  = Mixup(alpha=mixup_alpha,  num_classes=num_classes, p=1.0)
        self.cutmix = CutMix(alpha=cutmix_alpha, num_classes=num_classes, p=1.0)

        # Track usage for logging
        self._counts = {"none": 0, "mixup": 0, "cutmix": 0}

    @classmethod
    def from_config(cls, data_cfg: dict) -> "AugmentationManager":
        """
        Build AugmentationManager from data_config.yaml.
        Reads num_classes automatically — dataset-agnostic.
        """
        aug_cfg = data_cfg.get("augmentation", {}).get("advanced", {})
        return cls(
            num_classes  = data_cfg["dataset"]["num_classes"],
            mixup_alpha  = aug_cfg.get("mixup_alpha",  0.4),
            cutmix_alpha = aug_cfg.get("cutmix_alpha", 1.0),
            p_mixup      = aug_cfg.get("p_mixup",      0.3),
            p_cutmix     = aug_cfg.get("p_cutmix",     0.3),
        )

    def apply(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply one augmentation policy to the batch.

        Args:
            images : (B, C, H, W) — already on device
            labels : (B,) int64   — already on device

        Returns:
            images     : (B, C, H, W) — augmented
            soft_labels: (B, num_classes) float — for CrossEntropyLoss
        """
        roll = np.random.rand()

        if roll < self.p_cutmix:
            # CutMix takes priority (better for spatial food features)
            self._counts["cutmix"] += 1
            return self.cutmix(images, labels)

        elif roll < self.p_cutmix + self.p_mixup:
            self._counts["mixup"] += 1
            return self.mixup(images, labels)

        else:
            # No aug — still return soft labels (one-hot = hard label)
            self._counts["none"] += 1
            soft = F.one_hot(labels, self.num_classes).float()
            return images, soft

    def log_usage(self, epoch: int) -> None:
        """Log augmentation usage counts for the epoch."""
        total = sum(self._counts.values())
        if total == 0:
            return
        parts = " | ".join(
            f"{k}={v} ({100*v/total:.0f}%)"
            for k, v in self._counts.items()
        )
        logger.info(f"Epoch {epoch} — Aug usage: {parts}")
        self._counts = {"none": 0, "mixup": 0, "cutmix": 0}


# =============================================================================
# TRIVIALAUGMENT TRANSFORM — image-level, used in DataLoader
# =============================================================================

def get_train_transforms_with_trivial(cfg: dict) -> transforms.Compose:
    """
    Training transforms with TrivialAugmentWide added.

    Replaces the base transforms from dataset_loader.get_transforms()
    for the training split when TrivialAugment is enabled.

    TrivialAugmentWide (Müller & Hutter, 2021):
        - Randomly samples ONE transform from a fixed set
        - Randomly samples ONE magnitude from a wide range
        - Zero hyperparameters to tune — works out of the box
        - Outperforms RandAugment on most benchmarks

    Usage — replace get_transforms(cfg, 'train') with this
    when running augmentation ablation experiments:
        dataset = DynamicDataset(root, cfg, split='train',
                                 transform_override=get_train_transforms_with_trivial(cfg))

    Args:
        cfg : full data_config.yaml dict

    Returns:
        transforms.Compose with TrivialAugmentWide inserted
    """
    img_size = cfg["image"]["size"]
    mean     = cfg["image"]["mean"]
    std      = cfg["image"]["std"]

    return transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        # TrivialAugmentWide — zero hyperparameter policy
        TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# =============================================================================
# TEST-TIME AUGMENTATION (TTA)
# =============================================================================

class TTA:
    """
    Test-Time Augmentation — average predictions over multiple views.

    Views used (5 total):
        1. Original image (centre crop)
        2. Horizontal flip
        3. Slight zoom-in crop (90%)
        4. Top-left crop
        5. Bottom-right crop

    Zero training cost — only at inference.
    Consistent +0.8% accuracy on food classification benchmarks.

    Usage in /predict endpoint:
        tta       = TTA(model, device)
        probs     = tta.predict(image_tensor)   # (num_classes,)
        class_idx = probs.argmax().item()
    """

    def __init__(
        self,
        model:  nn.Module,
        device: torch.device,
        cfg:    Optional[dict] = None,
    ):
        self.model  = model
        self.device = device
        img_size    = cfg["image"]["size"] if cfg else 224
        mean        = cfg["image"]["mean"] if cfg else [0.485, 0.456, 0.406]
        std         = cfg["image"]["std"]  if cfg else [0.229, 0.224, 0.225]

        normalize = transforms.Normalize(mean=mean, std=std)

        # Define the 5 deterministic views
        crop90 = int(img_size * 0.9)
        self.views = [
            # 1. Centre crop
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(), normalize,
            ]),
            # 2. Horizontal flip
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(), normalize,
            ]),
            # 3. Zoom-in (90% crop, resize back)
            transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.CenterCrop(crop90),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(), normalize,
            ]),
            # 4. Top-left crop
            transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.Lambda(lambda img: transforms.functional.crop(
                    img, 0, 0, img_size, img_size)),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(), normalize,
            ]),
            # 5. Bottom-right crop
            transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.Lambda(lambda img: transforms.functional.crop(
                    img, 32, 32, img_size, img_size)),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(), normalize,
            ]),
        ]

    @torch.no_grad()
    def predict(self, pil_image) -> torch.Tensor:
        """
        Run all TTA views and return averaged probability vector.

        Args:
            pil_image : PIL.Image (RGB) — raw image before any transform

        Returns:
            probs : (num_classes,) float tensor — averaged softmax probs
        """
        self.model.eval()
        all_probs = []

        for view_transform in self.views:
            tensor = view_transform(pil_image).unsqueeze(0).to(self.device)
            logits = self.model(tensor)                    # (1, num_classes)
            probs  = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)
            all_probs.append(probs)

        # Average across all views
        return torch.stack(all_probs).mean(dim=0)          # (num_classes,)

    @torch.no_grad()
    def predict_batch(self, pil_images: list) -> torch.Tensor:
        """
        TTA for a list of PIL images.

        Returns:
            probs : (N, num_classes) float tensor
        """
        return torch.stack([self.predict(img) for img in pil_images])


# =============================================================================
# SOFT CROSS ENTROPY LOSS — required for Mixup / CutMix soft labels
# =============================================================================

class SoftCrossEntropyLoss(nn.Module):
    """
    CrossEntropy that accepts soft (float) target distributions.

    PyTorch's built-in CrossEntropyLoss with label_smoothing only
    supports hard integer labels. This version works with the soft
    labels produced by Mixup and CutMix.

    Also supports label smoothing on top of soft labels.

    Usage:
        criterion = SoftCrossEntropyLoss(smoothing=0.1)
        loss = criterion(logits, soft_labels)   # soft_labels: (B, C) float
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits      : (B, C) raw model output
            soft_targets: (B, C) float — from Mixup/CutMix or one-hot

        Returns:
            scalar loss
        """
        num_classes = logits.size(1)
        log_probs   = F.log_softmax(logits, dim=1)

        # Apply label smoothing on top of soft targets
        if self.smoothing > 0:
            smooth_val   = self.smoothing / num_classes
            soft_targets = (1 - self.smoothing) * soft_targets + smooth_val

        # KL-divergence style: -sum(target * log_prob)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss


# =============================================================================
# AUGMENTATION EXPERIMENT RUNNER — for ablation notebook
# =============================================================================

def get_augmentation_config(strategy: str, num_classes: int) -> AugmentationManager:
    """
    Return AugmentationManager for a named ablation strategy.
    Used in notebooks/ablation.ipynb.

    Strategies:
        'none'          : no batch augmentation
        'mixup_only'    : only Mixup (p=0.5)
        'cutmix_only'   : only CutMix (p=0.5)
        'combined'      : Mixup + CutMix (p=0.3 each)

    Args:
        strategy    : one of the strings above
        num_classes : from data_config.yaml

    Returns:
        AugmentationManager
    """
    configs = {
        "none": dict(
            p_mixup=0.0, p_cutmix=0.0,
        ),
        "mixup_only": dict(
            mixup_alpha=0.4, p_mixup=0.5, p_cutmix=0.0,
        ),
        "cutmix_only": dict(
            cutmix_alpha=1.0, p_mixup=0.0, p_cutmix=0.5,
        ),
        "combined": dict(
            mixup_alpha=0.4, cutmix_alpha=1.0,
            p_mixup=0.3, p_cutmix=0.3,
        ),
    }
    if strategy not in configs:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Choose from {list(configs.keys())}"
        )
    return AugmentationManager(num_classes=num_classes, **configs[strategy])