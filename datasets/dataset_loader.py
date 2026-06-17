# =============================================================================
# datasets/dataset_loader.py — DynamicDataset + DataLoader Factory
# =============================================================================
# Fully config-driven — reads everything from configs/data_config.yaml.
# Change dataset name in YAML → entire pipeline updates. Zero code changes.
#
# Interview note:
#   "My DynamicDataset reads any folder-structured dataset from a single YAML
#    config. Switch from Indian Food 101 to Stanford Dogs: change 4 lines in
#    data_config.yaml — no code changes anywhere in the pipeline."
#
# Folder structure expected:
#   data/raw/train/class_name/image.jpg
#   data/raw/val/class_name/image.jpg
#   data/raw/test/class_name/image.jpg
# =============================================================================

import os
import logging
import hashlib
import numpy as np
import yaml
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG LOADER
# =============================================================================

def load_config(config_path: str = "configs/data_config.yaml") -> dict:
    """Load and return the master data config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# =============================================================================
# TRANSFORMS
# =============================================================================

def get_transforms(cfg: dict, split: str) -> transforms.Compose:
    """
    Build torchvision transform pipeline from config.

    Args:
        cfg   : full data_config.yaml dict
        split : 'train' | 'val' | 'test'

    Returns:
        transforms.Compose ready to plug into DynamicDataset
    """
    img_size = cfg["image"]["size"]
    mean     = cfg["image"]["mean"]
    std      = cfg["image"]["std"]
    aug      = cfg["augmentation"]

    if split == "train":
        t = aug["train"]
        transform_list = [
            transforms.Resize((img_size + 16, img_size + 16)),  # slightly larger for crop
        ]

        if t["random_crop"]["enabled"]:
            transform_list.append(
                transforms.RandomCrop(img_size, padding=t["random_crop"]["padding"])
            )
        else:
            transform_list.append(transforms.CenterCrop(img_size))

        if t["horizontal_flip"]["enabled"]:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=t["horizontal_flip"]["p"])
            )

        if t["color_jitter"]["enabled"]:
            cj = t["color_jitter"]
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj["brightness"],
                    contrast=cj["contrast"],
                    saturation=cj["saturation"],
                    hue=cj["hue"],
                )
            )

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    else:
        # Val / Test — deterministic
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(transform_list)


# =============================================================================
# DYNAMIC DATASET
# =============================================================================

class DynamicDataset(Dataset):
    """
    Dataset-agnostic image loader. Works with ANY folder-structured dataset.

    Args:
        root_dir   : path to split folder (e.g. data/raw/train)
        cfg        : full data_config.yaml dict
        split      : 'train' | 'val' | 'test'  — controls augmentation
        cache_dir  : if set, caches preprocessed tensors to disk for speed
        indices    : optional subset of indices (for stratified split)

    Usage:
        cfg   = load_config("configs/data_config.yaml")
        train = DynamicDataset("data/raw/train", cfg, split="train")
        val   = DynamicDataset("data/raw/val",   cfg, split="val")
    """

    def __init__(
        self,
        root_dir:  str,
        cfg:       dict,
        split:     str = "train",
        cache_dir: Optional[str] = None,
        indices:   Optional[list] = None,
    ):
        self.root_dir  = Path(root_dir)
        self.cfg       = cfg
        self.split     = split
        self.transform = get_transforms(cfg, split)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # ------------------------------------------------------------------
        # Load class names from classes.txt (order = label index)
        # ------------------------------------------------------------------
        classes_file = Path(cfg["dataset"]["class_names_file"])
        if not classes_file.exists():
            raise FileNotFoundError(
                f"classes.txt not found at {classes_file}.\n"
                "Run: ls data/raw/train | sort > data/classes.txt"
            )

        self.classes     = [c.strip() for c in classes_file.read_text().splitlines() if c.strip()]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.num_classes  = len(self.classes)

        # ------------------------------------------------------------------
        # Scan all images
        # ------------------------------------------------------------------
        self.samples = self._scan_directory()

        if not self.samples:
            raise RuntimeError(
                f"No images found in {self.root_dir}.\n"
                "Expected structure: root_dir/class_name/image.jpg"
            )

        # Apply optional subset indices (for stratified split)
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        logger.info(
            f"DynamicDataset [{split}] | "
            f"dataset={cfg['dataset']['name']} | "
            f"classes={self.num_classes} | "
            f"samples={len(self.samples)}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_directory(self) -> list[tuple[Path, int]]:
        """
        Walk root_dir, collect (image_path, label_idx) tuples.
        Skips corrupt / non-image files with a warning.
        """
        VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        samples = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            if class_name not in self.class_to_idx:
                logger.warning(
                    f"Folder '{class_name}' not in classes.txt — skipping. "
                    f"Update data/classes.txt if this is a valid class."
                )
                continue

            label = self.class_to_idx[class_name]

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                samples.append((img_path, label))

        return samples

    def _cache_path(self, img_path: Path) -> Optional[Path]:
        """Return deterministic cache path for an image tensor."""
        if self.cache_dir is None:
            return None
        key     = hashlib.md5(str(img_path).encode()).hexdigest()
        subdir  = self.cache_dir / self.split
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{key}.pt"

    def _load_image(self, img_path: Path) -> torch.Tensor:
        """Load image → apply transform → return tensor. With cache support."""
        cache_path = self._cache_path(img_path)

        # Cache hit
        if cache_path and cache_path.exists():
            return torch.load(cache_path, weights_only=True)

        # Load + transform
        try:
            img = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            logger.warning(f"Corrupt image skipped: {img_path} | {e}")
            # Return a black image tensor of correct size
            size = self.cfg["image"]["size"]
            return torch.zeros(3, size, size)

        tensor = self.transform(img)

        # Cache miss — save
        if cache_path:
            torch.save(tensor, cache_path)

        return tensor

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        tensor = self._load_image(img_path)
        return tensor, label

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_class_counts(self) -> dict[str, int]:
        """Returns {class_name: count} — used for EDA + weighted sampling."""
        counts: dict[str, int] = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts

    def get_labels(self) -> list[int]:
        """Return all labels — used for stratified split."""
        return [label for _, label in self.samples]


# =============================================================================
# STRATIFIED SPLIT UTILITY
# =============================================================================

def make_stratified_splits(
    dataset: DynamicDataset,
    cfg: dict,
    save: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """
    Create stratified train/val/test index splits from a full dataset.
    Saves .npy files so splits are reproducible across runs.

    Args:
        dataset : DynamicDataset loaded from full data root
        cfg     : data_config.yaml dict
        save    : whether to save .npy index files

    Returns:
        (train_indices, val_indices, test_indices)

    Usage (when you have a single folder, not pre-split):
        full_ds = DynamicDataset("data/raw/all", cfg, split="train")
        train_idx, val_idx, test_idx = make_stratified_splits(full_ds, cfg)
    """
    split_cfg  = cfg["dataset"]["splits"]
    val_ratio  = split_cfg["val"]
    test_ratio = split_cfg["test"]
    seed       = split_cfg["seed"]
    save_path  = Path(cfg["paths"]["splits"])

    labels  = dataset.get_labels()
    indices = list(range(len(dataset)))

    # First split: train vs (val + test)
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=seed,
    )

    logger.info(
        f"Stratified split | "
        f"train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}"
    )

    if save:
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / "train_indices.npy", np.array(train_idx))
        np.save(save_path / "val_indices.npy",   np.array(val_idx))
        np.save(save_path / "test_indices.npy",  np.array(test_idx))
        logger.info(f"Split indices saved → {save_path}")

    return train_idx, val_idx, test_idx


# =============================================================================
# DATALOADER FACTORY
# =============================================================================

def get_dataloaders(cfg: dict) -> dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from config.
    This is the single function called by train.py — nothing else needed.

    Args:
        cfg : full data_config.yaml dict

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}

    Usage in train.py:
        cfg         = load_config()
        dataloaders = get_dataloaders(cfg)
        for images, labels in dataloaders["train"]:
            ...
    """
    ds_cfg   = cfg["dataset"]
    dl_cfg   = cfg["dataloader"]
    cache_on = cfg["image"]["cache"]["enabled"]
    cache_dir = cfg["image"]["cache"]["cache_dir"] if cache_on else None

    splits_path = Path(cfg["paths"]["splits"])

    datasets = {}
    for split, dir_key in [("train", "train_dir"), ("val", "val_dir"), ("test", "test_dir")]:
        root = ds_cfg[dir_key]

        # Load saved indices if they exist (reproducible splits)
        idx_file = splits_path / f"{split}_indices.npy"
        indices  = np.load(idx_file).tolist() if idx_file.exists() else None

        datasets[split] = DynamicDataset(
            root_dir  = root,
            cfg       = cfg,
            split     = split,
            cache_dir = cache_dir,
            indices   = indices,
        )

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size  = dl_cfg["batch_size"],
            shuffle     = (split == "train"),
            num_workers = dl_cfg["num_workers"],
            pin_memory  = dl_cfg["pin_memory"],
            drop_last   = dl_cfg["drop_last"] if split == "train" else False,
            # Faster multiprocessing on Linux/Kaggle
            persistent_workers = (dl_cfg["num_workers"] > 0),
        )
        for split in ["train", "val", "test"]
    }

    # Log dataset sizes
    for split, ds in datasets.items():
        logger.info(f"DataLoader [{split}] | {len(ds)} samples | "
                    f"batches={len(loaders[split])}")

    return loaders


# =============================================================================
# QUICK SANITY CHECK
# =============================================================================

def verify_dataset(cfg: dict) -> None:
    """
    Quick sanity check — run before training to catch issues early.
    Checks: image loading, label range, tensor shape, class count.
    """
    loaders = get_dataloaders(cfg)
    num_classes = cfg["dataset"]["num_classes"]
    img_size    = cfg["image"]["size"]

    for split, loader in loaders.items():
        images, labels = next(iter(loader))
        assert images.shape == (
            cfg["dataloader"]["batch_size"], 3, img_size, img_size
        ), f"[{split}] Unexpected image shape: {images.shape}"
        assert labels.max() < num_classes, \
            f"[{split}] Label {labels.max()} >= num_classes {num_classes}"
        assert labels.min() >= 0, \
            f"[{split}] Negative label found"
        logger.info(
            f"[{split}] ✓ shape={tuple(images.shape)} | "
            f"labels in [0, {num_classes-1}]"
        )

    logger.info("Dataset verification passed.")


# =============================================================================
# ENTRYPOINT — run directly for quick test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config("configs/data_config.yaml")
    verify_dataset(cfg)
