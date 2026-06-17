# =============================================================================
# api/deps.py — FastAPI Dependency Injection
# =============================================================================
# Centralised dependencies shared across all routes.
# Injected via FastAPI's Depends() — session always closed after request.
# =============================================================================

import yaml
import torch
from functools import lru_cache
from sqlalchemy.orm import Session

from db.session import SessionLocal
from models.resnet import build_model
from utils.checkpoint import load_checkpoint_with_hooks
from evaluation.nutrition import NutritionLookup
from evaluation.tta import TTAPredictor
from evaluation.uncertainty import UncertaintyEstimator


# =============================================================================
# DB SESSION
# =============================================================================

def get_db() -> Session:
    """
    FastAPI dependency — one DB session per request.
    Always closed after request, even on exception.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# =============================================================================
# CONFIGS — loaded once at startup
# =============================================================================

@lru_cache(maxsize=1)
def get_data_cfg() -> dict:
    with open("configs/data_config.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_model_cfg() -> dict:
    with open("configs/model_config.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_ai_cfg() -> dict:
    with open("configs/ai_config.yaml") as f:
        return yaml.safe_load(f)


# =============================================================================
# MODEL — loaded once, shared across requests
# =============================================================================

_model       = None
_device      = None
_predictor   = None
_estimator   = None
_nut_lookup  = None
_class_names = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def get_model() -> object:
    """
    Load model once at startup — reused for every request.
    Returns trained ResNet with Grad-CAM hook registered.
    """
    global _model
    if _model is None:
        cfg    = get_data_cfg()
        device = get_device()
        _model = build_model(
            model_cfg_path="configs/model_config.yaml",
            data_cfg_path ="configs/data_config.yaml",
        ).to(device)

        import os
        ckpt_path = "logs/checkpoints/best.pth"
        if os.path.exists(ckpt_path):
            load_checkpoint_with_hooks(ckpt_path, _model, device=device)
        else:
            # ONNX fallback or untrained — warn but don't crash
            import logging
            logging.getLogger(__name__).warning(
                f"Checkpoint not found at {ckpt_path} — using untrained model"
            )
        _model.eval()
    return _model


def get_class_names() -> list[str]:
    global _class_names
    if _class_names is None:
        from pathlib import Path
        cfg  = get_data_cfg()
        path = Path(cfg["dataset"]["class_names_file"])
        _class_names = [
            c.strip() for c in path.read_text().splitlines() if c.strip()
        ]
    return _class_names


def get_predictor() -> TTAPredictor:
    """TTA predictor — singleton, reused across requests."""
    global _predictor
    if _predictor is None:
        _predictor = TTAPredictor.from_config(
            model   = get_model(),
            device  = get_device(),
            cfg     = get_data_cfg(),
            use_tta = True,
        )
    return _predictor


def get_uncertainty_estimator() -> UncertaintyEstimator:
    """MC Dropout estimator — singleton."""
    global _estimator
    if _estimator is None:
        _estimator = UncertaintyEstimator(
            model    = get_model(),
            device   = get_device(),
            cfg      = get_data_cfg(),
            n_passes = 20,
        )
    return _estimator


def get_nutrition_lookup() -> NutritionLookup:
    """NutritionLookup — singleton, CSV read once."""
    global _nut_lookup
    if _nut_lookup is None:
        _nut_lookup = NutritionLookup.from_config(get_data_cfg())
    return _nut_lookup
