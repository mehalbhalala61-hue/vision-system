# =============================================================================
# api/routes/predict.py — POST /predict
# =============================================================================
# Accepts food image → returns class + confidence + nutrition + Grad-CAM path
# Uses TTA for accuracy + MC Dropout for uncertainty flagging.
# =============================================================================

import io
import logging
import time
import uuid
import torch
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse

from api.deps import (
    get_predictor, get_uncertainty_estimator,
    get_nutrition_lookup, get_class_names,
    get_data_cfg, get_model, get_device,
)
from evaluation.gradcam import GradCAM, overlay_heatmap
from utils.checkpoint import load_checkpoint_with_hooks

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# POST /predict
# =============================================================================

@router.post("/predict")
async def predict(
    file:        UploadFile = File(..., description="Food image (jpg/png)"),
    predictor    = Depends(get_predictor),
    estimator    = Depends(get_uncertainty_estimator),
    nut_lookup   = Depends(get_nutrition_lookup),
    class_names  = Depends(get_class_names),
    data_cfg     = Depends(get_data_cfg),
    model        = Depends(get_model),
    device       = Depends(get_device),
):
    """
    Predict food class from image.

    Returns:
    - class name + confidence (top-5)
    - Grad-CAM heatmap path
    - Nutrition facts (USDA-sourced)
    - MC Dropout uncertainty flag

    Interview note:
        "Single endpoint does 4 things: TTA prediction, Grad-CAM
         visualisation, USDA nutrition lookup, MC Dropout uncertainty
         — all within one request. Uncertainty > 0.15 flags for review."
    """
    t_start = time.perf_counter()

    # ── Validate file ────────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use jpg/png/webp."
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > 10 * 1024 * 1024:   # 10MB limit
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    # ── Load image ───────────────────────────────────────────────────────
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    # ── TTA Prediction ───────────────────────────────────────────────────
    prediction = predictor.predict(pil_image, top_k=5)
    pred_class = prediction["class"]
    confidence = prediction["confidence"]

    # ── MC Dropout Uncertainty ───────────────────────────────────────────
    from torchvision import transforms
    img_size  = data_cfg["image"]["size"]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=data_cfg["image"]["mean"],
            std=data_cfg["image"]["std"],
        ),
    ])
    img_tensor  = transform(pil_image)
    unc_result  = estimator.predict(pil_image)

    # ── Grad-CAM ─────────────────────────────────────────────────────────
    gradcam_path = None
    try:
        gradcam     = GradCAM(model, device)
        tensor_4d   = img_tensor.unsqueeze(0).to(device)
        pred_idx    = class_names.index(pred_class) if pred_class in class_names else 0
        heatmap     = gradcam.generate(tensor_4d, target_class=pred_idx)
        gradcam.remove_hooks()

        # Save overlay
        img_np   = np.array(pil_image.resize((img_size, img_size)))
        overlay  = overlay_heatmap(img_np, heatmap)
        out_name = f"{uuid.uuid4().hex[:8]}.png"
        out_dir  = Path("logs/gradcam_outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(out_dir / out_name)
        gradcam_path = f"logs/gradcam_outputs/{out_name}"

        # Re-register hook (always after Grad-CAM use)
        model._register_gradcam_hook()

    except Exception as e:
        logger.warning(f"Grad-CAM failed (non-fatal): {e}")

    # ── Nutrition ────────────────────────────────────────────────────────
    nutrition = {}
    if nut_lookup.is_enabled():
        nutrition = nut_lookup.get(pred_class)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

    return JSONResponse({
        "prediction": {
            "class":      pred_class,
            "confidence": confidence,
            "top_5":      prediction["top_k"],
            "used_tta":   prediction["used_tta"],
        },
        "uncertainty": {
            "value":       unc_result["uncertainty"],
            "flag_review": unc_result["flag_review"],
        },
        "gradcam_path": gradcam_path,
        "nutrition":    nutrition,
        "elapsed_ms":   elapsed_ms,
    })
