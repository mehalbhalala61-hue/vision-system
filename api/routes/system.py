# =============================================================================
# api/routes/system.py — /health  /model_info
# =============================================================================

import time
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.deps import get_data_cfg, get_model_cfg, get_model, get_device
from db.session import check_db_connection

logger    = logging.getLogger(__name__)
router    = APIRouter()
_start_ts = time.time()


@router.get("/health")
def health(
    model      = Depends(get_model),
    device     = Depends(get_device),
):
    """
    Health check — used by Railway healthcheckPath.
    Returns 200 if model loaded + DB reachable.
    Returns 503 if DB unreachable (model still works).
    """
    db_ok      = check_db_connection()
    uptime_sec = round(time.time() - _start_ts, 1)

    status = "ok" if db_ok else "degraded"

    return JSONResponse(
        content={
            "status":       status,
            "uptime_seconds": uptime_sec,
            "model_loaded": model is not None,
            "device":       str(device),
            "database":     "connected" if db_ok else "unreachable",
        },
        status_code=200 if db_ok else 503,
    )


@router.get("/model_info")
def model_info(
    data_cfg  = Depends(get_data_cfg),
    model_cfg = Depends(get_model_cfg),
    model     = Depends(get_model),
):
    """
    Returns current model + dataset configuration.
    Dynamic — reflects whatever is in data_config.yaml.
    Shows dataset name so interviewer can see it's config-driven.
    """
    return JSONResponse({
        "dataset":      data_cfg["dataset"]["name"],
        "num_classes":  data_cfg["dataset"]["num_classes"],
        "architecture": f"ResNet-{model_cfg['arch']['depth']}",
        "se_blocks":    model_cfg["blocks"]["se_block"]["enabled"],
        "params":       model.count_params() if model else 0,
        "image_size":   data_cfg["image"]["size"],
        "nutrition_enabled": data_cfg["nutrition"]["enabled"],
        "tta_enabled":  True,
        "mc_dropout_passes": 20,
    })
