# =============================================================================
# api/main.py — FastAPI Application Entry Point
# =============================================================================
# Reads dataset name from data_config.yaml → shown in API title.
# All 7 endpoints registered via routers.
# Startup: DB tables created, model loaded.
#
# Run locally:
#   uvicorn api.main:app --reload --port 8000
#   Open: http://localhost:8000/docs
# =============================================================================

import logging
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import predict, meal, system
from db.session import init_db

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB tables + pre-load model."""
    logger.info("Starting up Vision System API...")

    # Create DB tables (dev/test only — production uses Alembic)
    try:
        init_db()
        logger.info("DB tables ready")
    except Exception as e:
        logger.warning(f"DB init warning (non-fatal): {e}")

    # Pre-load model so first request isn't slow
    from api.deps import get_model, get_predictor, get_nutrition_lookup
    try:
        model = get_model()
        _     = get_predictor()
        _     = get_nutrition_lookup()
        logger.info(f"Model loaded: {model.count_params():,} params")
    except Exception as e:
        logger.warning(f"Model pre-load warning: {e}")

    yield

    logger.info("Shutting down...")


# =============================================================================
# APP — title reads from data_config.yaml (dataset-agnostic)
# =============================================================================

def _get_dataset_name() -> str:
    try:
        with open("configs/data_config.yaml") as f:
            cfg = yaml.safe_load(f)
        return cfg["dataset"]["name"].replace("_", " ").title()
    except Exception:
        return "Indian Food"


app = FastAPI(
    title       = f"Vision System Capstone v3 — {_get_dataset_name()}",
    description = (
        "Production-grade food classification API.\n\n"
        "**Stack:** ResNet-34 + SE · FastAPI · Gemini AI · "
        "LangChain · PostgreSQL · Docker · Railway\n\n"
        "**Features:** TTA · Grad-CAM · Temperature Scaling · "
        "MC Dropout · Agentic Meal Planning"
    ),
    version     = "3.0.0",
    lifespan    = lifespan,
)

# CORS — allow all origins for demo (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# =============================================================================
# ROUTERS
# =============================================================================

app.include_router(predict.router, tags=["Prediction"])
app.include_router(meal.router,    tags=["Meals"])
app.include_router(system.router,  tags=["System"])


# =============================================================================
# ROOT
# =============================================================================

@app.get("/", tags=["System"])
def root():
    return {
        "message": "Vision System Capstone v3 — API is running",
        "docs":    "/docs",
        "health":  "/health",
    }
