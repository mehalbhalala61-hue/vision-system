# =============================================================================
# api/routes/meal.py — /log  /dashboard  /chat  /plan
# =============================================================================
# All meal-related endpoints:
#   POST /log        — save a meal to PostgreSQL
#   GET  /dashboard  — daily/weekly nutrition summary
#   POST /chat       — Gemini AI health advice
#   POST /plan       — LangChain agentic meal planner
# =============================================================================

import asyncio
import logging
from datetime import datetime, timezone, date
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from api.deps import (
    get_db, get_nutrition_lookup,
    get_data_cfg, get_ai_cfg,
)
from db.models.meal_log import MealLog

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class MealLogRequest(BaseModel):
    user_id:   str
    food_name: str
    serving_g: float = 100.0
    meal_type: str   = "lunch"
    confidence: float = 0.0
    image_path: str  = None
    gradcam_path: str = None

class ChatRequest(BaseModel):
    user_id:     str
    food_name:   str
    serving_g:   float = 100.0
    goal_cal:    float = 2000.0

class PlanRequest(BaseModel):
    user_id:  str
    goal_cal: float = 2000.0


# =============================================================================
# POST /log — save meal to PostgreSQL
# =============================================================================

@router.post("/log")
def log_meal(
    req:        MealLogRequest,
    db:         Session       = Depends(get_db),
    nut_lookup                = Depends(get_nutrition_lookup),
):
    """
    Save a logged meal to PostgreSQL.

    Nutrition values are computed from NutritionLookup (USDA CSV)
    and stored with the log entry for fast dashboard queries.
    """
    # Get nutrition facts
    nutrition = nut_lookup.get(req.food_name, req.serving_g) if nut_lookup.is_enabled() else {}

    entry = MealLog(
        user_id      = req.user_id,
        food_name    = req.food_name,
        confidence   = req.confidence,
        serving_g    = req.serving_g,
        calories     = nutrition.get("calories"),
        protein_g    = nutrition.get("protein_g"),
        carbs_g      = nutrition.get("carbs_g"),
        fat_g        = nutrition.get("fat_g"),
        fiber_g      = nutrition.get("fiber_g"),
        meal_type    = req.meal_type,
        image_path   = req.image_path,
        gradcam_path = req.gradcam_path,
        logged_at    = datetime.now(timezone.utc),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    logger.info(f"Meal logged: user={req.user_id} food={req.food_name} cal={nutrition.get('calories')}")

    return JSONResponse({
        "success":  True,
        "meal_id":  entry.id,
        "food":     req.food_name,
        "calories": nutrition.get("calories"),
        "message":  f"Logged {req.food_name} ({req.serving_g}g)",
    })


# =============================================================================
# GET /dashboard — daily + weekly nutrition summary
# =============================================================================

@router.get("/dashboard")
def dashboard(
    user_id: str,
    date_str: str = None,
    db:       Session = Depends(get_db),
):
    """
    Return daily and weekly nutrition summary for a user.

    Args:
        user_id  : user identifier
        date_str : date in YYYY-MM-DD format (default: today UTC)
    """
    # Parse date
    if date_str:
        try:
            target_date = date.fromisoformat(date_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format: {date_str}. Use YYYY-MM-DD"
            )
    else:
        target_date = datetime.now(timezone.utc).date()

    # Query today's logs
    today_logs = (
        db.query(MealLog)
        .filter(
            MealLog.user_id == user_id,
            func.date(MealLog.logged_at) == target_date,
        )
        .order_by(MealLog.logged_at)
        .all()
    )

    def _aggregate(logs: list[MealLog]) -> dict:
        return {
            "total_calories":  round(sum(m.calories  or 0 for m in logs), 1),
            "total_protein_g": round(sum(m.protein_g or 0 for m in logs), 1),
            "total_carbs_g":   round(sum(m.carbs_g   or 0 for m in logs), 1),
            "total_fat_g":     round(sum(m.fat_g     or 0 for m in logs), 1),
            "total_fiber_g":   round(sum(m.fiber_g   or 0 for m in logs), 1),
            "n_meals":         len(logs),
        }

    daily   = _aggregate(today_logs)
    meals   = [m.to_dict() for m in today_logs]

    return JSONResponse({
        "user_id": user_id,
        "date":    str(target_date),
        "daily":   daily,
        "meals":   meals,
    })


# =============================================================================
# POST /chat — Gemini AI health advice
# =============================================================================

@router.post("/chat")
async def chat(
    req:        ChatRequest,
    db:         Session = Depends(get_db),
    nut_lookup          = Depends(get_nutrition_lookup),
    ai_cfg              = Depends(get_ai_cfg),
):
    """
    Get Gemini AI health advice for a meal.

    Sends food name + daily total to Gemini 1.5-flash.
    asyncio.wait_for() timeout — demo never crashes on rate limits.
    Exponential backoff retry: 1s → 2s → 4s.
    """
    import os
    import google.generativeai as genai

    gemini_cfg = ai_cfg["gemini"]
    api_key    = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        return JSONResponse(
            {"advice": "Gemini API key not configured. Add GEMINI_API_KEY to .env"},
            status_code=503,
        )

    # Get daily summary from DB
    target_date = datetime.now(timezone.utc).date()
    today_logs  = (
        db.query(MealLog)
        .filter(
            MealLog.user_id == req.user_id,
            func.date(MealLog.logged_at) == target_date,
        )
        .all()
    )
    daily_total = {
        "total_calories": sum(m.calories or 0 for m in today_logs),
        "n_meals":        len(today_logs),
    }

    # Build prompt using NutritionLookup context formatter
    nutrition_context = nut_lookup.format_for_gemini(
        food_name   = req.food_name,
        serving_g   = req.serving_g,
        daily_total = daily_total,
        goal_cal    = req.goal_cal,
    )

    prompt = (
        f"You are a friendly nutritionist. Give practical, specific health advice.\n\n"
        f"{nutrition_context}\n\n"
        f"In 2-3 sentences: comment on this meal's nutritional value and suggest "
        f"what to eat next to meet the daily goal of {req.goal_cal:.0f} kcal."
    )

    # Call Gemini with retry + timeout
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(
        model_name        = gemini_cfg["model"],
        generation_config = genai.GenerationConfig(
            temperature       = gemini_cfg["temperature"],
            max_output_tokens = gemini_cfg["max_output_tokens"],
        ),
    )

    backoff = gemini_cfg["retry"]["backoff_seconds"]
    advice  = None
    last_err = None

    for attempt, sleep_secs in enumerate(backoff, start=1):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(gemini_model.generate_content, prompt),
                timeout=gemini_cfg["timeout_seconds"],
            )
            advice = response.text.strip()
            break
        except asyncio.TimeoutError:
            last_err = "Gemini timeout"
            logger.warning(f"Gemini timeout (attempt {attempt}/{len(backoff)})")
        except Exception as e:
            last_err = str(e)
            logger.warning(f"Gemini error attempt {attempt}: {e}")
        await asyncio.sleep(sleep_secs)

    if advice is None:
        advice = f"Nutrition advice unavailable ({last_err}). Please try again."

    return JSONResponse({
        "food_name":     req.food_name,
        "advice":        advice,
        "daily_total":   daily_total,
        "nutrition_context": nutrition_context,
    })


# =============================================================================
# POST /plan — LangChain Agentic Meal Planner
# =============================================================================

@router.post("/plan")
async def plan_meals(
    req:    PlanRequest,
    db:     Session = Depends(get_db),
    ai_cfg          = Depends(get_ai_cfg),
):
    """
    Agentic meal planning using LangChain.

    Agent autonomously:
        1. Queries today's meal history from PostgreSQL
        2. Calculates remaining calories
        3. Generates 2-3 meal suggestions

    No hardcoded logic — agent decides the sequence.
    asyncio.wait_for() timeout prevents hanging.

    Interview note:
        "Agentic AI means the model decides which tools to call and
         in what order. My agent queries DB, calculates remaining
         calories, then generates suggestions — zero hardcoded logic."
    """
    from agents.meal_planner import run_agent_safe

    try:
        result = await asyncio.wait_for(
            run_agent_safe(
                user_id  = req.user_id,
                goal_cal = req.goal_cal,
                db       = db,
                ai_cfg   = ai_cfg,
            ),
            timeout=ai_cfg["langchain_agent"]["timeout_seconds"],
        )
        return JSONResponse(result)

    except asyncio.TimeoutError:
        logger.warning(f"LangChain agent timeout for user {req.user_id}")
        return JSONResponse(
            {"plan": "Meal planning timed out. Please try again.", "error": "timeout"},
            status_code=504,
        )
    except Exception as e:
        logger.error(f"LangChain agent error: {e}")
        return JSONResponse(
            {"plan": "Meal planning unavailable.", "error": str(e)},
            status_code=500,
        )
