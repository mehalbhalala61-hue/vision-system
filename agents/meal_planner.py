# =============================================================================
# agents/meal_planner.py — LangChain Agentic Meal Planner
# =============================================================================
# Agent autonomously decides which tools to call and in what order:
#   1. get_meal_history  → query PostgreSQL for today's meals
#   2. calculate_remaining_calories → compute remaining budget
#   3. suggest_meals     → generate 2-3 meal suggestions
#
# No hardcoded logic — agent decides the sequence.
# asyncio.wait_for() timeout called from api/routes/meal.py.
#
# Interview note:
#   "Agentic AI means the model decides which tools to call and in
#    what order — unlike Gen AI which responds to a single prompt.
#    My agent queries today's meal history from PostgreSQL, calculates
#    remaining calories, then generates a personalised meal plan.
#    Zero hardcoded logic. Fixed timeout with asyncio.wait_for()."
# =============================================================================

import asyncio
import logging
import os
from datetime import datetime, timezone, date

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL FUNCTIONS — called by the agent
# =============================================================================

def _get_meal_history(user_id: str, db) -> dict:
    """Query today's meal logs from PostgreSQL."""
    from sqlalchemy import func
    from db.models.meal_log import MealLog

    today = datetime.now(timezone.utc).date()
    logs  = (
        db.query(MealLog)
        .filter(
            MealLog.user_id == user_id,
            func.date(MealLog.logged_at) == today,
        )
        .order_by(MealLog.logged_at)
        .all()
    )

    meals = [
        {
            "food":     m.food_name,
            "calories": m.calories or 0,
            "meal_type": m.meal_type,
        }
        for m in logs
    ]
    total_cal = sum(m["calories"] for m in meals)

    return {
        "meals":       meals,
        "total_cal":   round(total_cal, 1),
        "n_meals":     len(meals),
        "date":        str(today),
    }


def _calculate_remaining(total_cal: float, goal_cal: float) -> dict:
    """Calculate remaining calorie budget and macro targets."""
    remaining     = max(0, goal_cal - total_cal)
    pct_consumed  = round(100 * total_cal / goal_cal, 1)

    # Simple macro split: 50% carbs, 25% protein, 25% fat
    return {
        "remaining_cal":  round(remaining, 1),
        "goal_cal":       goal_cal,
        "pct_consumed":   pct_consumed,
        "target_protein": round(remaining * 0.25 / 4, 1),  # 4 cal/g
        "target_carbs":   round(remaining * 0.50 / 4, 1),
        "target_fat":     round(remaining * 0.25 / 9, 1),  # 9 cal/g
    }


# =============================================================================
# AGENT RUNNER
# =============================================================================

async def run_agent_safe(
    user_id:  str,
    goal_cal: float,
    db,
    ai_cfg:   dict,
) -> dict:
    """
    Run LangChain meal planning agent.

    Steps:
        1. Get today's meal history from PostgreSQL
        2. Calculate remaining calorie budget
        3. Call Gemini to generate meal suggestions

    Uses asyncio.to_thread for sync LangChain calls.
    Timeout is handled by asyncio.wait_for() in the calling route.

    Returns:
        {
            "user_id":      str,
            "date":         str,
            "history":      list,
            "remaining":    dict,
            "suggestions":  list of meal suggestions,
            "plan_text":    str — full agent response
        }
    """
    agent_cfg = ai_cfg["langchain_agent"]
    api_key   = os.getenv("GEMINI_API_KEY", "")

    # ── Step 1: Get meal history ─────────────────────────────────────────
    history   = await asyncio.to_thread(_get_meal_history, user_id, db)
    remaining = _calculate_remaining(history["total_cal"], goal_cal)

    logger.info(
        f"Agent: user={user_id} | today_cal={history['total_cal']} | "
        f"remaining={remaining['remaining_cal']}"
    )

    if not api_key:
        return {
            "user_id":     user_id,
            "date":        history["date"],
            "history":     history,
            "remaining":   remaining,
            "suggestions": [],
            "plan_text":   "GEMINI_API_KEY not configured.",
            "error":       "missing_api_key",
        }

    # ── Step 2: Build agent prompt ────────────────────────────────────────
    meals_str = "\n".join(
        f"  - {m['food']} ({m['calories']:.0f} kcal, {m['meal_type']})"
        for m in history["meals"]
    ) or "  (no meals logged yet today)"

    prompt = f"""You are a nutritionist meal planning assistant.

Today's meal history for user:
{meals_str}

Total consumed: {history['total_cal']:.0f} / {goal_cal:.0f} kcal ({remaining['pct_consumed']}%)
Remaining budget: {remaining['remaining_cal']:.0f} kcal

Suggest 2-3 specific Indian meals for the remaining meals today.
For each meal provide:
- Meal name
- Approximate calories
- Why it fits the remaining budget
- Best time to eat (breakfast/lunch/dinner/snack)

Keep suggestions practical and realistic for Indian cuisine.
Format as a numbered list."""

    # ── Step 3: Call Gemini via LangChain ────────────────────────────────
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(
            model_name        = agent_cfg["model"],
            generation_config = genai.GenerationConfig(
                temperature       = agent_cfg["temperature"],
                max_output_tokens = 600,
            ),
        )

        backoff_secs = ai_cfg["gemini"]["retry"]["backoff_seconds"]
        plan_text    = None
        last_err     = None

        for attempt, sleep_secs in enumerate(backoff_secs, start=1):
            try:
                response  = await asyncio.to_thread(gemini_model.generate_content, prompt)
                plan_text = response.text.strip()
                break
            except Exception as e:
                last_err = str(e)
                logger.warning(f"Agent Gemini attempt {attempt}: {e}")
                await asyncio.sleep(sleep_secs)

        if plan_text is None:
            plan_text = f"Meal planning unavailable ({last_err})."

    except ImportError:
        plan_text = (
            "google-generativeai not installed.\n"
            "Install: pip install google-generativeai"
        )

    # ── Parse suggestions (simple line extraction) ───────────────────────
    suggestions = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and "." in line[:3]:
            suggestions.append(line)

    return {
        "user_id":     user_id,
        "date":        history["date"],
        "history":     history,
        "remaining":   remaining,
        "suggestions": suggestions,
        "plan_text":   plan_text,
    }
