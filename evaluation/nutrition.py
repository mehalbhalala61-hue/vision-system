# =============================================================================
# evaluation/nutrition.py — Nutrition Lookup Module
# =============================================================================
# Reads data/nutrition.csv (USDA-sourced) and returns per-serving macros.
# Works for any number of classes — reads num_classes from data_config.yaml.
# Used by:
#   - FastAPI /predict endpoint  → nutrition facts per prediction
#   - FastAPI /chat endpoint     → calorie context for Gemini
#   - FastAPI /plan endpoint     → LangChain meal planner
#   - FastAPI /dashboard endpoint → daily totals
#
# Interview note:
#   "NutritionLookup reads from a USDA-sourced CSV — no live API calls
#    at inference time. Values are scaled to actual serving size. Source
#    field is stored so we can distinguish USDA-verified vs fallback values."
# =============================================================================

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# NUTRITION LOOKUP
# =============================================================================

class NutritionLookup:
    """
    Fast in-memory nutrition lookup from data/nutrition.csv.

    Scales all macros from per-100g to actual serving size.
    Source field: 'usda' = verified | 'fallback' = estimated.

    Args:
        csv_path : path to nutrition.csv
        cfg      : data_config.yaml dict (for nutrition.enabled check)

    Usage:
        lookup = NutritionLookup.from_config(cfg)
        info   = lookup.get("biryani", serving_g=300)
        # {food_name, calories, protein_g, carbs_g, fat_g, fiber_g,
        #   serving_g, meal_type, source, per_100g}
    """

    def __init__(self, csv_path: str, cfg: dict = None):
        self.csv_path        = Path(csv_path)
        self.cfg             = cfg
        self._nutrition_enabled = True

        # Check if nutrition is enabled in config
        if cfg and not cfg.get("nutrition", {}).get("enabled", True):
            self._nutrition_enabled = False
            logger.info("Nutrition disabled in data_config.yaml — NutritionLookup inactive")
            return

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"nutrition.csv not found at {self.csv_path}\n"
                "Run: python scripts/fetch_nutrition.py"
            )

        # Load CSV and index by food_name for O(1) lookup
        self._df = pd.read_csv(self.csv_path)
        self._df["food_name"] = self._df["food_name"].str.strip().str.lower()
        self._df = self._df.set_index("food_name")

        # Validate required columns
        required = [
            "calories_per_100g", "protein_g", "carbs_g",
            "fat_g", "fiber_g", "serving_size_g", "meal_type"
        ]
        missing_cols = [c for c in required if c not in self._df.columns]
        if missing_cols:
            raise ValueError(
                f"nutrition.csv missing columns: {missing_cols}\n"
                "Re-run: python scripts/fetch_nutrition.py"
            )

        logger.info(
            f"NutritionLookup loaded: {len(self._df)} classes | "
            f"source={self.csv_path}"
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "NutritionLookup":
        """
        Build NutritionLookup from data_config.yaml.
        Returns inactive instance if nutrition.enabled = false.
        """
        csv_path = cfg.get("nutrition", {}).get("csv_path", "data/nutrition.csv")
        return cls(csv_path=csv_path, cfg=cfg)

    def get(
        self,
        food_name:  str,
        serving_g:  Optional[float] = None,
    ) -> dict:
        """
        Get nutrition facts for a food, scaled to serving size.

        Args:
            food_name : class name from model prediction (e.g. "biryani")
            serving_g : serving size in grams.
                        None → use default from CSV.

        Returns:
            dict with per-serving macros, or empty dict if not found.

        Example:
            lookup.get("biryani", serving_g=300)
            # {
            #   "food_name":   "biryani",
            #   "calories":    588.0,      ← 196 cal/100g × 300g
            #   "protein_g":   42.6,
            #   "carbs_g":     80.4,
            #   "fat_g":       14.4,
            #   "fiber_g":     3.6,
            #   "serving_g":   300,
            #   "meal_type":   "lunch",
            #   "source":      "usda",
            #   "per_100g": { calories: 196, protein: 14.2, ... }
            # }
        """
        if not self._nutrition_enabled:
            return {}

        key = food_name.strip().lower()

        if key not in self._df.index:
            logger.warning(
                f"'{food_name}' not found in nutrition.csv — "
                "check classes.txt matches CSV food_name column"
            )
            return self._empty(food_name)

        row = self._df.loc[key]

        # Use provided serving_g or fall back to CSV default
        actual_serving = float(serving_g) if serving_g else float(row["serving_size_g"])
        scale          = actual_serving / 100.0

        # Per-100g values
        per_100g = {
            "calories": round(float(row["calories_per_100g"]), 1),
            "protein":  round(float(row["protein_g"]),         1),
            "carbs":    round(float(row["carbs_g"]),           1),
            "fat":      round(float(row["fat_g"]),             1),
            "fiber":    round(float(row["fiber_g"]),           1),
        }

        return {
            "food_name":  food_name,
            "calories":   round(per_100g["calories"] * scale, 1),
            "protein_g":  round(per_100g["protein"]  * scale, 1),
            "carbs_g":    round(per_100g["carbs"]    * scale, 1),
            "fat_g":      round(per_100g["fat"]      * scale, 1),
            "fiber_g":    round(per_100g["fiber"]    * scale, 1),
            "serving_g":  actual_serving,
            "meal_type":  str(row.get("meal_type", "snack")),
            "source":     str(row.get("source", "usda")),
            "per_100g":   per_100g,
        }

    def get_daily_summary(self, meal_logs: list[dict]) -> dict:
        """
        Compute daily nutrition totals from a list of meal log dicts.

        Args:
            meal_logs : list of dicts with keys:
                        food_name, serving_g (optional)

        Returns:
            {
                total_calories, total_protein_g, total_carbs_g,
                total_fat_g, total_fiber_g, n_meals, meals
            }
        """
        total_cal     = 0.0
        total_protein = 0.0
        total_carbs   = 0.0
        total_fat     = 0.0
        total_fiber   = 0.0
        meals         = []

        for log in meal_logs:
            info = self.get(
                food_name = log.get("food_name", ""),
                serving_g = log.get("serving_g"),
            )
            if info:
                total_cal     += info["calories"]
                total_protein += info["protein_g"]
                total_carbs   += info["carbs_g"]
                total_fat     += info["fat_g"]
                total_fiber   += info["fiber_g"]
                meals.append(info)

        return {
            "total_calories":  round(total_cal,     1),
            "total_protein_g": round(total_protein, 1),
            "total_carbs_g":   round(total_carbs,   1),
            "total_fat_g":     round(total_fat,     1),
            "total_fiber_g":   round(total_fiber,   1),
            "n_meals":         len(meals),
            "meals":           meals,
        }

    def get_all(self) -> pd.DataFrame:
        """Return the full nutrition DataFrame — used by EDA notebook."""
        if not self._nutrition_enabled:
            return pd.DataFrame()
        return self._df.reset_index()

    def is_enabled(self) -> bool:
        """Returns True if nutrition is enabled in config."""
        return self._nutrition_enabled

    def _empty(self, food_name: str) -> dict:
        """Return zero-value dict for unknown food."""
        return {
            "food_name":  food_name,
            "calories":   0.0,
            "protein_g":  0.0,
            "carbs_g":    0.0,
            "fat_g":      0.0,
            "fiber_g":    0.0,
            "serving_g":  100.0,
            "meal_type":  "unknown",
            "source":     "not_found",
            "per_100g":   {},
        }

    # ------------------------------------------------------------------
    # Context for Gemini /chat endpoint
    # ------------------------------------------------------------------

    def format_for_gemini(
        self,
        food_name:   str,
        serving_g:   float,
        daily_total: dict,
        goal_cal:    float = 2000.0,
    ) -> str:
        """
        Format nutrition context string for Gemini /chat prompt.
        Called by api/routes/meal.py.

        Args:
            food_name   : predicted food class
            serving_g   : serving size
            daily_total : from get_daily_summary()
            goal_cal    : daily calorie goal (default 2000)

        Returns:
            Formatted string to inject into Gemini prompt.
        """
        info         = self.get(food_name, serving_g)
        remaining    = goal_cal - daily_total.get("total_calories", 0)
        pct_consumed = 100 * daily_total.get("total_calories", 0) / goal_cal

        if not info:
            return (
                f"Food: {food_name} (nutrition data unavailable)\n"
                f"Daily total: {daily_total.get('total_calories', 0):.0f} / "
                f"{goal_cal:.0f} kcal ({pct_consumed:.0f}% consumed)"
            )

        return (
            f"Current meal: {food_name}\n"
            f"  Calories: {info['calories']:.0f} kcal "
            f"({info['serving_g']:.0f}g serving)\n"
            f"  Macros: {info['protein_g']:.1f}g protein | "
            f"{info['carbs_g']:.1f}g carbs | {info['fat_g']:.1f}g fat\n"
            f"\n"
            f"Daily total so far: "
            f"{daily_total.get('total_calories', 0):.0f} / {goal_cal:.0f} kcal "
            f"({pct_consumed:.0f}%)\n"
            f"Remaining: {remaining:.0f} kcal\n"
            f"Meals logged today: {daily_total.get('n_meals', 0)}\n"
            f"Source: {info['source'].upper()}"
        )
