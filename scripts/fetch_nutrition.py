# =============================================================================
# scripts/fetch_nutrition.py — USDA FoodData Central Nutrition Fetcher
# =============================================================================
# Reads search terms from data/usda_map.json (generated ONCE by
# generate_usda_map.py) — never calls Gemini directly.
#
# Usage:
#   # Step 1 (once per dataset):
#   python scripts/generate_usda_map.py
#
#   # Step 2 (fetch actual nutrition data):
#   python scripts/fetch_nutrition.py               # DEMO_KEY (30 req/hr)
#   python scripts/fetch_nutrition.py --key YOUR_KEY  # free key (3600 req/hr)
#   python scripts/fetch_nutrition.py --dry-run       # validate CSV, no API
#
# Output: data/nutrition.csv
#
# Interview note:
#   "Gemini is used exactly once — to translate class names into English
#    search terms, saved to usda_map.json. All actual nutrition values come
#    from the USDA FoodData Central API — real, verifiable, defensible."
# =============================================================================

import os
import time
import argparse
import logging
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Import the load helper — same file that generated the map
from scripts.generate_usda_map import load_usda_map, validate_map

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# PATHS + CONFIG
# -----------------------------------------------------------------------------
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
OUTPUT_CSV    = Path("data/nutrition.csv")
CLASSES_FILE  = Path("data/classes.txt")
USDA_MAP_FILE = Path("data/usda_map.json")

# Serving sizes + meal types — domain knowledge, not from USDA
SERVING_META: dict[str, tuple[float, str]] = {
    "biryani": (300, "lunch"),       "butter_chicken": (250, "dinner"),
    "dal_makhani": (200, "dinner"),  "palak_paneer": (200, "dinner"),
    "paneer_butter_masala": (200, "dinner"), "kadai_paneer": (200, "dinner"),
    "chicken_tikka": (200, "dinner"), "chicken_tikka_masala": (250, "dinner"),
    "chapati": (40, "breakfast"),    "naan": (80, "dinner"),
    "idli": (50, "breakfast"),       "dosa": (100, "breakfast"),
    "masala_dosa": (150, "breakfast"), "chole_bhature": (250, "breakfast"),
    "pav_bhaji": (200, "lunch"),     "samosa": (80, "snack"),
    "jalebi": (80, "dessert"),       "gulab_jamun": (80, "dessert"),
    "lassi": (250, "breakfast"),     "mango_lassi": (250, "breakfast"),
    "kheer": (150, "dessert"),       "kulfi": (80, "dessert"),
    "poha": (200, "breakfast"),      "upma": (200, "breakfast"),
    "uttapam": (120, "breakfast"),   "medu_vada": (60, "breakfast"),
    "dhokla": (100, "breakfast"),    "aloo_paratha": (120, "breakfast"),
}

FALLBACK_NUTRITION: dict[str, dict] = {
    "_default_snack":   {"calories_per_100g": 380, "protein_g": 6.0,  "carbs_g": 55.0, "fat_g": 14.0, "fiber_g": 2.0,  "serving_size_g": 60,  "meal_type": "snack"},
    "_default_curry":   {"calories_per_100g": 140, "protein_g": 7.0,  "carbs_g": 16.0, "fat_g": 5.0,  "fiber_g": 3.5,  "serving_size_g": 200, "meal_type": "lunch"},
    "_default_bread":   {"calories_per_100g": 280, "protein_g": 8.0,  "carbs_g": 46.0, "fat_g": 7.0,  "fiber_g": 3.0,  "serving_size_g": 80,  "meal_type": "breakfast"},
    "_default_dessert": {"calories_per_100g": 290, "protein_g": 5.0,  "carbs_g": 42.0, "fat_g": 10.0, "fiber_g": 0.5,  "serving_size_g": 100, "meal_type": "dessert"},
}


# =============================================================================
# USDA API
# =============================================================================

def search_usda(query: str, api_key: str) -> dict | None:
    """
    Query USDA FoodData Central for a single food item.
    Returns nutrient dict or None if not found.
    """
    url    = f"{USDA_BASE_URL}/foods/search"
    params = {
        "query":    query,
        "api_key":  api_key,
        "pageSize": 1,
        "dataType": "SR Legacy,Foundation,Survey (FNDDS)",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data  = resp.json()
        foods = data.get("foods", [])
        if not foods:
            return None

        food      = foods[0]
        nutrients = {n["nutrientName"]: n["value"] for n in food.get("foodNutrients", [])}

        return {
            "calories_per_100g": round(nutrients.get("Energy", 0), 1),
            "protein_g":         round(nutrients.get("Protein", 0), 1),
            "carbs_g":           round(nutrients.get("Carbohydrate, by difference", 0), 1),
            "fat_g":             round(nutrients.get("Total lipid (fat)", 0), 1),
            "fiber_g":           round(nutrients.get("Fiber, total dietary", 0), 1),
            "usda_fdc_id":       food.get("fdcId", ""),
            "usda_description":  food.get("description", ""),
        }

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning("Rate limit hit — sleeping 60s...")
            time.sleep(60)
            return search_usda(query, api_key)      # one retry
        logger.error(f"USDA HTTP error for '{query}': {e}")
        return None
    except Exception as e:
        logger.error(f"USDA error for '{query}': {e}")
        return None


def get_fallback(class_name: str) -> dict:
    """Heuristic fallback when USDA returns no match."""
    name = class_name.lower()
    if any(w in name for w in ["halwa", "kheer", "jamun", "laddu", "barfi",
                                "jalebi", "kulfi", "peda", "sandesh", "rabri",
                                "payasam", "phirni", "shrikhand", "basundi"]):
        return dict(FALLBACK_NUTRITION["_default_dessert"])
    elif any(w in name for w in ["roti", "naan", "paratha", "bhatura", "dosa",
                                  "idli", "chapati", "puri", "poha", "upma"]):
        return dict(FALLBACK_NUTRITION["_default_bread"])
    elif any(w in name for w in ["masala", "curry", "dal", "sabzi", "bhaji",
                                  "korma", "chicken", "paneer", "aloo", "gobi"]):
        return dict(FALLBACK_NUTRITION["_default_curry"])
    else:
        return dict(FALLBACK_NUTRITION["_default_snack"])


# =============================================================================
# MAIN FETCH
# =============================================================================

def fetch_all(api_key: str, dry_run: bool = False) -> pd.DataFrame:
    """
    Fetch nutrition for all classes using the cached usda_map.json.

    Flow:
        1. Load classes from data/classes.txt
        2. Load search terms from data/usda_map.json   ← NO Gemini call here
        3. Query USDA API for each term
        4. Save to data/nutrition.csv
    """
    if not CLASSES_FILE.exists():
        raise FileNotFoundError(
            f"{CLASSES_FILE} not found.\n"
            "Run from project root: python scripts/fetch_nutrition.py"
        )

    classes = [c.strip() for c in CLASSES_FILE.read_text().splitlines() if c.strip()]
    logger.info(f"Classes: {len(classes)} loaded from {CLASSES_FILE}")

    # ── Load USDA map from JSON (generated once by generate_usda_map.py) ──
    try:
        usda_map = load_usda_map(USDA_MAP_FILE)
        logger.info(f"USDA map: {len(usda_map)} entries loaded from {USDA_MAP_FILE}")
        validate_map(usda_map, classes)
    except FileNotFoundError:
        logger.error(
            f"\n{USDA_MAP_FILE} not found!\n"
            "Run this FIRST (only once):\n"
            "  python scripts/generate_usda_map.py\n"
            "Then re-run this script."
        )
        raise

    # ── Dry run: validate only ──
    if dry_run:
        logger.info("DRY RUN — no USDA API calls")
        if OUTPUT_CSV.exists():
            df = pd.read_csv(OUTPUT_CSV)
            logger.info(f"Existing CSV: {len(df)} rows | columns: {list(df.columns)}")
            missing = set(classes) - set(df["food_name"])
            if missing:
                logger.warning(f"Missing in CSV: {missing}")
            else:
                logger.info("✓ All classes present in CSV")
        else:
            logger.warning("No existing CSV found — run without --dry-run")
        return pd.DataFrame()

    # ── Fetch from USDA ──
    rows = []
    for i, class_name in enumerate(classes):
        # Use map entry if available, else fall back to plain name
        query = usda_map.get(class_name, class_name.replace("_", " "))
        logger.info(f"[{i+1:3d}/{len(classes)}] {class_name:40s} → '{query}'")

        nutrients = search_usda(query, api_key)

        if nutrients:
            logger.info(f"  ✓ {nutrients.get('usda_description', '')[:55]}")
            source = "usda"
        else:
            logger.warning(f"  ✗ No match — fallback")
            nutrients = get_fallback(class_name)
            source    = "fallback"

        serving_g, meal_type = SERVING_META.get(class_name, (100.0, "snack"))

        rows.append({
            "food_name":         class_name,
            "calories_per_100g": nutrients.get("calories_per_100g", 0),
            "protein_g":         nutrients.get("protein_g", 0),
            "carbs_g":           nutrients.get("carbs_g", 0),
            "fat_g":             nutrients.get("fat_g", 0),
            "fiber_g":           nutrients.get("fiber_g", 0),
            "serving_size_g":    serving_g,
            "meal_type":         meal_type,
            "source":            source,
        })

        # Rate limit: DEMO_KEY = 30/hr, free key = 3600/hr
        time.sleep(1.5 if api_key == "DEMO_KEY" else 0.3)

    df = pd.DataFrame(rows)

    # Warn on zero-calorie rows (bad API response)
    zero_cal = df[df["calories_per_100g"] == 0]
    if not zero_cal.empty:
        logger.warning(f"Zero calories — check these: {list(zero_cal['food_name'])}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    usda_count     = (df["source"] == "usda").sum()
    fallback_count = (df["source"] == "fallback").sum()
    logger.info(f"\nSaved {len(df)} rows → {OUTPUT_CSV}")
    logger.info(f"USDA: {usda_count} | Fallback: {fallback_count}")

    return df


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch USDA nutrition data using cached usda_map.json"
    )
    parser.add_argument(
        "--key",
        default=os.getenv("USDA_API_KEY", "DEMO_KEY"),
        help="USDA API key (default: DEMO_KEY or USDA_API_KEY env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate existing CSV without API calls",
    )
    args = parser.parse_args()

    if args.key == "DEMO_KEY":
        logger.warning(
            "Using DEMO_KEY — 30 req/hr limit.\n"
            "Get a free key: https://fdc.nal.usda.gov/api-guide.html\n"
            "Set in .env: USDA_API_KEY=your_key"
        )

    fetch_all(api_key=args.key, dry_run=args.dry_run)
