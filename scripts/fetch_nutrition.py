# =============================================================================
# scripts/fetch_nutrition.py — USDA FoodData Central Nutrition Fetcher
# =============================================================================
# Reads search terms from data/usda_map.json (generated ONCE by
# generate_usda_map.py) — never calls Gemini directly... except for a
# SMALL, BOUNDED retry: if USDA returns no match, we ask Gemini ONE more
# time for a simpler/alternate 1-2 word search term and try again.
#
# Usage:
#   # Step 1 (once per dataset):
#   python scripts/generate_usda_map.py
#
#   # Step 2 (fetch actual nutrition data):
#   python scripts/fetch_nutrition.py               # DEMO_KEY (30 req/hr)
#   python scripts/fetch_nutrition.py --key YOUR_KEY  # free key (3600 req/hr)
#   python scripts/fetch_nutrition.py --dry-run       # validate CSV, no API
#   python scripts/fetch_nutrition.py --no-retry       # disable Gemini retry
#
# Output: data/nutrition.csv
#
# Interview note:
#   "Gemini does the primary translation once (cached to usda_map.json).
#    All nutrition values come from USDA FoodData Central — real and
#    verifiable. As a bounded enhancement, if USDA finds no match, we ask
#    Gemini for ONE alternate, simpler search term (e.g. just the main
#    ingredient) and retry once before falling back to heuristic defaults.
#    This is capped, logged, and never runs more than once per class —
#    so it stays predictable and cheap, not an open-ended agent loop."
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
RETRY_MAP_FILE = Path("data/usda_retry_map.json")  # cache for retry terms too

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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

FOOD_STOPWORDS = {
    "curry", "masala", "dish", "sweet", "fried", "spice", "spiced",
    "gravy", "snack", "mixed", "dessert", "with", "and", "the", "a",
    "of", "in", "style", "type", "sauce", "stew", "creamy", "syrup",
}


def _key_words(text: str) -> set[str]:
    """
    Extract meaningful ingredient-ish words from a query or description,
    lowercased, with generic filler words (curry, masala, sweet, etc.)
    stripped out. These filler words are exactly what causes false
    matches like 'potato cauliflower curry' -> 'Beef curry' — both share
    "curry" but no actual ingredient in common.
    """
    words = "".join(c if c.isalnum() else " " for c in text.lower()).split()
    return {w for w in words if w not in FOOD_STOPWORDS and len(w) > 2}


def _is_relevant_match(query: str, description: str) -> bool:
    """
    Returns True only if the USDA result shares at least one real
    ingredient/keyword with our search query. This rejects matches that
    only overlap on generic words like "curry" or "sweet" — e.g.
    'potato cauliflower curry' must NOT accept 'Beef curry', since
    "beef" never appeared in the query at all.
    """
    query_words = _key_words(query)
    desc_words  = _key_words(description)
    if not query_words:
        return True  # nothing meaningful to check against — don't block
    return bool(query_words & desc_words)


def search_usda(query: str, api_key: str) -> dict | None:
    """
    Query USDA FoodData Central for a single food item.
    Returns nutrient dict or None if not found / request rejected /
    the match is judged irrelevant (see _is_relevant_match).
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

        result = {
            "calories_per_100g": round(nutrients.get("Energy", 0), 1),
            "protein_g":         round(nutrients.get("Protein", 0), 1),
            "carbs_g":           round(nutrients.get("Carbohydrate, by difference", 0), 1),
            "fat_g":             round(nutrients.get("Total lipid (fat)", 0), 1),
            "fiber_g":           round(nutrients.get("Fiber, total dietary", 0), 1),
            "usda_fdc_id":       food.get("fdcId", ""),
            "usda_description":  food.get("description", ""),
        }

        # A "match" with zero calories is not a usable match (e.g. baking
        # soda matching "naan" — wrong food entirely). Treat as no-match
        # so the caller falls through to retry / fallback instead of
        # silently saving a 0-calorie row.
        if result["calories_per_100g"] <= 0:
            logger.warning(
                f"  Match '{result['usda_description']}' has 0 calories — "
                f"treating as no-match"
            )
            return None

        # Relevance check — USDA's search is keyword-overlap based, so it
        # can return things like "Beef curry" for "potato cauliflower
        # curry" (both share the generic word "curry", nothing else).
        # Reject matches that don't share a real ingredient word.
        if not _is_relevant_match(query, result["usda_description"]):
            logger.warning(
                f"  Match '{result['usda_description']}' shares no real "
                f"ingredient with query '{query}' — treating as no-match"
            )
            return None

        return result

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning("Rate limit hit — sleeping 60s...")
            time.sleep(60)
            return search_usda(query, api_key)      # one retry
        # 400s (bad query format) and other HTTP errors: treat as no-match,
        # let the retry-with-alternate-term logic decide what to do next.
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
# GEMINI — bounded single retry for an alternate search term
# =============================================================================

_gemini_model = None  # lazy singleton, only created if retry is enabled & needed


def _get_gemini_model():
    """Lazily configure and cache a Gemini model instance for retries."""
    global _gemini_model
    if _gemini_model is None:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set — required for --retry mode.\n"
                "Add it to .env, or run with --no-retry to skip this step."
            )
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(
            model_name=GEMINI_MODEL_NAME,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=30,
            ),
            system_instruction=(
                "You map Indian food dish names to a SHORT USDA "
                "FoodData Central search term. Reply with ONLY the search "
                "term — no quotes, no punctuation, no explanation. "
                "Keep it to 1-2 words, focused on the single main "
                "ingredient (e.g. 'paneer', 'lentils', 'rice', 'chicken'). "
                "This is a fallback after a longer description already "
                "failed to match, so go simpler and more generic, not "
                "more specific."
            ),
        )
    return _gemini_model


def get_alternate_term(class_name: str, failed_term: str) -> str | None:
    """
    Ask Gemini for ONE alternate, simpler USDA search term after the
    primary term (from usda_map.json) failed to find a usable match.

    Bounded: exactly one call, one retry, per class — never loops.
    Returns None if Gemini itself fails (caller should fall back).
    """
    try:
        model = _get_gemini_model()
        prompt = (
            f"Dish: {class_name.replace('_', ' ')}\n"
            f"Already tried (failed): '{failed_term}'\n"
            f"Give one simpler, more generic search term."
        )
        response = model.generate_content(prompt)
        term = response.text.strip().strip('"').strip("'")
        return term if term else None
    except Exception as e:
        logger.warning(f"  Gemini retry call failed: {e}")
        return None


# =============================================================================
# MAIN FETCH
# =============================================================================

def fetch_all(api_key: str, dry_run: bool = False, use_retry: bool = True) -> pd.DataFrame:
    """
    Fetch nutrition for all classes using the cached usda_map.json.

    Flow per class:
        1. Try USDA with the cached term from usda_map.json
        2. If no usable match AND use_retry: ask Gemini once for a
           simpler alternate term, try USDA again
        3. If still no match: heuristic fallback (no API/AI calls)
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

    # ── Load (or start) the retry-term cache so re-runs don't re-ask Gemini ──
    retry_cache: dict[str, str] = {}
    if RETRY_MAP_FILE.exists():
        import json
        retry_cache = json.loads(RETRY_MAP_FILE.read_text(encoding="utf-8"))

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

    # ── Fetch from USDA (+ bounded Gemini retry on miss) ──
    rows = []
    retry_used = 0
    for i, class_name in enumerate(classes):
        # Use map entry if available, else fall back to plain name
        primary_query = usda_map.get(class_name, class_name.replace("_", " "))
        logger.info(f"[{i+1:3d}/{len(classes)}] {class_name:40s} → '{primary_query}'")

        nutrients = search_usda(primary_query, api_key)
        source = "usda"

        if not nutrients and use_retry:
            # Reuse a cached retry term if we already asked Gemini for
            # this class in a previous run — keeps Gemini calls bounded
            # even across re-runs of this script.
            alt_query = retry_cache.get(class_name)
            if not alt_query:
                alt_query = get_alternate_term(class_name, primary_query)
                if alt_query:
                    retry_cache[class_name] = alt_query

            if alt_query:
                logger.info(f"  ↻ retry with '{alt_query}'")
                nutrients = search_usda(alt_query, api_key)
                if nutrients:
                    source = "usda_retry"

        if nutrients:
            logger.info(f"  ✓ {nutrients.get('usda_description', '')[:55]}")
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

    # Persist any new retry terms so future re-runs don't re-call Gemini
    if retry_cache:
        import json
        RETRY_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
        RETRY_MAP_FILE.write_text(
            json.dumps(dict(sorted(retry_cache.items())), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    df = pd.DataFrame(rows)

    # Warn on zero-calorie rows (shouldn't happen now — search_usda already
    # rejects 0-calorie matches — but keep as a safety net)
    zero_cal = df[df["calories_per_100g"] == 0]
    if not zero_cal.empty:
        logger.warning(f"Zero calories — check these: {list(zero_cal['food_name'])}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    usda_count       = (df["source"] == "usda").sum()
    usda_retry_count = (df["source"] == "usda_retry").sum()
    fallback_count    = (df["source"] == "fallback").sum()
    logger.info(f"\nSaved {len(df)} rows → {OUTPUT_CSV}")
    logger.info(
        f"USDA (direct): {usda_count} | "
        f"USDA (after Gemini retry): {usda_retry_count} | "
        f"Fallback: {fallback_count}"
    )

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
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable the bounded Gemini retry on USDA miss (faster, no extra Gemini calls)",
    )
    args = parser.parse_args()

    if args.key == "DEMO_KEY":
        logger.warning(
            "Using DEMO_KEY — 30 req/hr limit.\n"
            "Get a free key: https://fdc.nal.usda.gov/api-guide.html\n"
            "Set in .env: USDA_API_KEY=your_key"
        )

    fetch_all(api_key=args.key, dry_run=args.dry_run, use_retry=not args.no_retry)