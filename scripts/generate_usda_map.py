# =============================================================================
# scripts/generate_usda_map.py — One-Time USDA Search Map Generator
# =============================================================================
# WORKFLOW (Industry Pattern: Bootstrap & Cache):
#
#   Step 1 — Run THIS script ONCE per dataset:
#       python scripts/generate_usda_map.py
#       → Calls Gemini AI to get USDA search terms for all classes
#       → Saves result to data/usda_map.json  (permanent local cache)
#
#   Step 2 — Run fetch_nutrition.py (reads JSON, never calls Gemini again):
#       python scripts/fetch_nutrition.py
#       → Loads data/usda_map.json locally
#       → Uses terms to query USDA API
#       → Saves data/nutrition.csv
#
# Why this pattern?
#   - Gemini is called ONCE regardless of how many times you re-run fetch
#   - Works for ANY dataset size (80 classes → 500 classes → zero code change)
#   - usda_map.json is committed to Git → teammates never need to run this
#   - Interview defensible: "AI for translation only, USDA for actual data"
#
# Requires: GEMINI_API_KEY in .env
# =============================================================================

import os
import json
import time
import logging
import argparse
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
CLASSES_FILE  = Path("data/classes.txt")
USDA_MAP_FILE = Path("data/usda_map.json")

# -----------------------------------------------------------------------------
# GEMINI PROMPT
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a food and nutrition expert. Your task is to map
Indian food dish names to the best English search query for the
USDA FoodData Central database (fdc.nal.usda.gov).

Rules:
1. Search terms must be plain English ingredient descriptions
2. Keep each term under 6 words
3. Focus on main ingredients (e.g. "potato cauliflower curry" not "aloo gobi")
4. If the dish is too regional for USDA, describe its closest equivalent
5. Return ONLY valid JSON — no markdown, no explanation, no backticks

Output format (strict JSON object):
{
  "class_name": "usda search term",
  "class_name2": "usda search term 2"
}"""


# =============================================================================
# GEMINI BATCH CALLER
# =============================================================================

def call_gemini_batch(
    classes: list[str],
    api_key: str,
    batch_size: int = 30,
) -> dict[str, str]:
    """
    Call Gemini in batches to get USDA search terms for all classes.

    Args:
        classes    : list of class names from classes.txt
        api_key    : Gemini API key
        batch_size : classes per API call (30 is safe for token limits)

    Returns:
        dict mapping class_name → usda_search_term
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        generation_config=genai.GenerationConfig(
            temperature=0.1,          # Low temp = consistent, factual output
            response_mime_type="application/json",  # Force JSON response
        ),
        system_instruction=SYSTEM_PROMPT,
    )

    full_map: dict[str, str] = {}
    batches = [classes[i:i+batch_size] for i in range(0, len(classes), batch_size)]

    for batch_idx, batch in enumerate(batches):
        logger.info(
            f"Batch {batch_idx+1}/{len(batches)} — "
            f"classes {batch_idx*batch_size+1} to {min((batch_idx+1)*batch_size, len(classes))}"
        )

        prompt = (
            f"Map these {len(batch)} Indian food class names to USDA search terms.\n\n"
            f"Classes:\n" + "\n".join(f"- {c}" for c in batch)
        )

        # Retry with exponential backoff
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                raw_text = response.text.strip()

                # Strip markdown fences if Gemini adds them despite mime_type
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                    raw_text = raw_text.strip()

                batch_map = json.loads(raw_text)

                # Validate: all requested classes should be in response
                missing = set(batch) - set(batch_map.keys())
                if missing:
                    logger.warning(f"  Gemini missed {len(missing)} classes: {missing}")
                    # Fill missing with simple fallback
                    for m in missing:
                        batch_map[m] = m.replace("_", " ")

                full_map.update(batch_map)
                logger.info(f"  ✓ Got {len(batch_map)} terms")
                break

            except json.JSONDecodeError as e:
                logger.warning(f"  Attempt {attempt+1}/3 — JSON parse error: {e}")
                if attempt == 2:
                    logger.error("  All retries failed — using plain name fallback for batch")
                    for c in batch:
                        full_map[c] = c.replace("_", " ")
                else:
                    time.sleep(2 ** attempt)   # 1s → 2s → 4s backoff

            except Exception as e:
                logger.warning(f"  Attempt {attempt+1}/3 — API error: {e}")
                if attempt == 2:
                    logger.error("  All retries failed — using plain name fallback for batch")
                    for c in batch:
                        full_map[c] = c.replace("_", " ")
                else:
                    time.sleep(2 ** attempt)

        # Polite delay between batches — avoid rate limit
        if batch_idx < len(batches) - 1:
            time.sleep(2.0)

    return full_map


# =============================================================================
# SAVE + LOAD HELPERS
# =============================================================================

def save_usda_map(usda_map: dict[str, str], path: Path = USDA_MAP_FILE) -> None:
    """Save map to JSON — sorted keys for clean Git diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_map = dict(sorted(usda_map.items()))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted_map, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(sorted_map)} entries → {path}")


def load_usda_map(path: Path = USDA_MAP_FILE) -> dict[str, str]:
    """
    Load the cached USDA map from JSON.
    Called by fetch_nutrition.py — never calls Gemini.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run once: python scripts/generate_usda_map.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# VALIDATE — check map coverage against current classes.txt
# =============================================================================

def validate_map(usda_map: dict, classes: list[str]) -> None:
    """Log coverage stats — how many classes have a map entry."""
    covered = [c for c in classes if c in usda_map]
    missing = [c for c in classes if c not in usda_map]

    logger.info(f"Map coverage: {len(covered)}/{len(classes)} classes")

    if missing:
        logger.warning(f"Missing from map ({len(missing)}): {missing}")
        logger.warning("Re-run generate_usda_map.py or add entries manually to data/usda_map.json")
    else:
        logger.info("✓ All classes covered")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate USDA search map via Gemini (run once per dataset)"
    )
    parser.add_argument(
        "--classes",
        default=str(CLASSES_FILE),
        help="Path to classes.txt (default: data/classes.txt)",
    )
    parser.add_argument(
        "--output",
        default=str(USDA_MAP_FILE),
        help="Output JSON path (default: data/usda_map.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Classes per Gemini API call (default: 30)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only check existing JSON coverage — no Gemini call",
    )
    args = parser.parse_args()

    classes_path = Path(args.classes)
    output_path  = Path(args.output)

    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found at {classes_path}")

    classes = [c.strip() for c in classes_path.read_text().splitlines() if c.strip()]
    logger.info(f"Loaded {len(classes)} classes from {classes_path}")

    # Validate-only mode
    if args.validate_only:
        if output_path.exists():
            usda_map = load_usda_map(output_path)
            validate_map(usda_map, classes)
        else:
            logger.warning(f"No map file found at {output_path}")
        exit(0)

    # Skip if already generated and fully covers current classes
    if output_path.exists():
        existing = load_usda_map(output_path)
        missing  = [c for c in classes if c not in existing]
        if not missing:
            logger.info(
                f"✓ {output_path} already exists and covers all {len(classes)} classes.\n"
                f"  Delete it and re-run if you want to regenerate.\n"
                f"  Or use --validate-only to just check coverage."
            )
            exit(0)
        else:
            logger.info(f"Map exists but missing {len(missing)} classes — generating only missing entries")
            classes = missing   # Only generate for missing ones

    # Get Gemini API key
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "Add it to your .env file: GEMINI_API_KEY=your_key_here\n"
            "Get a free key at: aistudio.google.com"
        )

    logger.info(f"Generating USDA map for {len(classes)} classes via Gemini...")
    logger.info("This runs ONCE — result cached to data/usda_map.json forever.\n")

    # Generate
    usda_map = call_gemini_batch(classes, api_key, batch_size=args.batch_size)

    # Merge with existing if partial run
    if output_path.exists():
        existing = load_usda_map(output_path)
        existing.update(usda_map)   # new entries override old
        usda_map = existing

    # Save
    save_usda_map(usda_map, output_path)

    # Final validation
    all_classes = [c.strip() for c in classes_path.read_text().splitlines() if c.strip()]
    validate_map(usda_map, all_classes)

    logger.info(
        f"\nDone! Now run:\n"
        f"  python scripts/fetch_nutrition.py\n"
        f"  (It will use data/usda_map.json — no Gemini calls)"
    )
