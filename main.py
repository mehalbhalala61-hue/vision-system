# =============================================================================
# main.py — CLI Entry Point
# =============================================================================
# Run from project root:
#   python main.py api      → start FastAPI server
#   python main.py train    → start training
#   python main.py export   → export to ONNX
#   python main.py db-init  → create DB tables
# =============================================================================

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

COMMANDS = {
    "api":     "Start FastAPI server (uvicorn)",
    "train":   "Train ResNet model",
    "export":  "Export model to ONNX",
    "db-init": "Create DB tables (dev only)",
}


def run_api():
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting API on port {port}...")
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)


def run_train():
    import argparse
    from training.train import train
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lr-finder", action="store_true")
    parser.add_argument("--resume",         action="store_true")
    parser.add_argument("--overfit-test",   action="store_true")
    args = parser.parse_args(sys.argv[2:])
    train(args)


def run_export():
    import argparse
    from models.export import run_export
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     default="logs/checkpoints/best.pth")
    parser.add_argument("--output",         default=None)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--n-runs",         type=int, default=50)
    args = parser.parse_args(sys.argv[2:])
    run_export(args)


def run_db_init():
    from db.session import init_db
    init_db()
    logger.info("DB tables created.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python main.py <command>\n")
        print("Commands:")
        for cmd, desc in COMMANDS.items():
            print(f"  {cmd:<12} {desc}")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "api":      run_api()
    elif cmd == "train":  run_train()
    elif cmd == "export": run_export()
    elif cmd == "db-init": run_db_init()
