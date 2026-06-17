# =============================================================================
# utils/logger.py — CSV + TensorBoard Training Logger
# =============================================================================
import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Logs training metrics to both CSV and TensorBoard each epoch.

    Usage:
        tlogger = TrainingLogger(csv_path, tensorboard_dir)
        tlogger.log(epoch=1, train_loss=0.8, train_acc=0.62,
                    val_loss=0.7, val_acc=0.68, lr=3e-4)
        tlogger.close()
    """

    COLUMNS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]

    def __init__(self, csv_path: str, tensorboard_dir: Optional[str] = None):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Write CSV header
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()

        # TensorBoard (optional — skip gracefully if not installed)
        self.writer = None
        if tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=tensorboard_dir)
                logger.info(f"TensorBoard → {tensorboard_dir}")
            except ImportError:
                logger.warning("tensorboard not installed — CSV only. pip install tensorboard")

        logger.info(f"CSV log → {self.csv_path}")

    def log(
        self,
        epoch:      int,
        train_loss: float,
        train_acc:  float,
        val_loss:   float,
        val_acc:    float,
        lr:         float,
    ) -> None:
        # CSV
        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "lr":         f"{lr:.2e}",
        }
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writerow(row)

        # TensorBoard
        if self.writer:
            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
            self.writer.add_scalar("LR", lr, epoch)

    def close(self) -> None:
        if self.writer:
            self.writer.close()
