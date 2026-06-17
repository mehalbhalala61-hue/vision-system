# =============================================================================
# db/models/meal_log.py — MealLog ORM Model
# =============================================================================
# Stores every food prediction that a user logs.
# Powers:
#   POST /log       → insert a new meal entry
#   GET  /dashboard → aggregate daily / weekly nutrition summary
#
# Interview note:
#   "I modelled meal logs with SQLAlchemy ORM — full nutrition tracking per
#    entry, indexed on user_id + logged_at for fast dashboard queries.
#    Alembic handles schema migrations so the DB evolves without data loss."
# =============================================================================

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Index,
)
from db.base import Base


class MealLog(Base):
    """
    One row = one logged meal / food prediction.

    Columns
    -------
    id              Auto-increment primary key
    user_id         Identifies the user (string — no auth yet, extensible)
    food_name       Predicted class name  e.g. "biryani"
    confidence      Model confidence 0–1  e.g. 0.94
    serving_g       Serving size in grams (from nutrition.csv)
    calories        Total calories for this serving
    protein_g       Protein  (g)
    carbs_g         Carbohydrates (g)
    fat_g           Fat (g)
    fiber_g         Dietary fiber (g)
    meal_type       breakfast | lunch | dinner | snack
    image_path      Optional — path/URL of uploaded image
    gradcam_path    Optional — path/URL of Grad-CAM heatmap
    notes           Optional free-text from Gemini AI advice
    logged_at       UTC timestamp — auto-set on insert
    """

    __tablename__ = "meal_logs"

    # ------------------------------------------------------------------
    # Primary Key
    # ------------------------------------------------------------------
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # ------------------------------------------------------------------
    # User
    # ------------------------------------------------------------------
    user_id = Column(
        String(64),
        nullable=False,
        index=True,
        comment="User identifier — plain string for now, foreign key ready",
    )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    food_name = Column(
        String(128),
        nullable=False,
        comment="Predicted food class from ResNet model",
    )
    confidence = Column(
        Float,
        nullable=False,
        comment="Model softmax confidence 0.0 – 1.0",
    )

    # ------------------------------------------------------------------
    # Nutrition (scaled to actual serving size)
    # ------------------------------------------------------------------
    serving_g = Column(
        Float,
        nullable=True,
        default=100.0,
        comment="Serving size in grams",
    )
    calories = Column(Float, nullable=True, comment="kcal for this serving")
    protein_g = Column(Float, nullable=True, comment="Protein in grams")
    carbs_g = Column(Float, nullable=True, comment="Carbohydrates in grams")
    fat_g = Column(Float, nullable=True, comment="Fat in grams")
    fiber_g = Column(Float, nullable=True, comment="Dietary fiber in grams")

    # ------------------------------------------------------------------
    # Meal context
    # ------------------------------------------------------------------
    meal_type = Column(
        String(16),
        nullable=True,
        default="lunch",
        comment="breakfast | lunch | dinner | snack",
    )

    # ------------------------------------------------------------------
    # Optional media paths
    # ------------------------------------------------------------------
    image_path = Column(
        String(256),
        nullable=True,
        comment="Uploaded image path or URL",
    )
    gradcam_path = Column(
        String(256),
        nullable=True,
        comment="Grad-CAM heatmap path or URL",
    )

    # ------------------------------------------------------------------
    # AI notes
    # ------------------------------------------------------------------
    notes = Column(
        Text,
        nullable=True,
        comment="Gemini AI health advice for this meal",
    )

    # ------------------------------------------------------------------
    # Timestamp — UTC, auto-set on insert
    # ------------------------------------------------------------------
    logged_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        comment="UTC timestamp of when this meal was logged",
    )

    # ------------------------------------------------------------------
    # Composite index — fast dashboard queries
    # "Give me all meals for user X on date Y" hits this index directly
    # ------------------------------------------------------------------
    __table_args__ = (
        Index("ix_meal_logs_user_date", "user_id", "logged_at"),
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize to plain dict — used by /dashboard endpoint."""
        return {
            "id":           self.id,
            "user_id":      self.user_id,
            "food_name":    self.food_name,
            "confidence":   round(self.confidence, 4),
            "serving_g":    self.serving_g,
            "calories":     self.calories,
            "protein_g":    self.protein_g,
            "carbs_g":      self.carbs_g,
            "fat_g":        self.fat_g,
            "fiber_g":      self.fiber_g,
            "meal_type":    self.meal_type,
            "image_path":   self.image_path,
            "gradcam_path": self.gradcam_path,
            "notes":        self.notes,
            "logged_at":    self.logged_at.isoformat() if self.logged_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"<MealLog id={self.id} user={self.user_id!r} "
            f"food={self.food_name!r} cal={self.calories} "
            f"at={self.logged_at}>"
        )