# db/models/__init__.py
# Import all models here so Alembic auto-detects them during migrations
from db.models.meal_log import MealLog  # noqa: F401

__all__ = ["MealLog"]