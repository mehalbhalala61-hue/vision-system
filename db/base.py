# =============================================================================
# db/base.py — SQLAlchemy Declarative Base
# =============================================================================
# All ORM models must inherit from this Base.
# Import Base here, never create a second one — that causes table conflicts.
# =============================================================================

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Single source of truth for all SQLAlchemy ORM models.

    Usage:
        from db.base import Base

        class MyModel(Base):
            __tablename__ = "my_table"
            ...
    """
    pass