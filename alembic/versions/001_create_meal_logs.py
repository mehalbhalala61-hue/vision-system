"""create meal_logs table

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "meal_logs",
        sa.Column("id",           sa.Integer(),     nullable=False, autoincrement=True),
        sa.Column("user_id",      sa.String(64),    nullable=False),
        sa.Column("food_name",    sa.String(128),   nullable=False),
        sa.Column("confidence",   sa.Float(),       nullable=False),
        sa.Column("serving_g",    sa.Float(),       nullable=True),
        sa.Column("calories",     sa.Float(),       nullable=True),
        sa.Column("protein_g",    sa.Float(),       nullable=True),
        sa.Column("carbs_g",      sa.Float(),       nullable=True),
        sa.Column("fat_g",        sa.Float(),       nullable=True),
        sa.Column("fiber_g",      sa.Float(),       nullable=True),
        sa.Column("meal_type",    sa.String(16),    nullable=True),
        sa.Column("image_path",   sa.String(256),   nullable=True),
        sa.Column("gradcam_path", sa.String(256),   nullable=True),
        sa.Column("notes",        sa.Text(),        nullable=True),
        sa.Column("logged_at",    sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_meal_logs_user_date", "meal_logs", ["user_id", "logged_at"])
    op.create_index("ix_meal_logs_id",        "meal_logs", ["id"])


def downgrade() -> None:
    op.drop_index("ix_meal_logs_user_date", table_name="meal_logs")
    op.drop_index("ix_meal_logs_id",        table_name="meal_logs")
    op.drop_table("meal_logs")