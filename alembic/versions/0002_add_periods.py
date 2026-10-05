"""add periods + attendance_sessions.period_id

Revision ID: 0002_add_periods
Revises: 0001_initial
Create Date: 2026-09-24
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_add_periods"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "periods",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("start_time", sa.Time(), nullable=False),
        sa.Column("end_time", sa.Time(), nullable=False),
        sa.Column("days_bitmask", sa.Integer(), server_default="127", nullable=False),
        sa.Column("section", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("length(trim(name)) > 0", name="ck_periods_name_nonempty"),
        sa.CheckConstraint("start_time < end_time", name="ck_periods_time_range"),
        sa.CheckConstraint("days_bitmask >= 0 AND days_bitmask <= 127", name="ck_periods_days_range"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_periods_name"),
    )
    # batch_alter_table: plain op.create_foreign_key raises
    # "No support for ALTER of constraints" on SQLite (dev/test DBs);
    # on PostgreSQL batch mode just emits the normal ALTER statements.
    with op.batch_alter_table("attendance_sessions") as batch_op:
        batch_op.add_column(sa.Column("period_id", sa.Integer(), nullable=True))
        batch_op.create_index("ix_sessions_period_id", ["period_id"])
        batch_op.create_foreign_key(
            "fk_sessions_period_id",
            "periods",
            ["period_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("attendance_sessions") as batch_op:
        batch_op.drop_constraint("fk_sessions_period_id", type_="foreignkey")
        batch_op.drop_index("ix_sessions_period_id")
        batch_op.drop_column("period_id")
    op.drop_table("periods")
