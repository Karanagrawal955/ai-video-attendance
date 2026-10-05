"""add users + password_reset_requests (multi-user auth)

Revision ID: 0004_users
Revises: 0003_add_alerts
Create Date: 2026-10-05

The super-admin account itself is NOT inserted here: it is materialised at
runtime by ``app.users.ensure_seed_superuser`` from ``ADMIN_USERNAME`` /
``ADMIN_PASSWORD``, so the hash is computed with the live settings instead of
being frozen into the schema history.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_users"
down_revision: Union[str, None] = "0003_add_alerts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("password_hash", sa.String(length=200), nullable=False),
        sa.Column(
            "password_source",
            sa.String(length=8),
            nullable=False,
            server_default="env",
        ),
        sa.Column("display_name", sa.String(length=200), nullable=True),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column(
            "status", sa.String(length=16), nullable=False, server_default="pending"
        ),
        sa.Column(
            "security_level", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approved_by", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("role IN ('super_admin', 'admin')", name="ck_users_role"),
        sa.CheckConstraint(
            "status IN ('pending', 'active', 'rejected', 'suspended')",
            name="ck_users_status",
        ),
        sa.CheckConstraint(
            "security_level >= 0 AND security_level <= 5",
            name="ck_users_security_level",
        ),
        sa.CheckConstraint("length(trim(username)) >= 3", name="ck_users_username_len"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)

    op.create_table(
        "password_reset_requests",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("reason", sa.String(length=500), nullable=True),
        sa.Column(
            "status", sa.String(length=16), nullable=False, server_default="pending"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(length=100), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'resolved', 'rejected')",
            name="ck_reset_requests_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_reset_requests_username", "password_reset_requests", ["username"]
    )
    op.create_index(
        "ix_reset_requests_status", "password_reset_requests", ["status"]
    )


def downgrade() -> None:
    op.drop_index("ix_reset_requests_status", table_name="password_reset_requests")
    op.drop_index("ix_reset_requests_username", table_name="password_reset_requests")
    op.drop_table("password_reset_requests")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
