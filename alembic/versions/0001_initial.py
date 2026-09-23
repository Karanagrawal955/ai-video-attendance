"""initial schema: students, cameras, attendance_sessions, recognition_logs

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-23

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "students",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("registration_no", sa.String(length=64), nullable=False),
        sa.Column("section", sa.String(length=64), nullable=True),
        sa.Column(
            "embeddings",
            sa.JSON(),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "photo_paths",
            sa.JSON(),
            server_default="[]",
            nullable=False,
        ),
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
        sa.CheckConstraint(
            "length(trim(name)) > 0", name="ck_students_name_nonempty"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_students_registration_no"),
        "students",
        ["registration_no"],
        unique=True,
    )

    op.create_table(
        "cameras",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("location", sa.String(length=200), nullable=True),
        sa.Column("rtsp_url", sa.String(length=500), nullable=True),
        sa.Column("file_path", sa.String(length=500), nullable=True),
        sa.Column("type", sa.String(length=16), nullable=False),
        sa.Column(
            "sampling_rate", sa.Integer(), server_default="5", nullable=False
        ),
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
        sa.CheckConstraint(
            "type IN ('entry', 'exit', 'both')", name="ck_cameras_type"
        ),
        sa.CheckConstraint(
            "sampling_rate >= 1 AND sampling_rate <= 600",
            name="ck_cameras_sampling_rate",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_cameras_name"),
    )

    op.create_table(
        "attendance_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("camera_in_id", sa.Integer(), nullable=True),
        sa.Column("camera_out_id", sa.Integer(), nullable=True),
        sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column(
            "status", sa.String(length=16), server_default="ongoing", nullable=False
        ),
        sa.Column("total_duration", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('ongoing', 'completed')", name="ck_sessions_status"
        ),
        sa.ForeignKeyConstraint(
            ["student_id"],
            ["students.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["camera_in_id"],
            ["cameras.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["camera_out_id"],
            ["cameras.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_attendance_sessions_student_id",
                    "attendance_sessions", ["student_id"])
    op.create_index("ix_attendance_sessions_date",
                    "attendance_sessions", ["date"])
    op.create_index("ix_sessions_student_date", "attendance_sessions",
                    ["student_id", "date"])
    op.create_index("ix_sessions_date_status", "attendance_sessions",
                    ["date", "status"])

    op.create_table(
        "recognition_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=True),
        sa.Column("camera_id", sa.Integer(), nullable=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("gpu_inference_time_ms", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["student_id"],
            ["students.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["camera_id"],
            ["cameras.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_recognition_logs_timestamp",
                    "recognition_logs", ["timestamp"])
    op.create_index("ix_recognition_logs_student_id",
                    "recognition_logs", ["student_id"])
    op.create_index("ix_recognition_logs_camera_id",
                    "recognition_logs", ["camera_id"])
    op.create_index("ix_logs_student_ts", "recognition_logs",
                    ["student_id", "timestamp"])
    op.create_index("ix_logs_camera_ts", "recognition_logs",
                    ["camera_id", "timestamp"])


def downgrade() -> None:
    op.drop_table("recognition_logs")
    op.drop_table("attendance_sessions")
    op.drop_table("cameras")
    op.drop_table("students")
