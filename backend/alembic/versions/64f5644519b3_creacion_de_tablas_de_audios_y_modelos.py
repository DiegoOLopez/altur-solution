"""Creacion de tablas de audios y modelos

Revision ID: 64f5644519b3
Revises:
Create Date: 2026-09-12 04:16:29.982483
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "64f5644519b3"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "audios",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "ref",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "title",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "storage_key",
            sa.String(length=500),
            nullable=False,
        ),
        sa.Column(
            "duration",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "sample_rate",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "channels",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "is_synthetic",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "confidence",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "score_total",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "REVISADO",
                "NO_REVISADO",
                "DELETED",
                name="audiostatus",
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ref"),
    )

    op.create_index(
        op.f("ix_audios_id"),
        "audios",
        ["id"],
        unique=False,
    )

    op.create_table(
        "modelos",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "nombre",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "estado",
            sa.Enum(
                "ELIMINADO",
                "DISPONIBLE",
                "TRAINING",
                name="modelstatus",
            ),
            nullable=False,
        ),
        sa.Column(
            "referencia",
            sa.String(length=255),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_modelos_id"),
        "modelos",
        ["id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_modelos_id"),
        table_name="modelos",
    )
    op.drop_table("modelos")

    op.drop_index(
        op.f("ix_audios_id"),
        table_name="audios",
    )
    op.drop_table("audios")