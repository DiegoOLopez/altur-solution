"""add_trained_models_table

Revision ID: f77cc9863393
Revises: cb29e16d4e75
Create Date: 2026-09-12 12:23:50.227758

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'f77cc9863393'
down_revision: Union[str, Sequence[str], None] = 'cb29e16d4e75'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('trained_models',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('ref', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('storage_key', sa.String(length=500), nullable=False),
        sa.Column('n_samples_human', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('n_samples_synthetic', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('val_accuracy', sa.Float(), nullable=True),
        sa.Column('val_auc', sa.Float(), nullable=True),
        sa.Column('eta', sa.Float(), nullable=True),
        sa.Column('train_method', sa.String(length=50), nullable=False, server_default='online_update'),
        sa.Column('train_report', sa.Text(), nullable=True),
        sa.Column('status', sa.Enum('ELIMINADO', 'DISPONIBLE', 'TRAINING', name='modelstatus'), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_trained_models_id'), 'trained_models', ['id'], unique=False)
    op.create_index(op.f('ix_trained_models_ref'), 'trained_models', ['ref'], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_trained_models_ref'), table_name='trained_models')
    op.drop_index(op.f('ix_trained_models_id'), table_name='trained_models')
    op.drop_table('trained_models')
