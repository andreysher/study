"""create main tables

Revision ID: fa7fc7c07147
Revises: 
Create Date: 2018-11-15 02:04:51.346266

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'fa7fc7c07147'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'category',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String(50), nullable=False),
    )

    op.create_table(
        'operation',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('amount', sa.Integer, nullable=False),
        sa.Column('comment', sa.Unicode(200)),
        sa.Column('category_id', sa.Integer, sa.ForeignKey('category.id')),
    )

    op.create_table(
        'user_operation',
        sa.Column('user_id', sa.Integer),
        sa.Column('operation_id', sa.Integer, sa.ForeignKey('operation.id'), nullable=False),
        sa.Column('time', sa.DateTime)
    )


def downgrade():
    op.drop_table('operation')
    op.drop_table('category')
    op.drop_table('user_operation')
