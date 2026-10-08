# models/comparison_report.py
"""Storage for a validated contract v1 `ComparisonReport` (規劃書 Step 10).

The backend now creates the initial 2D report automatically once both sides
of a Compare pair have completed. Later pipeline stages extend the same
schema with 3D, speed and gait metrics; this table remains the stable storage
and lookup point keyed by `comparison_group_id`.
"""
from datetime import datetime
from typing import Dict
from uuid import UUID

from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel


class ComparisonReport(SQLModel, table=True):
    __tablename__ = "comparison_report"

    comparison_group_id: UUID = Field(primary_key=True)

    server_run_id: UUID = Field(foreign_key="analysis_run.id", index=True)
    local_run_id: UUID = Field(foreign_key="analysis_run.id", index=True)

    status: str = Field(description="complete | partial | not_comparable")
    input_hashes_match: bool

    payload: Dict = Field(
        sa_column=Column(JSON),
        description="The full schema-validated ComparisonReport document, stored verbatim "
        "so retrieval returns exactly what was submitted without re-deriving it.",
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
