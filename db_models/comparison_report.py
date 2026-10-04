# models/comparison_report.py
"""Storage for a validated contract v1 `ComparisonReport` (規劃書 Step 10).

Computing the actual metric/performance differences is out of scope here —
that needs real Local 2D/3D output to diff against Server output, which
depends on work (Steps 6-8, 15-18) that isn't done yet. This model and the
ingestion endpoint only provide schema-validated storage and lookup by
`comparison_group_id`, so whichever side first has both runs' numbers can
submit a report and either side can retrieve it afterward.
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
