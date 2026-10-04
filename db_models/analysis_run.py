# models/analysis_run.py
"""Per-compute-location analysis run record (規劃書 Step 9, runner-pose-ondevice/report/integration_guide.md §13).

A `RunSession` can have more than one `AnalysisRun`: exactly one for the
existing Server path, and — once Local/Compare ship — an additional one for
a Local (on-device) run. Two runs sharing a `comparison_group_id` are a
Compare pair and must never overwrite each other's results.

This table is purely additive: `RunSession`'s existing `analysis` relationship
(-> `AnalysisMeta`, the summary numbers used by today's result pages) and
every existing query against `RunSession` are untouched. Sessions created
before this table existed have no `AnalysisRun` row at all; `synthesize_legacy_server_run`
below provides a read-time, non-persisted stand-in for those so callers can
treat old and new sessions uniformly without a destructive backfill of the
live database.
"""
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import JSON
from sqlmodel import Column, Field, Relationship, SQLModel

if TYPE_CHECKING:
    from .run_session import RunSession


class AnalysisRun(SQLModel, table=True):
    __tablename__ = "analysis_run"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    run_session_id: UUID = Field(foreign_key="run_session.id", index=True)
    run_session: Optional["RunSession"] = Relationship(back_populates="analysis_runs")

    compute_location: str = Field(
        description="server | local — matches contract v1 manifest.compute_location"
    )
    comparison_group_id: Optional[UUID] = Field(
        default=None,
        index=True,
        description="Shared by the Server/Local pair of a Compare run; null outside Compare",
    )

    schema_version: str = Field(default="1.0.0")
    engine_version: str = Field(default="unknown")

    status: str = Field(
        default="pending",
        description="pending | processing | completed | failed",
    )

    timings: Optional[Dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Optional per-stage timing breakdown, contract v1 'stages' shape",
    )

    result_summary: Optional[Dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description=(
            "This run's own copy of the manifest 'summary' block (Step 10 ingestion). "
            "Lives here rather than on AnalysisMeta so a Compare pair's two runs never "
            "overwrite each other: AnalysisMeta stays the single-display-row table and "
            "is only mirrored from here for a non-Compare run (comparison_group_id is "
            "None) on its own RunSession, where there is no ambiguity about ownership."
        ),
    )

    idempotency_key: Optional[str] = Field(
        default=None,
        index=True,
        description="Client-supplied key for the manifest-ingestion request; a repeat "
        "with the same key returns the stored result instead of reprocessing.",
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)


def synthesize_legacy_server_run(run_session: "RunSession") -> Optional[AnalysisRun]:
    """Return a transient (never persisted) `AnalysisRun` standing in for a
    pre-existing Server session that has no real `AnalysisRun` row.

    Only synthesizes when `run_session.analysis_runs` is empty — a session
    with any real row (Server, Local, or a Compare pair) is returned as-is
    by the caller instead, so this never shadows real data. Returns `None`
    for sessions that were never analyzed (no `AnalysisMeta`), since there is
    nothing truthful to synthesize for those.
    """
    if run_session.analysis_runs:
        return None
    if run_session.analysis is None:
        return None
    return AnalysisRun(
        id=run_session.id,
        run_session_id=run_session.id,
        compute_location="server",
        comparison_group_id=None,
        schema_version="1.0.0",
        engine_version="pre-analysis-run-tracking",
        status="completed" if run_session.status == "done" else run_session.status,
        timings=None,
        created_at=run_session.created_at,
        completed_at=None,
    )
