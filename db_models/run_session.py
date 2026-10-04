# models/run_session.py
from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4

from sqlmodel import SQLModel, Field, Relationship


class RunSession(SQLModel, table=True):
    __tablename__ = "run_session"

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    runner_id: UUID = Field(foreign_key="runner.id")
    runner: Optional["Runner"] = Relationship(back_populates="runs")

    date: datetime = Field(default_factory=datetime.utcnow)
    fps: int = Field(default=60)
    camera_count: int = Field(default=5)
    note: str = Field(default="")
    is_long_jump: bool = Field(
        default=False,
        description="Whether this run session is a long-jump trial (enables long_jump_final_landing in the pipeline config)"
    )

    status: str = Field(
        default="pending",
        description="pending | processing | done | failed"
    )

    progress: int = Field(
        default=0,
        description="Analysis progress (0-100)"
    )

    result_dir: Optional[str] = Field(
        description="Path to folder containing analysis results"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)

    videos: List["Video"] = Relationship(back_populates="run_session")
    analysis: Optional["AnalysisMeta"] = Relationship(back_populates="run_session")
    analysis_runs: List["AnalysisRun"] = Relationship(back_populates="run_session")
