# models/athlete.py
from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4

from sqlmodel import SQLModel, Field, Relationship
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .run_session import RunSession
    from .user import User


class Runner(SQLModel, table=True):
    __tablename__ = "runner"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str
    user_id: Optional[UUID] = Field(default=None, foreign_key="user.id")

    created_at: datetime = Field(default_factory=datetime.utcnow)

    runs: List["RunSession"] = Relationship(back_populates="runner")
    user: Optional["User"] = Relationship(back_populates="runners")
