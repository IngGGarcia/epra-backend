"""
Evaluation Model
"""

from datetime import datetime
from uuid import uuid4

from sqlmodel import Field, SQLModel


class EvaluationBase(SQLModel):
    user_id: int = Field(default=None)
    eeg_file_path: str = Field(default=None)
    sam_file_path: str = Field(default=None)


class Evaluation(EvaluationBase, table=True):
    __tablename__ = "evaluation"
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class EvaluationCreate(EvaluationBase):
    pass


class EvaluationRead(EvaluationBase):
    id: str = Field(default=None, primary_key=True)


EVALUATION_MIGRATION = """
CREATE TABLE evaluation (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    eeg_file_path TEXT,
    sam_file_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_evaluation_user_id ON evaluation(user_id);
"""
