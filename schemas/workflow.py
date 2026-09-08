"""Public extraction contracts and private workflow state."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class OperationalTask(BaseModel):
    """A single actionable task extracted from unstructured input.

    This model is deliberately flat because it is passed directly to Gemini as
    a response_schema. Nested Pydantic models are not supported reliably there.
    """

    title: str = Field(min_length=1, description="Short imperative task title")
    description: str = Field(default="", description="Useful task details")
    priority: Literal["low", "medium", "high"] = "medium"
    owner: str | None = Field(default=None, description="Person or team responsible")
    due_date: str | None = Field(default=None, description="Due date in YYYY-MM-DD format")


class WorkflowState(BaseModel):
    """Internal state for one orchestration run; never pass as response_schema."""

    source_text: str
    tasks: list[OperationalTask] = Field(default_factory=list)
    completed_task_titles: list[str] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed"] = "pending"


class ExtractionResult(BaseModel):
    """Flat wrapper used for structured extraction responses."""

    tasks: list[OperationalTask] = Field(default_factory=list)
