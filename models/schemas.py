"""Pydantic schemas for structured LLM outputs."""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, field_validator


class SubTask(BaseModel):
    """A single sub-task in the research plan."""
    
    description: str = Field(
        description="Detailed description of what this sub-task should accomplish"
    )
    priority: str = Field(
        description="Priority level: high, medium, or low"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of sub-task descriptions that must be completed before this one"
    )
    estimated_complexity: str = Field(
        description="Complexity estimate: simple, moderate, or complex"
    )


class ResearchPlan(BaseModel):
    """Structured research plan with task decomposition."""
    
    main_goal: str = Field(
        description="The primary research goal derived from the topic"
    )
    sub_tasks: List[SubTask] = Field(
        description="List of sub-tasks to accomplish the research goal",
        min_length=3  # Enforce minimum sub-task count
    )
    total_estimated_steps: int = Field(
        description="Total number of estimated steps across all sub-tasks"
    )
    
    @field_validator('sub_tasks')
    @classmethod
    def validate_sub_task_count(cls, v: List[SubTask]) -> List[SubTask]:
        """Ensure at least 3 sub-tasks are generated."""
        if len(v) < 3:
            raise ValueError("Must generate at least 3 sub-tasks for comprehensive research")
        return v
    
    @field_validator('total_estimated_steps')
    @classmethod
    def validate_step_count(cls, v: int, info) -> int:
        """Ensure step count is reasonable for the number of sub-tasks."""
        sub_task_count = len(info.data.get('sub_tasks', []))
        if v < sub_task_count:
            raise ValueError("Total steps must be at least equal to number of sub-tasks")
        return v
