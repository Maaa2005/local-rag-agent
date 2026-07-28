from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Status(str, Enum):
    NEEDS_CLARIFICATION = "needs_clarification"
    READY = "ready"
    REJECTED = "rejected"


class IntentCategory(str, Enum):
    HR_POLICY = "hr_policy"
    EXPENSE = "expense"
    ATTENDANCE = "attendance"
    PAYROLL = "payroll"
    DATA_PROCESSING = "data_processing"
    GENERAL_DOCUMENT = "general_document"
    UNKNOWN = "unknown"


class Action(str, Enum):
    SEARCH_POLICY = "search_policy"
    PROCESS_FILE = "process_file"
    REQUEST_APPROVAL = "request_approval"
    CREATE_TICKET = "create_ticket"
    ANSWER = "answer"
    ESCALATE = "escalate"


class Intent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: IntentCategory
    summary: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class ClarificationQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(min_length=1)
    question: str = Field(min_length=1)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int = Field(ge=1)
    action: Action
    parameters: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    access_level: str = Field(min_length=1)
    document_type: str | None = None
    is_active: Literal[True] = True


class SearchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)
    filters: SearchFilters


class OrchestratorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    status: Status
    intent: Intent
    missing_fields: list[str] = Field(default_factory=list)
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    execution_plan: list[PlanStep] = Field(default_factory=list)
    search: SearchSpec | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "OrchestratorResponse":
        if self.status is Status.NEEDS_CLARIFICATION:
            if not self.missing_fields or not self.questions:
                raise ValueError("clarification requires missing_fields and questions")
            if self.execution_plan or self.search is not None:
                raise ValueError("tools must not be planned before clarification")
        if self.status is Status.READY:
            if self.missing_fields or self.questions or not self.execution_plan:
                raise ValueError("ready requires an executable plan and no questions")
        return self


class ConversationState(BaseModel):
    """Serializable state passed between turns."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    category: IntentCategory | None = None
    original_request: str = ""
    slots: dict[str, str] = Field(default_factory=dict)
    pending_fields: list[str] = Field(default_factory=list)
    turn_count: int = Field(default=0, ge=0, le=20)
    access_level: str = "employee"


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response: OrchestratorResponse
    state: ConversationState
