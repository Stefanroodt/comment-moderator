"""
Pydantic models for request validation and response serialisation.
All fields are documented so the API is self-describing in the OpenAPI UI.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModerationDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED_FOR_REVIEW = "flagged_for_review"


class FinalDecision(str, Enum):
    """Appeals can only produce a binary outcome — no further escalation."""
    APPROVED = "approved"
    REJECTED = "rejected"


class RejectionCategory(str, Enum):
    SPAM = "spam"
    HATE_SPEECH = "hate_speech"
    MISINFORMATION = "misinformation"
    OFF_TOPIC = "off_topic"
    ABUSIVE = "abusive"
    PROMOTIONAL = "promotional"
    NONE = "none"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class CommentRequest(BaseModel):
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Identifier for the submitting user (used for rate limiting).",
        examples=["user_abc123"],
    )
    comment: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The comment text to moderate.",
        examples=["Has anyone had experience with HMO licensing in Manchester?"],
    )

    @field_validator("comment")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Comment must not be blank or whitespace only.")
        return stripped


class AppealRequest(BaseModel):
    comment_id: UUID = Field(
        ...,
        description="The UUID of the original rejected comment.",
    )
    appeal_context: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="The user's explanation for why the comment should be reconsidered.",
        examples=["My comment was professional advice based on 10 years as a landlord."],
    )

    @field_validator("appeal_context")
    @classmethod
    def strip_context(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Appeal context must not be blank.")
        return stripped


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class ModerationResponse(BaseModel):
    comment_id: UUID = Field(..., description="Unique ID for this moderation event.")
    decision: ModerationDecision
    confidence: float = Field(..., ge=0.0, le=1.0, description="AI confidence 0–1.")
    reasoning: str = Field(..., description="Brief explanation of the decision.")
    rejection_category: RejectionCategory = Field(
        default=RejectionCategory.NONE,
        description="Category of rejection, if applicable.",
    )
    timestamp: datetime


class AppealResponse(BaseModel):
    comment_id: UUID
    original_decision: ModerationDecision
    appeal_decision: FinalDecision
    reasoning: str = Field(..., description="Explanation of the final appeal decision.")
    timestamp: datetime


# ---------------------------------------------------------------------------
# Log entry (persisted in-memory)
# ---------------------------------------------------------------------------

class LogEntry(BaseModel):
    comment_id: UUID
    user_id: str
    comment: str
    decision: ModerationDecision
    confidence: float
    reasoning: str
    rejection_category: RejectionCategory
    timestamp: datetime
    appealed: bool = False
    appeal_context: Optional[str] = None
    appeal_decision: Optional[FinalDecision] = None
    appeal_reasoning: Optional[str] = None
    appeal_timestamp: Optional[datetime] = None
