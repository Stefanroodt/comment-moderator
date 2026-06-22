"""
Pydantic models for request validation and response serialisation.
All fields are documented so the API is self-describing in the OpenAPI UI.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, Optional
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
    # Admin override fields
    admin_overridden: bool = False
    admin_decision: Optional[ModerationDecision] = None
    admin_note: Optional[str] = None
    admin_timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Stats response
# ---------------------------------------------------------------------------

class DecisionBreakdown(BaseModel):
    approved: int
    rejected: int
    flagged_for_review: int


class DecisionPercentages(BaseModel):
    approved: float = Field(..., description="Percentage of comments approved (0–100).")
    rejected: float = Field(..., description="Percentage of comments rejected (0–100).")
    flagged_for_review: float = Field(..., description="Percentage flagged for human review (0–100).")


class AppealStats(BaseModel):
    total: int = Field(..., description="Total number of appeals submitted.")
    overturned: int = Field(..., description="Appeals where the original rejection was reversed.")
    upheld: int = Field(..., description="Appeals where the original rejection was confirmed.")
    overturn_rate: float = Field(..., description="Fraction of appeals that overturned the original decision (0–1).")


class ModerationStats(BaseModel):
    total_comments: int = Field(..., description="Total comments submitted for moderation.")
    decisions: DecisionBreakdown
    decision_percentages: DecisionPercentages
    avg_confidence: Optional[float] = Field(
        None,
        description="Average AI confidence score across all decisions (0–1). Null when no data.",
    )
    top_rejection_categories: Dict[str, int] = Field(
        ...,
        description="Top 5 rejection category counts, sorted by frequency descending. Excludes 'none'.",
    )
    appeals: AppealStats
    admin_overrides: int = Field(..., description="Number of decisions manually overridden by an admin.")


# ---------------------------------------------------------------------------
# Admin override request
# ---------------------------------------------------------------------------

class AdminOverrideRequest(BaseModel):
    decision: ModerationDecision = Field(
        ...,
        description="The corrected decision to apply.",
    )
    note: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional note explaining the override (e.g. 'Approved — verified professional credentials').",
    )
