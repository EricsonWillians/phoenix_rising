"""
Phoenix Rising Data Schemas.

This module defines the core data structures and validation rules for the
Phoenix Rising application, ensuring data integrity and type safety across
all components while preserving the spiritual essence of the sanctuary.
"""

from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, validator, ConfigDict

from src.enums import EmotionState  # Importing from enums.py


class SentimentAnalysis(BaseModel):
    """Schema for sentiment analysis results."""
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment score ranging from -1 (negative) to 1 (positive)"
    )
    is_concerning: bool = Field(
        ...,
        description="Flag indicating if the sentiment requires attention"
    )
    requires_support: bool = Field(
        ...,
        description="Flag indicating if supportive intervention is recommended"
    )

    model_config = ConfigDict(
        title="Sentiment Analysis Result",
        frozen=True
    )


class LightToken(BaseModel):
    """Schema for generated light tokens."""
    content: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="The transformative wisdom generated for the user"
    )
    sentiment_context: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment context that influenced token generation"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Moment of token generation"
    )

    model_config = ConfigDict(
        title="Light Token",
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

    @validator('content')
    def validate_content(cls, v: str) -> str:
        """Ensure token content meets spiritual quality standards."""
        if len(v.split()) < 3:
            raise ValueError("Light token must contain at least three words")
        return v.strip()


class JournalEntryCreate(BaseModel):
    """Schema for creating new journal entries."""
    content: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's journal entry content"
    )
    emotion: EmotionState = Field(
        ...,
        description="The emotional state during journaling"
    )
    light_token: Optional[str] = Field(
        None,
        description="Associated light token, if generated"
    )
    sentiment_score: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Analyzed sentiment score"
    )

    model_config = ConfigDict(
        title="Journal Entry Creation",
        json_schema_extra={
            "example": {
                "content": "Found strength in the corporate shadows today",
                "emotion": "DAWN",
                "light_token": None,
                "sentiment_score": None
            }
        }
    )

    @validator('content')
    def clean_content(cls, v: str) -> str:
        """Clean and validate journal content."""
        v = v.strip()
        if not v:
            raise ValueError("Journal content cannot be empty")
        return v


class JournalEntryResponse(BaseModel):
    """Schema for journal entry responses."""
    id: int = Field(..., description="Unique identifier for the entry")
    content: str = Field(..., description="Journal entry content")
    emotion: EmotionState = Field(..., description="Recorded emotional state")
    light_token: str = Field(..., description="Generated light token")
    sentiment_score: Optional[float] = Field(
        None,
        description="Analyzed sentiment score"
    )
    created_at: datetime = Field(..., description="Entry creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(
        title="Journal Entry Response",
        from_attributes=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class EmotionalProgression(BaseModel):
    """Schema for emotional progression analytics."""
    time_period: str = Field(..., description="Analysis time period")
    entries_analyzed: int = Field(
        ...,
        description="Number of entries in analysis"
    )
    emotional_variance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Measure of emotional fluctuation"
    )
    growth_index: float = Field(
        ...,
        description="Indicator of emotional growth"
    )
    dominant_emotions: Dict[EmotionState, int] = Field(
        ...,
        description="Frequency of each emotional state"
    )
    average_sentiment: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Average sentiment over period"
    )

    model_config = ConfigDict(
        title="Emotional Progression Analytics",
        frozen=True
    )

    @validator('dominant_emotions')
    def validate_emotions(cls, v: Dict[EmotionState, int]) -> Dict[EmotionState, int]:
        """Ensure all emotion states are accounted for."""
        if set(EmotionState) != set(v.keys()):
            raise ValueError("Must include counts for all emotional states")
        return v


class UserInsight(BaseModel):
    """Schema for generated user insights."""
    insight_type: str = Field(..., description="Type of insight generated")
    description: str = Field(..., description="Detailed insight description")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance of insight to user's journey"
    )
    suggested_actions: List[str] = Field(
        ...,
        min_items=1,
        description="Suggested actions based on insight"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Insight generation timestamp"
    )

    model_config = ConfigDict(
        title="User Journey Insight",
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )

    @validator('suggested_actions')
    def validate_actions(cls, v: List[str]) -> List[str]:
        """Ensure suggested actions are meaningful."""
        if not all(len(action.split()) >= 2 for action in v):
            raise ValueError(
                "Each suggested action must contain at least two words"
            )
        return v


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    error_code: str = Field(..., description="Error identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict] = Field(
        None,
        description="Additional error context"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error occurrence timestamp"
    )

    model_config = ConfigDict(
        title="Error Response",
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
