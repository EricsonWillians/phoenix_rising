"""
Phoenix Rising SQLAlchemy Models.

This module defines the SQLAlchemy ORM models for the Phoenix Rising application,
representing the database schema and relationships.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Enum as SQLEnum,
    func,
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base

from src.enums import EmotionState  # Importing from enums.py

Base = declarative_base()


class JournalEntry(Base):
    """Model for storing spiritual journey entries."""
    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    content: Mapped[str] = mapped_column(String, nullable=False)
    emotion: Mapped[EmotionState] = mapped_column(SQLEnum(EmotionState), nullable=False)
    light_token: Mapped[str] = mapped_column(String, nullable=False)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    using_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Define relationship with EmotionalInsight
    emotional_insights: Mapped[List["EmotionalInsight"]] = relationship(
        "EmotionalInsight",
        back_populates="journal_entry",
        cascade="all, delete-orphan"
    )


class EmotionalInsight(Base):
    """Model for tracking emotional progression and insights."""
    __tablename__ = "emotional_insights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    journal_entry_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("journal_entries.id", ondelete="CASCADE"),
        nullable=False
    )
    insight_type: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Define relationship with JournalEntry
    journal_entry: Mapped["JournalEntry"] = relationship(
        "JournalEntry",
        back_populates="emotional_insights"
    )
