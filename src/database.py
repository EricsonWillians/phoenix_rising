"""
Phoenix Rising Database Layer.

This module handles data persistence for the spiritual journey of users,
storing their experiences, emotions, and received light tokens. It uses
SQLAlchemy for robust database operations and includes migration support.
"""

from datetime import datetime
from typing import List, Optional, AsyncGenerator
import logging
from pathlib import Path

from sqlalchemy import create_engine, select, desc
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship
)
from sqlalchemy.sql import func
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass

class JournalEntry(Base):
    """
    Model for storing spiritual journey entries.
    
    This model captures the essence of each moment in the user's journey,
    including their raw emotions and the light tokens received in response.
    """
    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(nullable=False)
    emotion: Mapped[str] = mapped_column(nullable=False)
    light_token: Mapped[str] = mapped_column(nullable=False)
    sentiment_score: Mapped[float] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships for emotional progression tracking
    emotional_insights: Mapped[List["EmotionalInsight"]] = relationship(
        back_populates="journal_entry",
        cascade="all, delete-orphan"
    )

class EmotionalInsight(Base):
    """
    Model for tracking emotional progression and insights.
    
    This helps users see patterns in their emotional journey and
    track their growth over time.
    """
    __tablename__ = "emotional_insights"

    id: Mapped[int] = mapped_column(primary_key=True)
    journal_entry_id: Mapped[int] = mapped_column(
        "journal_entry_id", 
        nullable=False
    )
    insight_type: Mapped[str] = mapped_column(nullable=False)
    value: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        nullable=False
    )

    journal_entry: Mapped[JournalEntry] = relationship(
        back_populates="emotional_insights"
    )

# Pydantic models for API interaction
class JournalEntryCreate(BaseModel):
    """Schema for creating a new journal entry."""
    content: str
    emotion: str
    light_token: str
    sentiment_score: Optional[float] = None

class JournalEntryResponse(BaseModel):
    """Schema for journal entry responses."""
    id: int
    content: str
    emotion: str
    light_token: str
    sentiment_score: Optional[float]
    created_at: datetime
    
    class Config:
        """Pydantic configuration."""
        from_attributes = True

class DatabaseManager:
    """
    Manages database connections and operations.
    
    This class handles the lifecycle of database connections and provides
    an interface for database operations while ensuring proper resource
    management.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        echo: bool = False
    ) -> None:
        """
        Initialize the database manager.
        
        Args:
            database_url: Optional database URL
            echo: Whether to echo SQL statements
        """
        self.database_url = (
            database_url or 
            "sqlite+aiosqlite:///./phoenix.db"
        )
        self.engine = create_async_engine(
            self.database_url,
            echo=echo,
            pool_pre_ping=True
        )
        self.async_session = async_sessionmaker(
            self.engine,
            expire_on_commit=False
        )

    async def create_tables(self) -> None:
        """Create all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session.
        
        Yields:
            AsyncSession for database operations
        """
        async with self.async_session() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()

    async def create_journal_entry(
        self,
        entry: JournalEntryCreate,
        session: AsyncSession
    ) -> JournalEntry:
        """
        Create a new journal entry.
        
        Args:
            entry: Entry data
            session: Database session
            
        Returns:
            Created JournalEntry
        """
        db_entry = JournalEntry(
            content=entry.content,
            emotion=entry.emotion,
            light_token=entry.light_token,
            sentiment_score=entry.sentiment_score
        )
        session.add(db_entry)
        await session.commit()
        await session.refresh(db_entry)
        return db_entry

    async def get_recent_entries(
        self,
        session: AsyncSession,
        limit: int = 10
    ) -> List[JournalEntry]:
        """
        Get recent journal entries.
        
        Args:
            session: Database session
            limit: Maximum number of entries to return
            
        Returns:
            List of recent JournalEntry objects
        """
        query = select(JournalEntry).order_by(
            desc(JournalEntry.created_at)
        ).limit(limit)
        result = await session.execute(query)
        return list(result.scalars().all())

    async def get_emotional_progression(
        self,
        session: AsyncSession,
        days: int = 30
    ) -> List[dict]:
        """
        Get emotional progression over time.
        
        Args:
            session: Database session
            days: Number of days to analyze
            
        Returns:
            List of emotional progression data points
        """
        query = select(
            JournalEntry.created_at,
            JournalEntry.emotion,
            JournalEntry.sentiment_score
        ).where(
            JournalEntry.created_at >= func.date('now', f'-{days} days')
        ).order_by(JournalEntry.created_at)
        
        result = await session.execute(query)
        return [
            {
                "date": row.created_at,
                "emotion": row.emotion,
                "sentiment": row.sentiment_score
            }
            for row in result.all()
        ]

    async def close(self) -> None:
        """Close database connections."""
        await self.engine.dispose()

# Database instance for application use
database = DatabaseManager()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI/Streamlit to get database sessions.
    
    Yields:
        AsyncSession for database operations
    """
    async for session in database.get_session():
        yield session