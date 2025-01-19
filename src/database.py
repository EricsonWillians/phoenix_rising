"""
Phoenix Rising Database Layer.

This module manages data persistence for the spiritual journey of users,
handling database connections and operations using SQLAlchemy and Pydantic schemas.
"""

from typing import List, Optional, AsyncGenerator, Dict, Any

import logging

from sqlalchemy import select, desc, func
from sqlalchemy.exc import SQLAlchemyError, OperationalError

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)

from src.models import Base, JournalEntry  # Importing from models.py
from src.schemas import (
    JournalEntryCreate,
    JournalEntryResponse
)  # Importing Pydantic models from schemas.py

from src.enums import EmotionState  # Importing EmotionState from enums.py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class SQLAlchemyDBError(DatabaseError):
    """Exception for SQLAlchemy database errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class DatabaseManager:
    """
    Manages database connections and operations.

    This class handles the lifecycle of database connections and provides
    an interface for database operations while ensuring proper resource
    management.
    """

    _instance = None  # Class variable for Singleton instance

    def __new__(cls, *args, **kwargs):
        """Implement Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

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
        if hasattr(self, 'initialized') and self.initialized:
            return  # Avoid re-initialization

        self.database_url = (
            database_url or
            "sqlite+aiosqlite:///./phoenix.db"
        )
        self.engine = create_async_engine(
            self.database_url,
            echo=echo,
            pool_pre_ping=True,
            future=True
        )
        self.async_session = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession
        )
        logger.info("DatabaseManager initialized with URL: %s", self.database_url)
        self.initialized = True  # Flag to prevent re-initialization

    async def initialize_database(self) -> None:
        """
        Initialize the database by creating all tables.

        This should be called at the application startup.
        """
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully.")
        except SQLAlchemyError as e:
            logger.error("Error creating database tables: %s", e)
            raise SQLAlchemyDBError("Failed to create database tables.", {"error": str(e)})

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session.

        Yields:
            AsyncSession for database operations
        """
        async with self.async_session() as session:
            try:
                yield session
            except SQLAlchemyError as e:
                await session.rollback()
                logger.error("Database session error: %s", e)
                raise SQLAlchemyDBError("Database session failed.", {"error": str(e)})
            finally:
                await session.close()

    async def create_journal_entry(
        self,
        entry: JournalEntryCreate
    ) -> JournalEntryResponse:
        """
        Create a new journal entry.

        Args:
            entry: Entry data

        Returns:
            JournalEntryResponse containing the created entry details
        """
        try:
            async with self.async_session() as session:
                db_entry = JournalEntry(
                    content=entry.content,
                    emotion=entry.emotion,
                    light_token=entry.light_token or "",  # Handle Optional
                    sentiment_score=entry.sentiment_score
                )
                session.add(db_entry)
                await session.commit()
                await session.refresh(db_entry)
                logger.info("Journal entry created with ID: %s", db_entry.id)
                return JournalEntryResponse.from_orm(db_entry)
        except OperationalError as oe:
            logger.warning("OperationalError encountered: %s. Attempting to reinitialize the database.", oe)
            await self.initialize_database()  # Attempt to recreate tables
            # Retry the operation once after reinitialization
            try:
                async with self.async_session() as session:
                    db_entry = JournalEntry(
                        content=entry.content,
                        emotion=entry.emotion,
                        light_token=entry.light_token or "",  # Handle Optional
                        sentiment_score=entry.sentiment_score
                    )
                    session.add(db_entry)
                    await session.commit()
                    await session.refresh(db_entry)
                    logger.info("Journal entry created with ID: %s", db_entry.id)
                    return JournalEntryResponse.from_orm(db_entry)
            except SQLAlchemyError as e:
                logger.error("Database error after reinitialization: %s", e)
                raise SQLAlchemyDBError("Failed to create journal entry after reinitialization.", {"error": str(e)})
        except SQLAlchemyError as e:
            logger.error("Database error while creating journal entry: %s", e)
            raise SQLAlchemyDBError("Failed to create journal entry.", {"error": str(e)})

    async def get_recent_entries(
        self,
        limit: int = 10
    ) -> List[JournalEntryResponse]:
        """
        Get recent journal entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of JournalEntryResponse objects
        """
        try:
            async with self.async_session() as session:
                query = select(JournalEntry).order_by(
                    desc(JournalEntry.created_at)
                ).limit(limit)
                result = await session.execute(query)
                entries = result.scalars().all()
                logger.info("Retrieved %d recent journal entries.", len(entries))
                return [JournalEntryResponse.from_orm(entry) for entry in entries]
        except OperationalError as oe:
            logger.warning("OperationalError encountered while fetching recent entries: %s. Attempting to reinitialize the database.", oe)
            await self.initialize_database()  # Attempt to recreate tables
            # Retry the operation once after reinitialization
            try:
                async with self.async_session() as session:
                    query = select(JournalEntry).order_by(
                        desc(JournalEntry.created_at)
                    ).limit(limit)
                    result = await session.execute(query)
                    entries = result.scalars().all()
                    logger.info("Retrieved %d recent journal entries after reinitialization.", len(entries))
                    return [JournalEntryResponse.from_orm(entry) for entry in entries]
            except SQLAlchemyError as e:
                logger.error("Database error after reinitialization while fetching recent entries: %s", e)
                raise SQLAlchemyDBError("Failed to fetch recent journal entries after reinitialization.", {"error": str(e)})
        except SQLAlchemyError as e:
            logger.error("Database error while fetching recent entries: %s", e)
            raise SQLAlchemyDBError("Failed to fetch recent journal entries.", {"error": str(e)})

    async def add_journal_entry(
        self,
        content: str,
        token: str,
        emotion: EmotionState,
        using_fallback: bool = False,
        sentiment_score: Optional[float] = None
    ) -> JournalEntryResponse:
        """
        Add a new journal entry to the database.

        Args:
            content: The content of the journal entry
            token: The light token associated with the entry
            emotion: The emotion associated with the entry
            using_fallback: Indicates if a fallback was used
            sentiment_score: Sentiment analysis score

        Returns:
            JournalEntryResponse containing the created entry details
        """
        entry = JournalEntryCreate(
            content=content,
            emotion=emotion,
            light_token=token,
            sentiment_score=sentiment_score
        )
        try:
            async with self.async_session() as session:
                db_entry = JournalEntry(
                    content=entry.content,
                    emotion=entry.emotion,
                    light_token=entry.light_token or "",  # Handle Optional
                    sentiment_score=entry.sentiment_score,
                    using_fallback=using_fallback
                )
                session.add(db_entry)
                await session.commit()
                await session.refresh(db_entry)
                logger.info("Journal entry added with ID: %s", db_entry.id)
                return JournalEntryResponse.from_orm(db_entry)
        except OperationalError as oe:
            logger.warning("OperationalError encountered while adding journal entry: %s. Attempting to reinitialize the database.", oe)
            await self.initialize_database()  # Attempt to recreate tables
            # Retry the operation once after reinitialization
            try:
                async with self.async_session() as session:
                    db_entry = JournalEntry(
                        content=entry.content,
                        emotion=entry.emotion,
                        light_token=entry.light_token or "",  # Handle Optional
                        sentiment_score=entry.sentiment_score,
                        using_fallback=using_fallback
                    )
                    session.add(db_entry)
                    await session.commit()
                    await session.refresh(db_entry)
                    logger.info("Journal entry added with ID: %s after reinitialization.", db_entry.id)
                    return JournalEntryResponse.from_orm(db_entry)
            except SQLAlchemyError as e:
                logger.error("Database error after reinitialization while adding journal entry: %s", e)
                raise SQLAlchemyDBError("Failed to add journal entry after reinitialization.", {"error": str(e)})
        except SQLAlchemyError as e:
            logger.error("Database error while adding journal entry: %s", e)
            raise SQLAlchemyDBError("Failed to add journal entry.", {"error": str(e)})

    async def get_emotional_progression(
        self,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get emotional progression over time.

        Args:
            days: Number of days to analyze

        Returns:
            List of emotional progression data points
        """
        try:
            async with self.async_session() as session:
                query = select(
                    JournalEntry.created_at,
                    JournalEntry.emotion,
                    JournalEntry.sentiment_score
                ).where(
                    JournalEntry.created_at >= func.datetime('now', f'-{days} days')
                ).order_by(JournalEntry.created_at)

                result = await session.execute(query)
                progression = [
                    {
                        "date": row.created_at,
                        "emotion": row.emotion.value,  # Access enum value
                        "sentiment": row.sentiment_score
                    }
                    for row in result.all()
                ]
                logger.info("Retrieved emotional progression for the last %d days.", days)
                return progression
        except OperationalError as oe:
            logger.warning("OperationalError encountered while fetching emotional progression: %s. Attempting to reinitialize the database.", oe)
            await self.initialize_database()  # Attempt to recreate tables
            # Retry the operation once after reinitialization
            try:
                async with self.async_session() as session:
                    query = select(
                        JournalEntry.created_at,
                        JournalEntry.emotion,
                        JournalEntry.sentiment_score
                    ).where(
                        JournalEntry.created_at >= func.datetime('now', f'-{days} days')
                    ).order_by(JournalEntry.created_at)

                    result = await session.execute(query)
                    progression = [
                        {
                            "date": row.created_at,
                            "emotion": row.emotion.value,  # Access enum value
                            "sentiment": row.sentiment_score
                        }
                        for row in result.all()
                    ]
                    logger.info("Retrieved emotional progression for the last %d days after reinitialization.", days)
                    return progression
            except SQLAlchemyError as e:
                logger.error("Database error after reinitialization while fetching emotional progression: %s", e)
                raise SQLAlchemyDBError("Failed to fetch emotional progression after reinitialization.", {"error": str(e)})
        except SQLAlchemyError as e:
            logger.error("Database error while fetching emotional progression: %s", e)
            raise SQLAlchemyDBError("Failed to fetch emotional progression.", {"error": str(e)})

    async def close(self) -> None:
        """Close database connections."""
        await self.engine.dispose()
        logger.info("Database connections closed.")


# Initialize the Singleton DatabaseManager instance
database = DatabaseManager()

# Dependency for FastAPI/Streamlit to get database sessions
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to provide a database session.

    Yields:
        AsyncSession for database operations
    """
    async for session in database.get_session():
        yield session
