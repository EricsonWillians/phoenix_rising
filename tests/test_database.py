"""
Test Suite for the Database Layer.

This module provides comprehensive testing coverage for all database operations,
ensuring data integrity and proper handling of journal entries and emotional insights.
"""

import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, List
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker
)

from src.database import (
    Base,
    DatabaseManager,
    JournalEntry,
    EmotionalInsight
)
from src.schemas import (
    JournalEntryCreate,
    JournalEntryResponse,
    EmotionState
)

@pytest_asyncio.fixture
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    try:
        yield engine
    finally:
        await engine.dispose()

@pytest_asyncio.fixture
async def test_session(
    test_engine
) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session

@pytest_asyncio.fixture
async def database_manager(
    test_engine
) -> AsyncGenerator[DatabaseManager, None]:
    """Create test database manager."""
    manager = DatabaseManager(
        database_url="sqlite+aiosqlite:///:memory:"
    )
    manager.engine = test_engine
    manager.async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    yield manager
    await manager.close()

@pytest.fixture
def sample_entry() -> JournalEntryCreate:
    """Provide sample journal entry data."""
    return JournalEntryCreate(
        content="Finding strength in the depths of shadow",
        emotion=EmotionState.SHADOW,
        light_token="Through darkness, wisdom emerges",
        sentiment_score=-0.2
    )

@pytest.fixture
def sample_entries() -> List[JournalEntryCreate]:
    """Provide multiple sample journal entries."""
    return [
        JournalEntryCreate(
            content="Navigating corporate shadows",
            emotion=EmotionState.SHADOW,
            light_token="Shadows teach us to seek light",
            sentiment_score=-0.4
        ),
        JournalEntryCreate(
            content="Dawn breaks through despair",
            emotion=EmotionState.DAWN,
            light_token="Hope illuminates the path",
            sentiment_score=0.6
        ),
        JournalEntryCreate(
            content="Storm of transformation",
            emotion=EmotionState.STORM,
            light_token="Chaos births new beginnings",
            sentiment_score=0.1
        )
    ]

class TestDatabaseInitialization:
    """Test database initialization and configuration."""
    
    @pytest.mark.asyncio
    async def test_database_creation(self, database_manager):
        """Test database creation and table setup."""
        await database_manager.create_tables()
        
        async with database_manager.engine.begin() as conn:
            # Verify tables exist
            result = await conn.run_sync(
                lambda sync_conn: sync_conn.execute(
                    select(1)
                    .from_(JournalEntry.__table__)
                    .limit(1)
                )
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_session_management(self, database_manager):
        """Test session creation and management."""
        async for session in database_manager.get_session():
            assert isinstance(session, AsyncSession)
            # Test session is active
            assert not session.is_active

class TestJournalEntryOperations:
    """Test journal entry creation and retrieval operations."""
    
    @pytest.mark.asyncio
    async def test_create_journal_entry(
        self,
        database_manager,
        test_session,
        sample_entry
    ):
        """Test creation of a single journal entry."""
        entry = await database_manager.create_journal_entry(
            sample_entry,
            test_session
        )
        
        assert entry.id is not None
        assert entry.content == sample_entry.content
        assert entry.emotion == sample_entry.emotion
        assert entry.light_token == sample_entry.light_token
        assert entry.sentiment_score == sample_entry.sentiment_score
        assert isinstance(entry.created_at, datetime)

    @pytest.mark.asyncio
    async def test_get_recent_entries(
        self,
        database_manager,
        test_session,
        sample_entries
    ):
        """Test retrieval of recent journal entries."""
        # Create multiple entries
        for entry_data in sample_entries:
            await database_manager.create_journal_entry(
                entry_data,
                test_session
            )
        
        # Retrieve recent entries
        entries = await database_manager.get_recent_entries(
            test_session,
            limit=2
        )
        
        assert len(entries) == 2
        assert isinstance(entries[0], JournalEntry)
        assert entries[0].created_at > entries[1].created_at

    @pytest.mark.asyncio
    async def test_get_emotional_progression(
        self,
        database_manager,
        test_session,
        sample_entries
    ):
        """Test emotional progression data retrieval."""
        # Create entries with different timestamps
        base_time = datetime.utcnow()
        for i, entry_data in enumerate(sample_entries):
            entry = await database_manager.create_journal_entry(
                entry_data,
                test_session
            )
            entry.created_at = base_time - timedelta(days=i)
            await test_session.commit()
        
        progression = await database_manager.get_emotional_progression(
            test_session,
            days=30
        )
        
        assert len(progression) == len(sample_entries)
        for point in progression:
            assert "date" in point
            assert "emotion" in point
            assert "sentiment" in point

class TestEmotionalInsights:
    """Test emotional insight tracking and analysis."""
    
    @pytest.mark.asyncio
    async def test_create_emotional_insight(
        self,
        database_manager,
        test_session,
        sample_entry
    ):
        """Test creation of emotional insights."""
        # Create journal entry
        entry = await database_manager.create_journal_entry(
            sample_entry,
            test_session
        )
        
        # Create insight
        insight = EmotionalInsight(
            journal_entry_id=entry.id,
            insight_type="resilience",
            value=0.75
        )
        test_session.add(insight)
        await test_session.commit()
        
        # Verify insight
        assert insight.id is not None
        assert insight.journal_entry_id == entry.id
        assert insight.value == 0.75

    @pytest.mark.asyncio
    async def test_get_insights_for_entry(
        self,
        database_manager,
        test_session,
        sample_entry
    ):
        """Test retrieval of insights for a specific entry."""
        entry = await database_manager.create_journal_entry(
            sample_entry,
            test_session
        )
        
        # Create multiple insights
        insights = [
            EmotionalInsight(
                journal_entry_id=entry.id,
                insight_type="growth",
                value=0.6
            ),
            EmotionalInsight(
                journal_entry_id=entry.id,
                insight_type="resilience",
                value=0.8
            )
        ]
        test_session.add_all(insights)
        await test_session.commit()
        
        # Refresh entry to load relationships
        await test_session.refresh(entry)
        
        assert len(entry.emotional_insights) == 2
        assert isinstance(entry.emotional_insights[0], EmotionalInsight)

class TestErrorHandling:
    """Test database error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_duplicate_entry_handling(
        self,
        database_manager,
        test_session,
        sample_entry
    ):
        """Test handling of duplicate entries."""
        # Create first entry
        entry1 = await database_manager.create_journal_entry(
            sample_entry,
            test_session
        )
        
        # Attempt to create duplicate
        entry2 = await database_manager.create_journal_entry(
            sample_entry,
            test_session
        )
        
        assert entry1.id != entry2.id
        assert entry1.content == entry2.content

    @pytest.mark.asyncio
    async def test_transaction_rollback(
        self,
        database_manager,
        test_session
    ):
        """Test transaction rollback on error."""
        async with test_session.begin():
            # Create valid entry
            await database_manager.create_journal_entry(
                JournalEntryCreate(
                    content="Valid entry",
                    emotion=EmotionState.DAWN,
                    light_token="Test token"
                ),
                test_session
            )
            
            # Attempt invalid operation
            with pytest.raises(Exception):
                await test_session.execute(
                    "Invalid SQL"
                )
        
        # Verify transaction was rolled back
        result = await test_session.execute(
            select(JournalEntry).where(
                JournalEntry.content == "Valid entry"
            )
        )
        entry = result.scalar_one_or_none()
        assert entry is None

if __name__ == "__main__":
    pytest.main(["-v", "--asyncio-mode=auto"])