"""
Phoenix Rising Utilities and Configuration.

This module provides essential utilities, configurations, and helper functions
that support the core components of the Phoenix Rising application. It ensures
consistent behavior, proper error handling, and meaningful logging across
the entire system.
"""

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
import aiofiles
from pydantic import BaseModel, Field
import asyncio
from functools import wraps
import traceback

# Configure logging with rotation
log_path = Path("logs")
log_path.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            log_path / "phoenix.log",
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EmotionState(str, Enum):
    """Enumeration of possible emotional states."""
    EMBER = "Ember"
    SHADOW = "Shadow"
    STORM = "Storm"
    DAWN = "Dawn"
    STARLIGHT = "Starlight"

class SupportLevel(str, Enum):
    """Enumeration of support response levels."""
    NONE = "none"
    GENTLE = "gentle"
    SUPPORTIVE = "supportive"
    CONCERNED = "concerned"

class ApplicationConfig(BaseModel):
    """Configuration settings for the application."""
    app_name: str = Field(default="Phoenix Rising")
    version: str = Field(default="1.0.0")
    debug_mode: bool = Field(default=False)
    max_entry_length: int = Field(default=2000)
    max_token_length: int = Field(default=200)
    sentiment_threshold: float = Field(default=-0.7)
    support_thresholds: Dict[str, float] = Field(default={
        "gentle": -0.3,
        "supportive": -0.5,
        "concerned": -0.7
    })

class AsyncRetry:
    """Decorator for async function retry logic."""
    
    def __init__(
        self,
        retries: int = 3,
        delay: float = 1.0,
        exceptions: tuple = (Exception,)
    ):
        self.retries = retries
        self.delay = delay
        self.exceptions = exceptions

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(self.retries):
                try:
                    return await func(*args, **kwargs)
                except self.exceptions as e:
                    last_exception = e
                    if attempt < self.retries - 1:
                        delay = self.delay * (attempt + 1)
                        logger.warning(
                            f"Retry attempt {attempt + 1} for {func.__name__} "
                            f"after {delay}s due to {str(e)}"
                        )
                        await asyncio.sleep(delay)
            raise last_exception
        return wrapper

class Journey:
    """Utility class for managing spiritual journey analytics."""

    @staticmethod
    def calculate_growth_metrics(
        entries: list,
        window_size: int = 7
    ) -> Dict[str, float]:
        """
        Calculate growth metrics from journal entries.
        
        Args:
            entries: List of journal entries
            window_size: Size of the rolling window for calculations
            
        Returns:
            Dictionary containing growth metrics
        """
        if not entries:
            return {
                "emotional_variance": 0.0,
                "growth_index": 0.0,
                "resilience_score": 0.0
            }

        sentiment_scores = [
            entry.sentiment_score for entry in entries 
            if entry.sentiment_score is not None
        ]
        
        if not sentiment_scores:
            return {
                "emotional_variance": 0.0,
                "growth_index": 0.0,
                "resilience_score": 0.0
            }

        # Calculate emotional variance
        emotional_variance = sum(
            abs(a - b) 
            for a, b in zip(sentiment_scores[1:], sentiment_scores[:-1])
        ) / len(sentiment_scores)

        # Calculate growth index
        rolling_avg = sum(sentiment_scores[-window_size:]) / min(
            window_size,
            len(sentiment_scores)
        )
        overall_avg = sum(sentiment_scores) / len(sentiment_scores)
        growth_index = (rolling_avg - overall_avg + 1) / 2

        # Calculate resilience score
        negative_rebounds = sum(
            1 for a, b in zip(sentiment_scores[1:], sentiment_scores[:-1])
            if a < -0.3 and b > 0
        )
        resilience_score = negative_rebounds / len(sentiment_scores)

        return {
            "emotional_variance": emotional_variance,
            "growth_index": growth_index,
            "resilience_score": resilience_score
        }

class DataProcessor:
    """Utility class for data processing and transformation."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text input.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        text = text.strip()
        # Remove excess whitespace
        text = " ".join(text.split())
        return text

    @staticmethod
    def validate_emotion(emotion: str) -> bool:
        """
        Validate if an emotion is recognized.
        
        Args:
            emotion: Emotion to validate
            
        Returns:
            True if emotion is valid
        """
        try:
            EmotionState(emotion)
            return True
        except ValueError:
            return False

    @staticmethod
    def get_support_level(
        sentiment_score: float,
        config: ApplicationConfig
    ) -> SupportLevel:
        """
        Determine appropriate support level based on sentiment.
        
        Args:
            sentiment_score: Sentiment analysis score
            config: Application configuration
            
        Returns:
            Appropriate support level
        """
        thresholds = config.support_thresholds
        
        if sentiment_score <= thresholds["concerned"]:
            return SupportLevel.CONCERNED
        elif sentiment_score <= thresholds["supportive"]:
            return SupportLevel.SUPPORTIVE
        elif sentiment_score <= thresholds["gentle"]:
            return SupportLevel.GENTLE
        return SupportLevel.NONE

async def save_backup(data: Dict[str, Any], backup_path: Path) -> None:
    """
    Save application data backup asynchronously.
    
    Args:
        data: Data to backup
        backup_path: Path to save backup
    """
    backup_path.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_file = backup_path / f"phoenix_backup_{timestamp}.json"
    
    async with aiofiles.open(backup_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data, default=str, indent=2))

def setup_error_handling() -> None:
    """Configure global error handling and logging."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error(
            "Uncaught exception:",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception

# Application configuration instance
config = ApplicationConfig()

# Initialize error handling
setup_error_handling()