"""
Test Suite for the LightBearer Service.

This module provides comprehensive testing coverage for the LightBearer
service, ensuring reliable token generation and sentiment analysis while
maintaining the spiritual integrity of our sanctuary.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import aiohttp
from aiohttp import ClientResponse, StreamReader

from src.llm_service import (
    LightBearer,
    LightBearerException,
    PromptTemplateError,
    APIConnectionError
)
from src.schemas import EmotionState, SentimentAnalysis
from src.utils import ApplicationConfig

class MockResponse:
    """Mock aiohttp response for testing."""
    def __init__(self, status, json_data):
        self.status = status
        self._json_data = json_data

    async def json(self):
        return self._json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

@pytest.fixture
def mock_config():
    """Provide test configuration."""
    return ApplicationConfig(
        app_name="Phoenix Rising Test",
        debug_mode=True,
        max_entry_length=500,
        max_token_length=100,
        sentiment_threshold=-0.7
    )

@pytest.fixture
def mock_prompts():
    """Provide mock prompt templates."""
    return {
        "transformation_prompt": "Transform: {entry}\nEmotion: {emotion}",
        "healing_prompt": "Heal: {entry}\nEmotion: {emotion}",
        "emotions": {
            "Ember": "Warmth of rebirth",
            "Shadow": "Hidden strength",
            "Storm": "Transformative chaos",
            "Dawn": "Breaking light",
            "Starlight": "Eternal guidance"
        }
    }

@pytest_asyncio.fixture
async def light_bearer(mock_config, tmp_path):
    """Create LightBearer instance for testing."""
    # Create temporary prompt file
    prompt_path = tmp_path / "light_seeds.json"
    prompt_path.write_text(json.dumps({
        "transformation_prompt": "Transform: {entry}\nEmotion: {emotion}",
        "emotions": {
            "Ember": "Test description",
            "Shadow": "Test description",
            "Storm": "Test description",
            "Dawn": "Test description",
            "Starlight": "Test description"
        }
    }))
    
    with patch.dict('os.environ', {
        'HUGGINGFACE_API_TOKEN': 'test_token',
        'MODEL_ENDPOINT': 'https://test.endpoint'
    }):
        bearer = LightBearer(prompt_path=str(prompt_path))
        yield bearer

class TestLightBearerInitialization:
    """Test LightBearer initialization and configuration."""

    @pytest.mark.asyncio
    async def test_initialization_with_valid_config(self, mock_config, tmp_path):
        """Test successful initialization."""
        prompt_path = tmp_path / "light_seeds.json"
        prompt_path.write_text(json.dumps({
            "transformation_prompt": "Test prompt",
            "emotions": {"Dawn": "Test"}
        }))
        
        with patch.dict('os.environ', {
            'HUGGINGFACE_API_TOKEN': 'test_token',
            'MODEL_ENDPOINT': 'https://test.endpoint'
        }):
            bearer = LightBearer(prompt_path=str(prompt_path))
            assert bearer.api_token == 'test_token'
            assert bearer.endpoint == 'https://test.endpoint'

    def test_initialization_without_environment_variables(self):
        """Test initialization failure without environment variables."""
        with patch.dict('os.environ', clear=True):
            with pytest.raises(ValueError) as exc_info:
                LightBearer()
            assert "Missing required environment variables" in str(exc_info.value)

    def test_initialization_with_invalid_prompt_file(self):
        """Test initialization with invalid prompt file."""
        with pytest.raises(PromptTemplateError):
            LightBearer(prompt_path="nonexistent.json")

class TestLightTokenGeneration:
    """Test light token generation functionality."""

    @pytest.mark.asyncio
    async def test_successful_token_generation(self, light_bearer):
        """Test successful generation of a light token."""
        mock_response = MockResponse(
            200,
            [{"generated_text": "A beacon of hope emerges"}]
        )
        
        with patch('aiohttp.ClientSession.post', 
                  return_value=mock_response):
            token, support = await light_bearer.generate_light_token(
                entry="Finding strength in darkness",
                emotion=EmotionState.DAWN
            )
            
            assert token == "A beacon of hope emerges"
            assert support is None

    @pytest.mark.asyncio
    async def test_token_generation_with_concerning_sentiment(self, light_bearer):
        """Test token generation with concerning sentiment."""
        # Mock sentiment analysis response
        sentiment_response = MockResponse(
            200,
            [{"generated_text": "-0.8"}]
        )
        
        # Mock token generation response
        token_response = MockResponse(
            200,
            [{"generated_text": "Light persists even in darkness"}]
        )
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = [sentiment_response, token_response]
            
            token, support = await light_bearer.generate_light_token(
                entry="Feeling overwhelmed by shadows",
                emotion=EmotionState.SHADOW
            )
            
            assert token is not None
            assert support is not None
            assert "support" in support.lower()

    @pytest.mark.asyncio
    async def test_token_generation_with_api_error(self, light_bearer):
        """Test error handling during token generation."""
        mock_response = MockResponse(500, {"error": "API Error"})
        
        with patch('aiohttp.ClientSession.post',
                  return_value=mock_response):
            with pytest.raises(APIConnectionError) as exc_info:
                await light_bearer.generate_light_token(
                    entry="Test entry",
                    emotion=EmotionState.STORM
                )
            assert "API request failed" in str(exc_info.value)

class TestSentimentAnalysis:
    """Test sentiment analysis functionality."""

    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, light_bearer):
        """Test successful sentiment analysis."""
        mock_response = MockResponse(
            200,
            [{"generated_text": "0.5"}]
        )
        
        with patch('aiohttp.ClientSession.post',
                  return_value=mock_response):
            sentiment = await light_bearer._analyze_sentiment(
                "Finding hope in the journey"
            )
            
            assert isinstance(sentiment, SentimentAnalysis)
            assert sentiment.score == 0.5
            assert not sentiment.is_concerning

    @pytest.mark.asyncio
    async def test_sentiment_analysis_with_concerning_content(self, light_bearer):
        """Test sentiment analysis with concerning content."""
        mock_response = MockResponse(
            200,
            [{"generated_text": "-0.8"}]
        )
        
        with patch('aiohttp.ClientSession.post',
                  return_value=mock_response):
            sentiment = await light_bearer._analyze_sentiment(
                "Drowning in corporate darkness"
            )
            
            assert sentiment.score == -0.8
            assert sentiment.is_concerning
            assert sentiment.requires_support

class TestPromptManagement:
    """Test prompt template management."""

    def test_prompt_validation(self, light_bearer, mock_prompts):
        """Test prompt template validation."""
        with patch.object(light_bearer, 'prompts', mock_prompts):
            prompt = light_bearer._construct_prompt(
                "Test entry",
                EmotionState.DAWN
            )
            assert "Test entry" in prompt
            assert "Dawn" in prompt

    def test_invalid_emotion_handling(self, light_bearer):
        """Test handling of invalid emotions."""
        with pytest.raises(ValueError) as exc_info:
            light_bearer._validate_emotion("InvalidEmotion")
        assert "Unsupported emotion" in str(exc_info.value)

class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_retry_logic(self, light_bearer):
        """Test retry logic for API failures."""
        mock_success = MockResponse(
            200,
            [{"generated_text": "Success after retry"}]
        )
        mock_error = MockResponse(500, {"error": "Temporary error"})
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = [mock_error, mock_error, mock_success]
            
            token, _ = await light_bearer.generate_light_token(
                entry="Test retry",
                emotion=EmotionState.DAWN
            )
            
            assert token == "Success after retry"
            assert mock_post.call_count == 3

    @pytest.mark.asyncio
    async def test_input_validation(self, light_bearer):
        """Test input validation for token generation."""
        with pytest.raises(ValueError) as exc_info:
            await light_bearer.generate_light_token(
                entry="",  # Empty entry
                emotion=EmotionState.DAWN
            )
        assert "cannot be empty" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main(["-v", "--asyncio-mode=auto"])