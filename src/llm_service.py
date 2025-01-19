"""
LightBearer Service: A bridge between human experience and AI-generated wisdom.

This module handles interactions with the Phi-3.5-mini-instruct model,
including input validation, sentiment analysis, and response generation.
It maintains a balance between technical robustness and spiritual comfort.
"""

from typing import Dict, Optional, Tuple, Any
import json
import os
import logging
from pathlib import Path
from datetime import datetime
import asyncio
import aiohttp
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JournalEntry(BaseModel):
    """Validation model for journal entries."""
    content: str = Field(..., min_length=1, max_length=2000)
    emotion: str = Field(..., regex="^(Ember|Shadow|Storm|Dawn|Starlight)$")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SentimentResponse(BaseModel):
    """Model for sentiment analysis response."""
    score: float = Field(..., ge=-1, le=1)
    is_concerning: bool
    requires_support: bool

class LightBearerException(Exception):
    """Base exception for LightBearer service."""
    pass

class PromptTemplateError(LightBearerException):
    """Raised when there's an error with prompt templates."""
    pass

class APIConnectionError(LightBearerException):
    """Raised when there's an error connecting to the HuggingFace API."""
    pass

class InputValidationError(LightBearerException):
    """Raised when input validation fails."""
    pass

class LightBearer:
    """
    A service for transforming human experiences into tokens of light using
    the Phi-3.5-mini-instruct model.
    """
    
    def __init__(
        self, 
        prompt_path: Optional[str] = None,
        sentiment_threshold: float = -0.7
    ) -> None:
        """
        Initialize the LightBearer service.
        
        Args:
            prompt_path: Path to prompt templates JSON file
            sentiment_threshold: Threshold for concerning negative sentiment
            
        Raises:
            PromptTemplateError: If prompt templates cannot be loaded
            ValueError: If required environment variables are missing
        """
        load_dotenv()
        
        self.api_token: str = os.getenv("HUGGINGFACE_API_TOKEN", "")
        self.endpoint: str = os.getenv("MODEL_ENDPOINT", "")
        self.sentiment_threshold = sentiment_threshold
        
        if not self.api_token or not self.endpoint:
            raise ValueError(
                "Missing required environment variables: "
                "HUGGINGFACE_API_TOKEN and MODEL_ENDPOINT must be set"
            )
            
        self.prompt_path = prompt_path or Path("assets/prompts/light_seeds.json")
        self.prompts: Dict[str, Any] = self._load_prompts()
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Phi-3.5-mini-instruct specific configurations
        self.model_config = {
            "max_length": 128,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }

    async def _analyze_sentiment(self, text: str) -> SentimentResponse:
        """
        Analyze the sentiment of input text using Phi-3.5-mini-instruct.
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentResponse containing analysis results
        """
        sentiment_prompt = f"""<|system|>
You are a careful emotional analyst. Analyze the following text and provide a sentiment score between -1 (extremely negative) and 1 (extremely positive). Consider context and nuance.<|end|>
<|user|>
Text: {text}
Provide only a number between -1 and 1.<|end|>
<|assistant|>"""

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_token}"},
                json={"inputs": sentiment_prompt}
            ) as response:
                result = await response.json()
                try:
                    score = float(result[0]["generated_text"].strip())
                    score = max(-1, min(1, score))  # Ensure bounds
                except (ValueError, KeyError, IndexError):
                    score = 0.0  # Default to neutral if parsing fails
                
                return SentimentResponse(
                    score=score,
                    is_concerning=score <= self.sentiment_threshold,
                    requires_support=score <= -0.5
                )

    def _validate_and_transform_input(
        self, 
        entry: str, 
        emotion: str
    ) -> JournalEntry:
        """
        Validate and transform input data.
        
        Args:
            entry: Journal entry text
            emotion: Selected emotion
            
        Returns:
            Validated JournalEntry object
            
        Raises:
            InputValidationError: If validation fails
        """
        try:
            return JournalEntry(
                content=entry,
                emotion=emotion
            )
        except ValidationError as e:
            raise InputValidationError(str(e))

    async def generate_light_token(
        self, 
        entry: str, 
        emotion: str,
    ) -> Tuple[str, Optional[str]]:
        """
        Generate a light token and optional support message.
        
        Args:
            entry: Journal entry text
            emotion: Selected emotion
            
        Returns:
            Tuple of (light token, optional support message)
            
        Raises:
            InputValidationError: If input validation fails
            APIConnectionError: If API call fails
        """
        # Validate input
        validated_entry = self._validate_and_transform_input(entry, emotion)
        
        # Analyze sentiment
        sentiment = await self._analyze_sentiment(validated_entry.content)
        
        # Prepare support message if needed
        support_message = None
        if sentiment.requires_support:
            support_message = self._get_support_message(sentiment.score)
            
        # Adjust prompt based on sentiment
        base_prompt = self.prompts["transformation_prompt"]
        if sentiment.is_concerning:
            base_prompt = self.prompts.get("healing_prompt", base_prompt)
            
        # Construct final prompt with chat format
        prompt = f"""<|system|>
You are LightBearer, a compassionate guide transforming human experiences into tokens of light and hope. Your responses are brief but meaningful, focusing on growth and resilience.<|end|>
<|user|>
Entry: {validated_entry.content}
Emotion: {validated_entry.emotion}
Transform this experience into a token of light.<|end|>
<|assistant|>"""

        # Generate response with Phi-3.5 specific settings
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_token}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": prompt,
                        "parameters": self.model_config
                    }
                ) as response:
                    if response.status != 200:
                        raise APIConnectionError(
                            f"API request failed with status {response.status}"
                        )
                    
                    result = await response.json()
                    token = result[0]["generated_text"].strip()
                    return self._post_process_token(token), support_message
                    
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Failed to connect to API: {str(e)}")

    def _get_support_message(self, sentiment_score: float) -> str:
        """Generate an appropriate support message based on sentiment."""
        if sentiment_score <= -0.8:
            return ("Your pain is heard. Remember that you're not alone. "
                   "Consider reaching out to supportive friends, family, "
                   "or mental health professionals.")
        elif sentiment_score <= -0.5:
            return ("Remember to be gentle with yourself. Each moment "
                   "carries the potential for transformation.")
        return None

    def _post_process_token(self, token: str) -> str:
        """Clean and format the generated token."""
        # Remove any system/user prompts that might have been generated
        if "<|system|>" in token:
            token = token.split("<|system|>")[-1]
        if "<|user|>" in token:
            token = token.split("<|user|>")[-1]
        if "<|assistant|>" in token:
            token = token.split("<|assistant|>")[-1]
            
        # Clean up whitespace and ensure proper formatting
        token = token.strip()
        token = token[0].upper() + token[1:] if token else token
        
        # Ensure token isn't too long
        if len(token) > 200:
            token = token[:197] + "..."
            
        return token

    async def __aenter__(self) -> 'LightBearer':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self.session:
            await self.session.close()