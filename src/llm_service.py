# src/llm_service.py

import os
import json
import asyncio
import logging
import re
import traceback
from typing import Optional, Tuple, Dict, Any, Set
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
from pydantic import BaseModel, Field, validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    RetryError,
)
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

# Configure logging with more detailed format
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# Exception Hierarchy
class LightBearerException(Exception):
    """Base exception for LightBearer service."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


class APIConnectionError(LightBearerException):
    """Exception raised for API connection errors."""

    pass


class APIEndpointUnavailableError(LightBearerException):
    """Exception raised when the API endpoint is unavailable."""

    pass


class InvalidAPIResponseError(LightBearerException):
    """Exception raised for invalid API responses."""

    pass


class ConfigurationError(LightBearerException):
    """Exception raised for configuration-related errors."""

    pass


# Add after the exception classes


class ServiceStatus:
    """Track service availability status with thread-safe operations."""

    def __init__(self, cooldown_minutes: int = 1):
        self.sentiment_available: bool = True
        self.generation_available: bool = True
        self.last_check: datetime = datetime.utcnow()
        self.cooldown_period: timedelta = timedelta(minutes=cooldown_minutes)
        self._lock = asyncio.Lock()

    async def mark_service_unavailable(self, service_type: str) -> None:
        """Thread-safe marking of service unavailability."""
        async with self._lock:
            if service_type == "sentiment":
                self.sentiment_available = False
            elif service_type == "generation":
                self.generation_available = False
            self.last_check = datetime.utcnow()
            logger.warning(f"Service marked unavailable: {service_type}")

    def can_retry(self) -> bool:
        """Check if cooldown period has elapsed."""
        return datetime.utcnow() - self.last_check > self.cooldown_period

    async def mark_service_available(self, service_type: str) -> None:
        """Thread-safe marking of service availability."""
        async with self._lock:
            if service_type == "sentiment":
                self.sentiment_available = True
            elif service_type == "generation":
                self.generation_available = True
            logger.info(f"Service marked available: {service_type}")


# Enhanced Configuration Models
class ModelConfig(BaseModel):
    """Model configuration with validation."""

    name: str
    endpoint: str
    pipeline: str
    parameters: Dict[str, Any]
    timeout: int = 60
    max_retries: int = 5

    @validator("pipeline")
    def validate_pipeline(cls, v):
        allowed_pipelines = {"text-generation", "zero-shot-classification"}
        if v not in allowed_pipelines:
            raise ValueError(f"Pipeline must be one of {allowed_pipelines}")
        return v


class EmotionState(BaseModel):
    """Enhanced emotion state schema with validation."""

    name: str
    description: str
    context: str
    transformation_patterns: list[str]
    generation_parameters: Dict[str, Any]
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    @validator("transformation_patterns")
    def validate_patterns(cls, v):
        if not v:
            raise ValueError("Must provide at least one transformation pattern")
        return v


class SupportMessage(BaseModel):
    """Enhanced support message schema."""

    threshold: float = Field(..., ge=-1.0, le=1.0)
    messages: list[str]
    context: Optional[str] = None

    @validator("messages")
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Must provide at least one support message")
        return v


class StyleGuide(BaseModel):
    """Enhanced style guide with specific constraints."""

    tone: str
    language: str = Field(default="English")
    format: str
    use_emojis: bool
    max_length: int = Field(default=150, ge=50, le=500)


class TokenGenerationRules(BaseModel):
    """Enhanced token generation rules."""

    max_length: int = Field(default=150, ge=50, le=500)
    min_length: int = Field(default=20, ge=10, le=100)
    style_guide: StyleGuide
    require_emoji: bool = True
    temperature_range: Tuple[float, float] = Field(default=(0.6, 0.9))


class PromptsConfig(BaseModel):
    """Enhanced prompts configuration."""

    transformation_prompt: str
    healing_prompt: str
    sentiment_prompt: str
    emotions: Dict[str, EmotionState]
    support_messages: Dict[str, SupportMessage]
    token_generation_rules: TokenGenerationRules

    @validator("emotions")
    def validate_emotions(cls, v):
        if not v:
            raise ValueError("Must provide at least one emotion state")
        return v


class SentimentResponse(BaseModel):
    """Enhanced sentiment analysis response."""

    score: float = Field(..., ge=-1.0, le=1.0)
    is_concerning: bool
    requires_support: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    dominant_emotion: str
    emotion_scores: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LightBearer:
    """
    Enhanced LightBearer Service for emotional analysis and transformation.

    This service handles interactions with Hugging Face's Inference Endpoints
    for sentiment analysis and therapeutic text generation, incorporating
    robust error handling, validation, and response processing.
    """

    RETRY_ATTEMPTS: int = 5
    RETRY_WAIT = wait_exponential(multiplier=1, min=2, max=20)
    DEFAULT_TIMEOUT: int = 60

    def __init__(
        self,
        config_path: str = "assets/prompts/light_seeds.json",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the LightBearer service with configuration and caching support.

        Args:
            config_path: Path to the prompts configuration JSON file
            cache_dir: Optional directory for response caching

        Raises:
            ConfigurationError: If required configuration is missing or invalid
        """
        """Initialize the LightBearer service."""
        self._load_environment()
        self._initialize_models()
        self._load_configuration(config_path)
        self._setup_caching(cache_dir)
        
        self.service_status = ServiceStatus()  # Initialize service status
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_sentiment_score: Optional[float] = None

    def _load_environment(self) -> None:
        """Load and validate environment variables."""
        required_vars = {
            "HUGGINGFACE_API_TOKEN": "API token",
            "CHAT_MODEL_ENDPOINT": "Chat model endpoint",
            "SENTIMENT_MODEL_ENDPOINT": "Sentiment model endpoint",
        }

        missing = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing.append(description)

        if missing:
            raise ConfigurationError(
                "Missing required environment variables", {"missing": missing}
            )

        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.chat_endpoint = os.getenv("CHAT_MODEL_ENDPOINT")
        self.sentiment_endpoint = os.getenv("SENTIMENT_MODEL_ENDPOINT")

    def _initialize_models(self) -> None:
        """Initialize model configurations."""
        self.models = {
            "sentiment": ModelConfig(
                name="sentiment-analysis",
                endpoint=self.sentiment_endpoint,
                pipeline="zero-shot-classification",
                parameters={
                    "candidate_labels": [
                        "joy",
                        "contentment",
                        "gratitude",
                        "hope",
                        "pride",
                        "love",
                        "awe",
                        "anxiety",
                        "sadness",
                        "anger",
                        "fear",
                        "shame",
                        "loneliness",
                    ],
                    "multi_label": True,
                    "hypothesis_template": "This text expresses {}",
                },
            ),
            "generation": ModelConfig(
                name="text-generation",
                endpoint=self.chat_endpoint,
                pipeline="text-generation",
                parameters={
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repetition_penalty": 1.2,
                    "do_sample": True,
                },
            ),
        }

    def _load_configuration(self, config_path: str) -> None:
        """Load and validate the prompts configuration."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            self.prompts = PromptsConfig(**config_data)
            logger.info("Successfully loaded prompts configuration")
        except Exception as e:
            raise ConfigurationError(
                "Failed to load prompts configuration",
                {"error": str(e), "traceback": traceback.format_exc()},
            )

    def _setup_caching(self, cache_dir: Optional[str]) -> None:
        """Set up response caching if enabled."""
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized caching in {self.cache_dir}")
        else:
            self.cache_dir = None

    async def __aenter__(self):
        """Asynchronous context manager entry."""
        await self._init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Asynchronous context manager exit."""
        await self._close_session()

    async def _init_session(self) -> None:
        """Initialize the aiohttp ClientSession with proper configuration."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.DEFAULT_TIMEOUT)
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
            logger.debug("Initialized aiohttp ClientSession")

    async def _close_session(self) -> None:
        """Safely close the aiohttp ClientSession."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.debug("Closed aiohttp ClientSession")

    def _construct_sentiment_prompt(self, text: str) -> str:
        """
        Construct an enhanced prompt for sentiment analysis.

        Args:
            text: Input text to analyze

        Returns:
            str: Constructed prompt
        """
        template = """
            Analyze the emotional content and sentiment of the following text.
            Consider both explicit emotions and subtle undertones.
            Look for signs of:
            - Primary emotions
            - Emotional intensity
            - Underlying patterns
            - Need for support

            Text: {text}

            Please provide a detailed analysis focusing on the emotional state of the writer.
            """
        return template.format(text=text)

    def _construct_generation_prompt(
            self, entry: str, emotion: str, sentiment: SentimentResponse
        ) -> str:
            """
            Construct an enhanced prompt for token generation with proper markers.

            Args:
                entry: Journal entry text
                emotion: Selected emotion state
                sentiment: Sentiment analysis results

            Returns:
                str: Constructed prompt with markers
            """
            emotion_state = self.prompts.emotions[emotion]
            transformation_focus = self._determine_transformation_focus(sentiment)

            template = """
        <|user|>
        Emotion Context: {emotion_context}
        Journal Entry: {journal_entry}
        <|end|>
        <|assistant|>
        Task: Generate a transformative message that:
        1. Acknowledges their current emotional experience with empathy
        2. Identifies potential for growth or healing
        3. Offers gentle guidance or perspective
        4. Uses relevant metaphors or imagery
        5. Maintains a hopeful, supportive tone

        Requirements:
        - Be genuine and specific to their situation
        - Avoid generic platitudes
        - Include one meaningful metaphor
        - Keep response under {max_length} words
        - Include one appropriate ✨ emoji

        Focus particularly on: {focus}
        """.strip()

            emotion_context = emotion_state.context

            prompt = template.format(
                emotion_context=emotion_context,
                journal_entry=entry,
                max_length=self.prompts.token_generation_rules.max_length,
                focus=transformation_focus,
            )

            return prompt

    def _determine_transformation_focus(self, sentiment: SentimentResponse) -> str:
        """
        Determine the appropriate transformation focus based on sentiment.

        Args:
            sentiment: Sentiment analysis results

        Returns:
            str: Transformation focus
        """
        if sentiment.score <= -0.7:
            return "providing emotional support and acknowledging difficulty"
        elif sentiment.score <= -0.3:
            return "encouraging resilience while validating emotions"
        elif sentiment.score <= 0.3:
            return "finding balance and perspective"
        else:
            return "building on positive momentum and growth"

    @retry(
        retry=retry_if_exception_type(APIConnectionError),
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=RETRY_WAIT,
    )
    async def _analyze_sentiment_zero_shot(self, text: str) -> SentimentResponse:
        """
        Analyze sentiment using zero-shot classification with enhanced processing.

        Args:
            text: Input text to analyze

        Returns:
            SentimentResponse: Detailed sentiment analysis results

        Raises:
            APIConnectionError: For connection issues
            InvalidAPIResponseError: For invalid responses
        """
        prompt = self._construct_sentiment_prompt(text)
        payload = {"inputs": prompt, "parameters": self.models["sentiment"].parameters}

        try:
            async with self.session.post(
                self.sentiment_endpoint, json=payload
            ) as response:
                await self._handle_response_status(response)
                result = await response.json()

                return await self._process_sentiment_response(result)

        except aiohttp.ClientError as e:
            raise APIConnectionError(
                "Failed to connect to sentiment analysis service", {"error": str(e)}
            )

    async def _process_sentiment_response(
        self, result: Dict[str, Any]
    ) -> SentimentResponse:
        """
        Process and validate sentiment analysis response.

        Args:
            result: Raw API response

        Returns:
            SentimentResponse: Processed response

        Raises:
            InvalidAPIResponseError: For invalid response format
        """
        try:
            labels = result.get("labels", [])
            scores = result.get("scores", [])

            if not labels or not scores or len(labels) != len(scores):
                raise InvalidAPIResponseError(
                    "Invalid sentiment response format", {"response": result}
                )

            # Create emotion scores dictionary
            emotion_scores = dict(zip(labels, scores))

            # Find dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])

            # Calculate overall sentiment score
            sentiment_score = self._calculate_sentiment_score(emotion_scores)

            return SentimentResponse(
                score=sentiment_score,
                is_concerning=sentiment_score <= -0.7,
                requires_support=sentiment_score <= -0.5,
                confidence=dominant_emotion[1],
                dominant_emotion=dominant_emotion[0],
                emotion_scores=emotion_scores,
            )

        except Exception as e:
            raise InvalidAPIResponseError(
                "Failed to process sentiment response",
                {"error": str(e), "response": result},
            )

    def _calculate_sentiment_score(self, emotion_scores: Dict[str, float]) -> float:
        """
        Calculate weighted sentiment score from emotion scores.

        Args:
            emotion_scores: Dictionary of emotions and their scores

        Returns:
            float: Calculated sentiment score between -1 and 1
        """
        # Define emotion valence weights
        positive_emotions = {
            "joy": 1.0,
            "contentment": 0.8,
            "gratitude": 0.9,
            "hope": 0.7,
            "pride": 0.6,
            "love": 1.0,
            "awe": 0.8,
        }
        negative_emotions = {
            "anxiety": -0.7,
            "sadness": -0.8,
            "anger": -0.9,
            "fear": -0.8,
            "shame": -0.9,
            "loneliness": -0.7,
        }

        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0

        for emotion, score in emotion_scores.items():
            if emotion in positive_emotions:
                weight = positive_emotions[emotion]
            elif emotion in negative_emotions:
                weight = negative_emotions[emotion]
            else:
                weight = 0.0

            total_score += score * weight
            total_weight += abs(score)

        if total_weight == 0:
            return 0.0

        normalized_score = total_score / total_weight
        return max(min(normalized_score, 1.0), -1.0)

    async def _handle_response_status(self, response: aiohttp.ClientResponse) -> None:
        """
        Handle API response status with detailed error reporting.

        Args:
            response: aiohttp response object

        Raises:
            APIEndpointUnavailableError: For service unavailability
            APIConnectionError: For other HTTP errors
        """
        if response.status != 200:
            response_text = await response.text()

            if response.status == 503:
                raise APIEndpointUnavailableError(
                    "Service temporarily unavailable",
                    {"status": response.status, "response": response_text},
                )
            else:
                raise APIConnectionError(
                    f"Request failed with status {response.status}",
                    {"status": response.status, "response": response_text},
                )

    @retry(
        retry=retry_if_exception_type(APIConnectionError),
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=RETRY_WAIT,
    )
    async def _generate_chat_token(self, prompt: str, emotion: str) -> str:
        """
        Generate a chat token with enhanced error handling and validation.

        Args:
            prompt: Generation prompt
            emotion: Selected emotion state

        Returns:
            str: Generated token

        Raises:
            APIConnectionError: For connection issues
            InvalidAPIResponseError: For invalid responses
        """
        generation_params = self._apply_generation_parameters(emotion)

        payload = {"inputs": prompt, "parameters": generation_params}

        try:
            async with self.session.post(self.chat_endpoint, json=payload) as response:
                await self._handle_response_status(response)
                result = await response.json()

                return self._process_generation_response(result)

        except aiohttp.ClientError as e:
            raise APIConnectionError(
                "Failed to connect to generation service", {"error": str(e)}
            )

    def _process_generation_response(self, result: Any) -> str:
        """
        Process and validate generation response.

        Args:
            result: Raw API response

        Returns:
            str: Processed token

        Raises:
            InvalidAPIResponseError: For invalid response format
        """
        if not isinstance(result, list) or not result:
            raise InvalidAPIResponseError(
                "Invalid generation response format", {"response": result}
            )

        token = result[0].get("generated_text", "")
        if not token:
            raise InvalidAPIResponseError(
                "Empty generation response", {"response": result}
            )

        # Post-process the token to clean and format it
        return self._post_process_token(token)

    def _post_process_token(self, token: str) -> str:
        """
        Clean and format the generated token with enhanced processing.

        Args:
            token: Raw generated token

        Returns:
            str: Processed token
        """
        # Remove any system/user prompts
        markers = ["<|system|>", "<|user|>", "<|assistant|>"]
        for marker in markers:
            if marker in token:
                token = token.split(marker)[-1]

        # Clean whitespace and formatting
        token = token.strip()

        # Capitalize first letter if necessary
        if token and not token[0].isupper():
            token = token[0].upper() + token[1:]

        # Ensure emoji presence if required
        if self.prompts.token_generation_rules.style_guide.use_emojis:
            if not self._contains_emoji(token):
                token = "✨ " + token

        # Enforce length limits
        max_length = self.prompts.token_generation_rules.max_length
        if len(token) > max_length:
            token = token[: max_length - 3] + "..."

        return token

    def _contains_emoji(self, text: str) -> bool:
        """
        Check if text contains at least one emoji.

        Args:
            text: Input text

        Returns:
            bool: True if contains emoji, False otherwise
        """
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            "\U0001F680-\U0001F6FF"  # Transport & Map
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return bool(emoji_pattern.search(text))

    def _apply_generation_parameters(self, emotion: str) -> Dict[str, Any]:
        """
        Apply emotion-specific generation parameters with validation and stop sequence.

        Args:
            emotion: Selected emotion state

        Returns:
            Dict[str, Any]: Generation parameters including 'stop' sequence
        """
        base_params = self.prompts.emotions[emotion].generation_parameters

        # Ensure required parameters are present
        required_params = {
            "max_new_tokens": self.prompts.token_generation_rules.max_length,
            "temperature": base_params.get("temperature", 0.7),
            "top_p": base_params.get("top_p", 0.9),
            "repetition_penalty": base_params.get("repetition_penalty", 1.2),
            "do_sample": base_params.get("do_sample", True),
            "stop": ["<|user|>"],  # Add stop sequence to prevent echoing prompt
            "return_full_text": False,  # Ensure only the generated part is returned
        }

        # Validate and adjust parameters
        if required_params["temperature"] < 0.1 or required_params["temperature"] > 1.0:
            logger.warning(
                f"Invalid temperature value: {required_params['temperature']}, using default 0.7"
            )
            required_params["temperature"] = 0.7

        return required_params

    async def generate_light_token(
        self, entry: str, emotion: str, retries: int = 2
    ) -> Tuple[str, Optional[str]]:
        """
        Generate a light token with comprehensive processing and validation.

        This method coordinates the complete token generation process, including
        sentiment analysis, prompt construction, and token generation with
        fallback mechanisms.

        Args:
            entry: Journal entry text
            emotion: Selected emotion state
            retries: Number of generation retries allowed

        Returns:
            Tuple[str, Optional[str]]: Generated token and optional support message

        Raises:
            LightBearerException: For various failure modes
        """
        try:
            # Validate inputs
            validated_data = self._validate_and_transform_input(entry, emotion)

            # Analyze sentiment with fallback
            sentiment, using_sentiment_fallback = (
                await self._analyze_sentiment_with_fallback(validated_data["entry"])
            )

            # Get support message if needed
            support_message = None
            if sentiment.requires_support:
                support_message = self._get_support_message(sentiment.score)

            # Generate token with fallback if sentiment was successful
            if not using_sentiment_fallback:
                token, using_generation_fallback = (
                    await self._generate_token_with_fallback(
                        self._construct_generation_prompt(
                            entry=validated_data["entry"],
                            emotion=validated_data["emotion"],
                            sentiment=sentiment,
                        ),
                        emotion,
                    )
                )
            else:
                token = self._get_emotion_specific_fallback(emotion)
                using_generation_fallback = True

            return (
                token,
                support_message,
                (using_sentiment_fallback or using_generation_fallback),
            )

        except Exception as e:
            logger.error(f"Unexpected error in token generation: {str(e)}")
            logger.debug(traceback.format_exc())
            return (
                self._get_emotion_specific_fallback(emotion),
                "Remember to be gentle with yourself during this time.",
                True,
            )

    async def _analyze_sentiment_with_fallback(
        self, text: str
    ) -> Tuple[SentimentResponse, bool]:
        """
        Analyze sentiment with fallback handling when service is unavailable.

        Args:
            text: Input text to analyze

        Returns:
            Tuple[SentimentResponse, bool]: (sentiment_response, using_fallback)
        """
        if (
            not self.service_status.sentiment_available
            and not self.service_status.can_retry()
        ):
            logger.info(
                "Using fallback sentiment analysis due to service unavailability"
            )
            return self._get_fallback_sentiment(), True

        try:
            sentiment = await self._analyze_sentiment_zero_shot(text)
            await self.service_status.mark_service_available("sentiment")
            return sentiment, False

        except APIEndpointUnavailableError:
            await self.service_status.mark_service_unavailable("sentiment")
            return self._get_fallback_sentiment(), True

    def _get_support_message(self, sentiment_score: float) -> Optional[str]:
        """
        Generate an appropriate support message based on sentiment score.

        Args:
            sentiment_score: The analyzed sentiment score

        Returns:
            Optional[str]: A support message if needed, None otherwise
        """
        for level, support in self.prompts.support_messages.items():
            if sentiment_score <= support.threshold:
                message = random.choice(support.messages)
                logger.debug(f"Selected support message for level {level}: {message}")
                return message
        return None

    async def _generate_token_with_fallback(
        self, prompt: str, emotion: str
    ) -> Tuple[str, bool]:
        """
        Generate token with fallback handling when service is unavailable.

        Args:
            prompt: Generation prompt
            emotion: Selected emotion state

        Returns:
            Tuple[str, bool]: (generated_token, using_fallback)
        """
        if (
            not self.service_status.generation_available
            and not self.service_status.can_retry()
        ):
            logger.info("Using fallback token generation due to service unavailability")
            return self._get_emotion_specific_fallback(emotion), True

        try:
            token = await self._generate_chat_token(prompt, emotion)
            await self.service_status.mark_service_available("generation")
            return token, False

        except APIEndpointUnavailableError:
            await self.service_status.mark_service_unavailable("generation")
            return self._get_emotion_specific_fallback(emotion), True

    def _get_fallback_sentiment(self) -> SentimentResponse:
        """Generate a balanced fallback sentiment response."""
        return SentimentResponse(
            score=0.0,
            is_concerning=False,
            requires_support=False,
            confidence=1.0,
            dominant_emotion="neutral",
            emotion_scores={"neutral": 1.0},
            timestamp=datetime.utcnow(),
        )

    def _get_emotion_specific_fallback(self, emotion: str) -> str:
        """Get contextually appropriate fallback message based on emotion."""
        emotion_fallbacks = {
            "Dawn": "✨ Each new dawn brings fresh opportunities for growth and healing.",
            "Twilight": "🌅 In this moment of transition, embrace the gentle wisdom that comes with reflection.",
            "Midnight": "🌟 Even in darkness, your inner strength remains an unwavering beacon of hope.",
            "Noon": "☀️ Stand confident in your clarity and purpose, knowing each step forward carries meaning.",
        }

        return emotion_fallbacks.get(
            emotion,
            "✨ Your journey continues with strength and wisdom, each moment holding potential for transformation.",
        )

    def _get_fallback_token(self) -> str:
        """
        Get a contextually appropriate fallback token.

        Returns:
            str: Fallback token
        """
        fallback_tokens = [
            "✨ Each step forward carries the light of possibility, illuminating new paths of growth and understanding.",
            "🌟 In life's gentle ebb and flow, your strength shines as a beacon of resilience and hope.",
            "💫 Like stars emerging in twilight, wisdom often reveals itself in moments of reflection.",
        ]
        return random.choice(fallback_tokens)

    def _validate_generated_token(self, token: str) -> bool:
        """
        Validate generated token against quality criteria.

        Args:
            token: Generated token

        Returns:
            bool: True if valid, False otherwise
        """
        # Check minimum length
        if len(token.split()) < self.prompts.token_generation_rules.min_length:
            return False

        # Check maximum length
        if len(token) > self.prompts.token_generation_rules.max_length:
            return False

        # Check emoji presence if required
        if (
            self.prompts.token_generation_rules.require_emoji
            and not self._contains_emoji(token)
        ):
            return False

        # Check for common quality issues
        quality_checks = [
            lambda t: len(t.split()) >= 10,  # Minimum word count
            lambda t: not any(
                marker in t.lower() for marker in ["error", "invalid", "failed"]
            ),
            lambda t: t.strip() != "",  # Non-empty
            lambda t: len(set(t.split())) >= 7,  # Vocabulary diversity
        ]

        return all(check(token) for check in quality_checks)

    def _validate_and_transform_input(self, entry: str, emotion: str) -> Dict[str, Any]:
        """
        Validate and transform input data with comprehensive checks.

        Args:
            entry: Journal entry text
            emotion: Selected emotion state

        Returns:
            Dict[str, Any]: Validated and transformed data

        Raises:
            LightBearerException: For validation failures
        """
        # Validate entry
        if not entry or not entry.strip():
            raise LightBearerException(
                "Journal entry cannot be empty", {"entry": entry}
            )

        # Validate emotion
        if emotion not in self.prompts.emotions:
            raise LightBearerException(
                f"Invalid emotion state: {emotion}",
                {
                    "emotion": emotion,
                    "valid_emotions": list(self.prompts.emotions.keys()),
                },
            )

        # Check entry length
        max_length = 2000  # Adjust as needed
        if len(entry) > max_length:
            raise LightBearerException(
                f"Journal entry exceeds maximum length of {max_length} characters",
                {"entry_length": len(entry), "max_length": max_length},
            )

        # Basic content validation
        if len(entry.split()) < 3:
            raise LightBearerException(
                "Journal entry must contain at least 3 words",
                {"entry_word_count": len(entry.split())},
            )

        # Transform and sanitize input
        sanitized_entry = " ".join(entry.split())  # Normalize whitespace

        return {"entry": sanitized_entry, "emotion": emotion}
