# src/llm_service.py

import os
import json
import asyncio
import logging
import traceback
from typing import Optional, Tuple, Dict, Any

import aiohttp
from pydantic import BaseModel, ValidationError, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    RetryError,
)
from dotenv import load_dotenv
import random

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for more detailed logs

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


# Exception Classes
class LightBearerException(Exception):
    """Base exception for LightBearer service."""
    pass


class APIConnectionError(LightBearerException):
    """Exception raised for API connection errors."""
    pass


class APIEndpointUnavailableError(LightBearerException):
    """Exception raised when the API endpoint is unavailable."""
    pass


class InvalidAPIResponseError(LightBearerException):
    """Exception raised for invalid API responses."""
    pass


# Configuration Models
class EmotionState(BaseModel):
    """Emotion State Schema."""
    name: str
    description: str
    context: str
    transformation_patterns: list[str]
    generation_parameters: Dict[str, Any]


class SupportMessage(BaseModel):
    """Support Message Schema."""
    threshold: float
    messages: list[str]


class TransformationPrinciple(BaseModel):
    """Transformation Principle Schema."""
    acknowledgment: str
    potential: str
    agency: str
    connection: str
    emergence: str


class StyleGuide(BaseModel):
    """Style Guide Schema."""
    tone: str
    language: str
    format: str
    use_emojis: bool  # Correctly set as a boolean


class TokenGenerationRules(BaseModel):
    """Token Generation Rules Schema."""
    max_length: int
    style_guide: StyleGuide  # Changed from Dict[str, str] to StyleGuide


class PromptsConfig(BaseModel):
    """Prompts Configuration Schema."""
    transformation_prompt: str
    healing_prompt: str
    sentiment_prompt: str
    emotions: Dict[str, EmotionState]
    support_messages: Dict[str, SupportMessage]
    transformation_principles: TransformationPrinciple
    token_generation_rules: TokenGenerationRules


# Response Models
class SentimentResponse(BaseModel):
    """Sentiment Analysis Response Schema."""
    score: float
    is_concerning: bool
    requires_support: bool


# LightBearer Service Class
class LightBearer:
    """
    LightBearer Service Class.

    Handles interactions with Hugging Face's Inference Endpoints for sentiment analysis
    and text-based token generation.
    """

    RETRY_ATTEMPTS: int = 5
    RETRY_WAIT = wait_exponential(multiplier=1, min=2, max=20)

    def __init__(self, config_path: str = "assets/prompts/light_seeds.json"):
        """
        Initialize the LightBearer service.

        Args:
            config_path (str): Path to the prompts configuration JSON file.
        """
        # Load environment variables
        self.api_token: str = os.getenv("HUGGINGFACE_API_TOKEN")
        self.chat_endpoint: str = os.getenv("CHAT_MODEL_ENDPOINT")
        self.chat_pipeline: str = os.getenv("CHAT_MODEL_PIPELINE", "text-generation")  # 'text-generation'
        self.sentiment_endpoint: str = os.getenv("SENTIMENT_MODEL_ENDPOINT")
        self.sentiment_pipeline: str = os.getenv("SENTIMENT_MODEL_PIPELINE", "zero-shot-classification")  # e.g., 'zero-shot-classification'
        self.sentiment_threshold: float = -0.5  # Example threshold for determining support message necessity
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_sentiment_score: Optional[float] = None

        # Validate environment variables
        if not all([self.api_token, self.chat_endpoint, self.sentiment_endpoint]):
            logger.error("One or more required environment variables are missing.")
            raise LightBearerException("Missing required environment variables.")

        # Load prompts configuration
        try:
            with open(config_path, "r") as f:
                prompts_data = json.load(f)
            self.prompts: PromptsConfig = PromptsConfig(**prompts_data)
            logger.debug("Loaded prompts configuration successfully.")
            # Log pipeline types
            logger.debug(f"Chat Pipeline: {self.chat_pipeline}")
            logger.debug(f"Sentiment Pipeline: {self.sentiment_pipeline}")
        except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to load prompts configuration: {e}")
            logger.debug(traceback.format_exc())
            raise LightBearerException("Invalid prompts configuration.") from e

        # Define sentiment model configuration
        self.sentiment_model_config: Dict[str, Any] = {
            "candidate_labels": ["happiness", "sadness", "neutral"],
            "multi_label": False
        }

    async def __aenter__(self):
        """Enter the asynchronous context manager."""
        await self._init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the asynchronous context manager."""
        await self._close_session()

    async def _init_session(self) -> None:
        """Initialize the aiohttp ClientSession."""
        if not self.session:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            self.session = aiohttp.ClientSession(headers=headers)
            logger.debug("Initialized aiohttp ClientSession.")

    async def _close_session(self) -> None:
        """Close the aiohttp ClientSession."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.debug("Closed aiohttp ClientSession.")

    def _post_process_token(self, token: str) -> str:
        """Clean and format the generated token."""
        original_token = token  # Preserve the original for debugging
        # Remove any system/user prompts that might have been generated
        for marker in ["<|system|>", "<|user|>", "<|assistant|>"]:
            if marker in token:
                token = token.split(marker)[-1]

        # Clean up whitespace and ensure proper formatting
        token = token.strip()
        if token:
            token = token[0].upper() + token[1:]

        # Ensure token isn't too long
        if len(token) > self.prompts.token_generation_rules.max_length:
            token = token[:self.prompts.token_generation_rules.max_length - 3] + "..."

        # Log the original and processed tokens
        logger.debug(f"Original token: {original_token[:100]}...")
        logger.debug(f"Processed token: {token[:50]}...")  # Only log the first 50 characters

        return token

    def _get_support_message(self, sentiment_score: float) -> str:
        """Generate an appropriate support message based on sentiment."""
        for level, support in self.prompts.support_messages.items():
            if sentiment_score <= support.threshold:
                # Select a random message from the applicable support messages
                message = random.choice(support.messages)
                logger.debug(f"Selected support message: {message}")
                return message
        return ""

    def _validate_and_transform_input(
        self,
        entry: str,
        emotion: str
    ) -> Dict[str, Any]:
        """
        Validate and transform input data.

        Args:
            entry (str): Journal entry text.
            emotion (str): Selected emotion.

        Returns:
            Dict[str, Any]: Validated and transformed data.

        Raises:
            LightBearerException: If validation fails.
        """
        if emotion not in self.prompts.emotions:
            logger.error(f"Invalid emotion state: {emotion}")
            raise LightBearerException(f"Invalid emotion state: {emotion}")

        # Additional validation can be added here if necessary
        return {
            "entry": entry,
            "emotion": emotion
        }

    def _construct_chat_prompt(
        self,
        entry: str,
        emotion: str,
        sentiment_score: float
    ) -> str:
        """
        Construct an enhanced prompt for chat token generation.

        Args:
            entry (str): Journal entry text.
            emotion (str): Selected emotional state.
            sentiment_score (float): Analyzed sentiment score.

        Returns:
            str: Constructed prompt string.
        """
        emotion_context = self.prompts.emotions[emotion].context
        patterns = self.prompts.emotions[emotion].transformation_patterns

        # Select appropriate transformation pattern based on sentiment
        index = min(
            max(int((sentiment_score + 1) / 2 * len(patterns)), 0),
            len(patterns) - 1
        )
        selected_pattern = patterns[index]

        # Determine emoji instruction based on configuration
        if self.prompts.token_generation_rules.style_guide.use_emojis:
            emoji_instruction = " Include at least one ✨ emoji in the token."
        else:
            emoji_instruction = ""

        # Populate the transformation prompt with emoji instruction if applicable
        transformation_prompt = self.prompts.transformation_prompt.format(
            emotion_context=emotion_context,
            entry=entry
        ) + emoji_instruction

        logger.debug(f"Constructed chat prompt: {transformation_prompt[:100]}...")
        return transformation_prompt


    @retry(
        retry=retry_if_exception_type(APIConnectionError),
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=RETRY_WAIT
    )
    async def _analyze_sentiment_zero_shot(self, text: str) -> SentimentResponse:
        """
        Analyze the sentiment of input text using a zero-shot classification model.

        Args:
            text (str): Input text to analyze.

        Returns:
            SentimentResponse: Containing analysis results.

        Raises:
            APIConnectionError: If the API call fails.
            InvalidAPIResponseError: If the response is invalid.
        """
        payload = {
            "inputs": text,
            "parameters": self.sentiment_model_config
        }

        await self._init_session()

        try:
            async with self.session.post(
                self.sentiment_endpoint,
                json=payload,
                timeout=60  # Adjust timeout as needed
            ) as response:
                if response.status != 200:
                    response_text = await response.text()
                    if response.status == 503:
                        logger.error(f"Sentiment endpoint unavailable: {response_text}")
                        raise APIEndpointUnavailableError(
                            "The sentiment analysis endpoint is currently unavailable. Please try again later."
                        )
                    else:
                        logger.error(f"Sentiment analysis request failed with status {response.status}: {response_text}")
                        raise APIConnectionError(
                            f"Sentiment analysis request failed with status {response.status}: {response_text}"
                        )

                result = await response.json()

                try:
                    sequence = result.get("sequence", "")
                    labels = result.get("labels", [])
                    scores = result.get("scores", [])

                    if not labels or not scores or len(labels) != len(scores):
                        raise ValueError("Labels and scores mismatch or are empty.")

                    # Identify the label with the highest score
                    top_index = scores.index(max(scores))
                    top_label = labels[top_index]
                    top_score = scores[top_index]

                    # Map labels to sentiment scores
                    label_score_mapping = {
                        "happiness": 1.0,
                        "neutral": 0.0,
                        "sadness": -1.0
                    }

                    score = label_score_mapping.get(top_label.lower(), 0.0)

                except (KeyError, IndexError, ValueError) as e:
                    logger.error(f"Error parsing sentiment classification: {e}")
                    logger.debug(traceback.format_exc())
                    raise InvalidAPIResponseError(
                        "Unable to parse sentiment classification from the API response."
                    ) from e

                is_concerning = score <= self.sentiment_threshold
                requires_support = score <= -0.5

                logger.debug(f"Sentiment analysis result: {score}, is_concerning: {is_concerning}, requires_support: {requires_support}")

                return SentimentResponse(
                    score=score,
                    is_concerning=is_concerning,
                    requires_support=requires_support
                )

        except aiohttp.ClientError as e:
            logger.error(f"Client error during sentiment analysis: {e}")
            logger.debug(traceback.format_exc())
            raise APIConnectionError(
                "Failed to connect to the sentiment analysis service."
            ) from e
        except asyncio.TimeoutError:
            logger.error("Sentiment analysis request timed out.")
            logger.debug(traceback.format_exc())
            raise APIConnectionError(
                "The sentiment analysis request timed out."
            )

    @retry(
        retry=retry_if_exception_type(APIConnectionError),
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=RETRY_WAIT
    )
    async def _generate_chat_token(self, prompt: str, emotion: str) -> str:
        """
        Send prompt to the text-generation API and retrieve the generated token.

        Args:
            prompt (str): The generation prompt.
            emotion (str): Selected emotional state.

        Returns:
            str: Generated token string.

        Raises:
            APIConnectionError: If API call fails.
            InvalidAPIResponseError: If response is invalid.
        """
        # Retrieve emotion-specific generation parameters
        generation_params = self._apply_generation_parameters(emotion)

        # Determine payload based on pipeline type
        if self.chat_pipeline.lower() == "text-generation":
            payload = {
                "inputs": prompt,
                "parameters": generation_params
            }
        else:
            logger.error(f"Unsupported chat pipeline type: {self.chat_pipeline}")
            raise LightBearerException(f"Unsupported chat pipeline type: {self.chat_pipeline}")

        logger.debug(f"Sending request to {self.chat_endpoint} with payload: {payload}")

        try:
            async with self.session.post(
                self.chat_endpoint,
                json=payload,
                timeout=60  # Adjust timeout as needed
            ) as response:
                if response.status != 200:
                    response_text = await response.text()
                    if response.status == 503:
                        logger.error(f"Chat endpoint unavailable: {response_text}")
                        raise APIEndpointUnavailableError(
                            "The chat token generation endpoint is currently unavailable. Please try again later."
                        )
                    else:
                        logger.error(f"Chat request failed with status {response.status}: {response_text}")
                        raise APIConnectionError(
                            f"Chat request failed with status {response.status}: {response_text}"
                        )

                response_data = await response.json()
                logger.debug(f"Received response from model: {response_data}")

                # Handle response format for text-generation
                if self.chat_pipeline.lower() == "text-generation":
                    if not isinstance(response_data, list) or not response_data:
                        logger.error("Invalid response format from text-generation API.")
                        raise InvalidAPIResponseError(
                            "Received invalid response format from the text-generation API."
                        )
                    token = self._post_process_token(response_data[0].get("generated_text", ""))
                else:
                    logger.error(f"Unsupported chat pipeline type: {self.chat_pipeline}")
                    raise LightBearerException(f"Unsupported chat pipeline type: {self.chat_pipeline}")

                logger.debug(f"Generated chat token: {token[:100]}...")

                return token

        except aiohttp.ClientError as e:
            logger.error(f"Client error during chat token generation: {e}")
            logger.debug(traceback.format_exc())
            raise APIConnectionError(
                "Failed to connect to the chat token generation service."
            ) from e
        except asyncio.TimeoutError:
            logger.error("Chat token generation request timed out.")
            logger.debug(traceback.format_exc())
            raise APIConnectionError(
                "The chat token generation request timed out."
            )

    @retry(
        retry=retry_if_exception_type(APIConnectionError),
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=RETRY_WAIT
    )
    async def _regenerate_chat_token(
        self,
        prompt: str,
        emotion: str,
        attempt: int = 1,
        max_attempts: int = 3
    ) -> str:
        """
        Regenerate chat token if initial response is too short or inadequate.

        Args:
            prompt (str): The generation prompt.
            emotion (str): Selected emotional state.
            attempt (int): Current attempt number.
            max_attempts (int): Maximum number of attempts.

        Returns:
            str: Generated token.

        Raises:
            APIConnectionError: If API call fails after retries.
        """
        if attempt > max_attempts:
            default_token = (
                "🌟✨ Embrace your journey with courage and hope, knowing that each step forward brings you closer to brighter horizons."
            )
            logger.info("Max regeneration attempts reached. Returning default token.")
            return default_token

        # Adjust temperature slightly for each retry to introduce variability
        temp_adjust = 0.05 * attempt
        modified_config = self._apply_generation_parameters(emotion)
        modified_config["temperature"] = min(modified_config["temperature"] + temp_adjust, 1.0)  # Cap at 1.0

        logger.debug(f"Regeneration attempt {attempt} with temperature {modified_config['temperature']}.")

        # Determine payload based on pipeline type
        if self.chat_pipeline.lower() == "text-generation":
            payload = {
                "inputs": prompt,
                "parameters": modified_config
            }
        else:
            logger.error(f"Unsupported chat pipeline type: {self.chat_pipeline}")
            raise LightBearerException(f"Unsupported chat pipeline type: {self.chat_pipeline}")

        logger.debug(f"Sending regeneration request with payload: {payload}")

        try:
            async with self.session.post(
                self.chat_endpoint,
                json=payload,
                timeout=60  # Adjust timeout as needed
            ) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logger.error(f"Chat regeneration failed with status {response.status}: {response_text}")
                    raise APIConnectionError(
                        f"Chat regeneration failed with status {response.status}: {response_text}"
                    )

                response_data = await response.json()
                logger.debug(f"Received regeneration response from model: {response_data}")

                # Handle response format for text-generation
                if self.chat_pipeline.lower() == "text-generation":
                    if not isinstance(response_data, list) or not response_data:
                        logger.error("Invalid response format from text-generation API during regeneration.")
                        raise InvalidAPIResponseError(
                            "Received invalid response format from the text-generation API during regeneration."
                        )
                    token = self._post_process_token(response_data[0].get("generated_text", ""))
                else:
                    logger.error(f"Unsupported chat pipeline type: {self.chat_pipeline}")
                    raise LightBearerException(f"Unsupported chat pipeline type: {self.chat_pipeline}")

                if len(token.split()) >= 20 and "✨" in token:
                    logger.info(f"Regenerated chat token on attempt {attempt}: {token[:100]}...")
                    return token
                else:
                    logger.warning(f"Regenerated chat token too short or inadequate on attempt {attempt}. Retrying...")
                    return await self._regenerate_chat_token(prompt, emotion, attempt + 1, max_attempts)

        except aiohttp.ClientError as e:
            logger.error(f"Client error during chat token regeneration: {e}")
            logger.debug(traceback.format_exc())
            raise APIConnectionError(
                "Failed to connect to the chat token regeneration service."
            ) from e
        except asyncio.TimeoutError:
            logger.error("Chat token regeneration request timed out.")
            logger.debug(traceback.format_exc())
            raise APIConnectionError(
                "The chat token regeneration request timed out."
            )

    def _apply_generation_parameters(self, emotion: str) -> Dict[str, Any]:
        """
        Retrieve and apply generation parameters based on the selected emotion.

        Args:
            emotion (str): Selected emotional state.

        Returns:
            Dict[str, Any]: Generation parameters.
        """
        params = self.prompts.emotions[emotion].generation_parameters
        return {
            "max_new_tokens": self.prompts.token_generation_rules.max_length,
            "temperature": params.get("temperature", 0.6),
            "do_sample": params.get("do_sample", False),
            "top_p": params.get("top_p", 0.7),
            "repetition_penalty": params.get("repetition_penalty", 3.0)
        }

    async def generate_light_token(
        self,
        entry: str,
        emotion: str
    ) -> Tuple[str, Optional[str]]:
        """
        Generate a light token and optional support message.

        This method processes user input, analyzes sentiment, and generates a meaningful
        token of light using the text-generation model. It includes validation,
        sentiment analysis, and fallback mechanisms for ensuring quality responses.

        Args:
            entry (str): Journal entry text.
            emotion (str): Selected emotion.

        Returns:
            Tuple[str, Optional[str]]: Generated token and optional support message.

        Raises:
            LightBearerException: For various failure modes.
        """
        try:
            # Validate and transform input
            validated_data = self._validate_and_transform_input(entry, emotion)

            # Analyze sentiment
            sentiment = await self._analyze_sentiment_zero_shot(validated_data["entry"])

            # Get support message if required
            support_message = self._get_support_message(sentiment.score) if sentiment.requires_support else None

            # Construct prompt for chat token generation
            prompt = self._construct_chat_prompt(
                entry=validated_data["entry"],
                emotion=validated_data["emotion"],
                sentiment_score=sentiment.score
            )

            # Generate chat token
            token = await self._generate_chat_token(prompt, emotion)

            # Validate the generated token
            if not self._validate_generated_token(token):
                logger.warning("Generated token did not meet criteria. Attempting regeneration.")
                token = await self._regenerate_chat_token(prompt, emotion)

            # Final validation
            if not self._validate_generated_token(token):
                logger.warning("Regenerated token still did not meet criteria. Using fallback.")
                token = "🌟✨ Embrace your journey with courage and hope, knowing that each step forward brings you closer to brighter horizons."

            # Store the token in session state
            # (Assuming you have a session state or similar mechanism)
            # e.g., st.session_state.app_state['light_tokens'].append(token)

            # Optionally, store in the database
            # await self.db_manager.add_journal_entry(content, token, st.session_state.app_state['current_emotion'])

            return token, support_message

        except RetryError as e:
            logger.error(f"Max retries exceeded: {e}")
            logger.debug(traceback.format_exc())
            raise APIConnectionError(
                "Failed to generate token after multiple attempts."
            ) from e
        except LightBearerException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in token generation: {e}")
            logger.debug(traceback.format_exc())
            raise LightBearerException(
                "An unexpected error occurred while generating the token."
            ) from e

    def _validate_generated_token(self, token: str) -> bool:
        """
        Validate that the generated token meets minimum criteria.

        Args:
            token (str): Generated token.

        Returns:
            bool: True if valid, False otherwise.
        """
        # Check for minimum word count
        if len(token.split()) < 20:
            return False

        # Check for at least one emoji
        if self.prompts.token_generation_rules.style_guide.use_emojis:
            emoji_pattern = re.compile(
                "[" 
                "\U0001F600-\U0001F64F"  # Emoticons
                "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                "\U0001F680-\U0001F6FF"  # Transport & Map
                "\U0001F1E0-\U0001F1FF"  # Flags
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE
            )
            if not emoji_pattern.search(token):
                return False
        return True

