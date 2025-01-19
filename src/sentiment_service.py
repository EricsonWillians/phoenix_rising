import re
import math
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SentimentIntensityAnalyzer:
    """
    A professional implementation of sentiment intensity analyzer using VADER-inspired approach.
    This class provides methods for analyzing sentiment in text, with particular attention to
    emotional content and intensity.
    """

    def __init__(self):
        """Initialize the SentimentIntensityAnalyzer with lexicon and rules."""
        self._lexicon = self._initialize_lexicon()
        self._emoji_lexicon = self._initialize_emoji_lexicon()
        
        # Special case idioms
        self._special_case_idioms = {
            "the bomb": 3, "the shit": 3, "the best": 3, "cut deep": -2,
            "fall apart": -2, "falling apart": -2, "blown away": 2,
            "not good": -2, "not great": -2
        }
        
        # Booster word scores
        self._booster_dict = {
            "absolutely": 0.293, "amazingly": 0.293, "completely": 0.293,
            "considerably": 0.293, "decidedly": 0.293, "deeply": 0.293,
            "enormously": 0.293, "entirely": 0.293, "especially": 0.293,
            "exceptionally": 0.293, "extremely": 0.293, "fantastically": 0.293,
            "fully": 0.293, "greatly": 0.293, "highly": 0.293, "hugely": 0.293,
            "incredibly": 0.293, "intensely": 0.293, "majorly": 0.293,
            "more": 0.293, "most": 0.293, "particularly": 0.293,
            "purely": 0.293, "quite": 0.293, "really": 0.293,
            "remarkably": 0.293, "so": 0.293, "substantially": 0.293,
            "thoroughly": 0.293, "totally": 0.293, "tremendously": 0.293,
            "uber": 0.293, "unbelievably": 0.293, "unusually": 0.293,
            "utterly": 0.293, "very": 0.293, "almost": -0.293,
            "barely": -0.293, "hardly": -0.293, "just enough": -0.293,
            "kind of": -0.293, "kinda": -0.293, "kindof": -0.293,
            "kind-of": -0.293, "less": -0.293, "little": -0.293,
            "marginally": -0.293, "occasionally": -0.293, "partly": -0.293,
            "scarcely": -0.293, "slightly": -0.293, "somewhat": -0.293,
            "sort of": -0.293, "sorta": -0.293, "sortof": -0.293,
            "sort-of": -0.293
        }

    def _initialize_lexicon(self) -> Dict[str, float]:
        """Initialize the sentiment lexicon with word-sentiment pairs."""
        # Core emotional words with their sentiment scores
        return {
            # Positive emotions
            "good": 1.9, "great": 2.3, "excellent": 2.7,
            "happy": 2.1, "joy": 2.4, "wonderful": 2.6,
            "amazing": 2.5, "love": 2.8, "fantastic": 2.4,
            "beautiful": 2.0, "peaceful": 1.8, "calm": 1.5,
            
            # Negative emotions
            "bad": -1.9, "terrible": -2.3, "horrible": -2.7,
            "sad": -2.1, "angry": -2.4, "awful": -2.6,
            "poor": -1.8, "hate": -2.8, "disappointed": -1.9,
            "anxious": -1.7, "afraid": -2.0, "depressed": -2.5,
            
            # Additional emotional nuances
            "grateful": 2.2, "blessed": 2.1, "hopeful": 1.9,
            "inspired": 2.0, "proud": 1.8, "confident": 1.7,
            "worried": -1.6, "stressed": -1.8, "overwhelmed": -1.9,
            "frustrated": -1.7, "confused": -1.4, "lonely": -2.1
        }

    def _initialize_emoji_lexicon(self) -> Dict[str, float]:
        """Initialize emoji sentiment scores."""
        return {
            "😊": 1.8, "😃": 2.0, "😄": 2.0, "😁": 1.9,
            "🥰": 2.5, "😍": 2.4, "❤️": 2.3, "💕": 2.2,
            "😢": -1.8, "😭": -2.0, "😔": -1.7, "😟": -1.6,
            "😡": -2.2, "😠": -2.0, "😤": -1.9, "😞": -1.8,
            "✨": 1.5, "🌟": 1.6, "💫": 1.4, "🌈": 1.7
        }

    def polarity_scores(self, text: str) -> Dict[str, float]:
        """
        Calculate sentiment polarity scores for the input text.

        Args:
            text: Input text for sentiment analysis

        Returns:
            Dict containing positive, negative, neutral, and compound scores
        """
        try:
            if not text or not isinstance(text, str):
                logger.warning("Invalid input text provided")
                return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}

            # Tokenize and clean text
            words = self._tokenize(text.lower())
            
            # Initialize sentiment scores
            pos_score = 0.0
            neg_score = 0.0
            
            # Process special case idioms first
            words = self._handle_special_cases(words)
            
            # Calculate word scores with context
            for i, word in enumerate(words):
                word_score = self._get_word_sentiment(word)
                if word_score != 0:
                    # Apply valence modifiers
                    word_score = self._apply_valence_modifiers(words, i, word_score)
                    
                    if word_score > 0:
                        pos_score += word_score
                    else:
                        neg_score += abs(word_score)

            # Add emoji sentiment
            emoji_scores = self._calculate_emoji_sentiment(text)
            pos_score += emoji_scores["pos"]
            neg_score += emoji_scores["neg"]

            # Normalize scores
            total = pos_score + neg_score + 0.0001
            pos_norm = pos_score / total
            neg_norm = neg_score / total
            neu_norm = 1.0 - (pos_norm + neg_norm)

            # Calculate compound score
            compound = self._compute_compound_score(pos_score, neg_score, total)

            return {
                "neg": round(neg_norm, 3),
                "neu": round(neu_norm, 3),
                "pos": round(pos_norm, 3),
                "compound": round(compound, 4)
            }

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize input text into words."""
        # Remove punctuation except in emoticons and special cases
        text = re.sub(r'[^\w\s:)(;]', ' ', text)
        return text.split()

    def _get_word_sentiment(self, word: str) -> float:
        """Get base sentiment score for a word."""
        return self._lexicon.get(word, 0.0)

    def _handle_special_cases(self, words: List[str]) -> List[str]:
        """Process special case idioms in the text."""
        processed_words = []
        i = 0
        while i < len(words):
            multi_word = False
            for idiom in self._special_case_idioms:
                idiom_words = idiom.split()
                if (i + len(idiom_words) <= len(words) and 
                    " ".join(words[i:i+len(idiom_words)]) == idiom):
                    processed_words.append(idiom)
                    i += len(idiom_words)
                    multi_word = True
                    break
            if not multi_word:
                processed_words.append(words[i])
                i += 1
        return processed_words

    def _apply_valence_modifiers(self, words: List[str], index: int, score: float) -> float:
        """Apply valence modifiers based on context."""
        modified_score = score
        
        # Check for negation
        if index > 0:
            if words[index-1] in {"not", "no", "never", "none", "nobody", "nowhere", "nothing"}:
                modified_score *= -0.74
            
            # Apply booster words
            if words[index-1] in self._booster_dict:
                modified_score += self._booster_dict[words[index-1]]

        # Check for preceding tri-gram for negation
        if index > 2:
            if "never so" in " ".join(words[index-3:index]):
                modified_score *= 1.25
                
        # Handle exclamation marks in original text
        if "!" in words[index]:
            modified_score += min(words[index].count("!") * 0.292, 0.876)

        return modified_score

    def _calculate_emoji_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment scores from emojis in text."""
        pos_score = 0.0
        neg_score = 0.0
        
        for emoji, score in self._emoji_lexicon.items():
            count = text.count(emoji)
            if score > 0:
                pos_score += count * score
            else:
                neg_score += count * abs(score)
                
        return {"pos": pos_score, "neg": neg_score}

    def _compute_compound_score(self, pos: float, neg: float, total: float) -> float:
        """Compute normalized compound score."""
        if total == 0:
            return 0.0
            
        compound = (pos - neg) / ((pos + neg) + 15)
        
        # Normalize between -1 and 1
        if compound > 0:
            return min(1, compound)
        elif compound < 0:
            return max(-1, compound)
        else:
            return 0.0