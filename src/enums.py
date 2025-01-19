"""
Phoenix Rising Enums.

This module defines enumeration classes used across the Phoenix Rising application
to ensure consistency and data integrity.
"""

from enum import Enum


class EmotionState(str, Enum):
    """Enumeration of possible emotional states in the spiritual journey."""
    EMBER = "Ember"
    SHADOW = "Shadow"
    STORM = "Storm"
    DAWN = "Dawn"
    STARLIGHT = "Starlight"

    @classmethod
    def get_description(cls, emotion: "EmotionState") -> str:
        """Provide spiritual context for each emotional state."""
        descriptions = {
            cls.EMBER: "The last warmth of a dying fire, holding potential for rebirth",
            cls.SHADOW: "The depth where hidden strengths germinate",
            cls.STORM: "Chaos that precedes transformation",
            cls.DAWN: "First light breaking through darkness",
            cls.STARLIGHT: "Eternal guidance in the void",
        }
        return descriptions.get(emotion, "Unknown emotional state")
