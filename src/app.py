"""
Phoenix Rising Main Application.

This module serves as the entry point for the Phoenix Rising sanctuary,
integrating all components into a cohesive interface that provides users
with a space for reflection and growth away from corporate mechanization.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import logging

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy.ext.asyncio import AsyncSession

from src.llm_service import LightBearer, LightBearerException
from src.database import DatabaseManager, get_db
from src.schemas import (
    EmotionState,
    JournalEntryCreate,
    JournalEntryResponse,
    EmotionalProgression
)
from src.utils import (
    Journey,
    DataProcessor,
    ApplicationConfig,
    setup_error_handling
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = ApplicationConfig()

class PhoenixRisingApp:
    """Main application class for Phoenix Rising sanctuary."""
    
    def __init__(self):
        """Initialize application components and state."""
        self.setup_streamlit_config()
        self.initialize_session_state()
        self.database = DatabaseManager()
        self.setup_page_styling()

    def setup_streamlit_config(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Phoenix Rising | Digital Sanctuary",
            page_icon="🔥",
            layout="centered",
            initial_sidebar_state="expanded"
        )

    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if 'light_tokens' not in st.session_state:
            st.session_state.light_tokens = []
        if 'current_emotion' not in st.session_state:
            st.session_state.current_emotion = EmotionState.DAWN
        if 'show_analytics' not in st.session_state:
            st.session_state.show_analytics = False

    def setup_page_styling(self) -> None:
        """Apply custom styling to the interface."""
        st.markdown("""
            <style>
            .stApp {
                background: linear-gradient(
                    135deg,
                    #1a1a2e 0%,
                    #16213e 50%,
                    #1a1a2e 100%
                );
                color: #e2e2e2;
            }
            .element-container {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 1rem;
                margin: 1rem 0;
            }
            .stTextInput, .stTextArea {
                background-color: rgba(255, 255, 255, 0.05) !important;
                border-radius: 10px !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
            }
            .stButton > button {
                background: linear-gradient(
                    45deg,
                    #4a90e2,
                    #67b26f
                ) !important;
                color: white !important;
                border: none !important;
                padding: 0.75rem 2rem !important;
                border-radius: 25px !important;
                transition: all 0.3s ease !important;
            }
            .stButton > button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
            }
            .light-token {
                background: rgba(74, 144, 226, 0.1);
                padding: 2rem;
                border-radius: 15px;
                border: 1px solid rgba(74, 144, 226, 0.2);
                margin: 1rem 0;
            }
            </style>
        """, unsafe_allow_html=True)

    async def process_journal_entry(
        self,
        content: str,
        emotion: EmotionState,
        db_session: AsyncSession
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Process a journal entry and generate a light token.
        
        Args:
            content: Journal entry text
            emotion: Selected emotion
            db_session: Database session
            
        Returns:
            Tuple of (light token, optional support message)
        """
        async with LightBearer() as light_bearer:
            try:
                # Generate light token
                token, support_message = await light_bearer.generate_light_token(
                    entry=content,
                    emotion=emotion
                )
                
                if token:
                    # Create database entry
                    entry = JournalEntryCreate(
                        content=content,
                        emotion=emotion,
                        light_token=token,
                        sentiment_score=light_bearer.last_sentiment_score
                    )
                    
                    await self.database.create_journal_entry(
                        entry,
                        db_session
                    )
                    st.session_state.light_tokens.append(token)
                    
                return token, support_message
                
            except LightBearerException as e:
                logger.error(f"Error generating light token: {e}")
                return None, str(e)

    def render_emotion_selector(self) -> None:
        """Render the emotion selection interface."""
        st.markdown("### 🌟 How does your soul feel today?")
        
        emotion = st.select_slider(
            "",
            options=[e.value for e in EmotionState],
            value=st.session_state.current_emotion.value,
            format_func=lambda x: f"{x} - {EmotionState.get_description(x)}"
        )
        
        st.session_state.current_emotion = EmotionState(emotion)

    def render_journal_entry(self) -> None:
        """Render the journal entry interface."""
        st.markdown("### 📝 Share your truth")
        
        with st.form("journal_entry", clear_on_submit=True):
            content = st.text_area(
                "",
                height=150,
                placeholder=(
                    "Let your thoughts flow freely in this protected space..."
                ),
                max_chars=config.max_entry_length
            )
            
            cols = st.columns([3, 1])
            with cols[0]:
                submitted = st.form_submit_button(
                    "Transform 🔥",
                    use_container_width=True
                )
                
            if submitted and content:
                self.handle_journal_submission(content)

    async def handle_journal_submission(self, content: str) -> None:
        """Handle journal entry submission and token generation."""
        with st.spinner("Transmuting experience into light..."):
            db_session = await get_db().__anext__()
            token, support = await self.process_journal_entry(
                content,
                st.session_state.current_emotion,
                db_session
            )
            
            if token:
                st.markdown(
                    f"""
                    <div class='light-token'>
                        <h3>✨ Your Light Token</h3>
                        <p style='font-size: 1.1em; font-style: italic;'>
                        {token}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                if support:
                    st.info(support)
            else:
                st.error(
                    "The light dims momentarily. Please try again later."
                )

    def render_journey_analytics(self) -> None:
        """Render journey analytics visualization."""
        if not st.session_state.show_analytics:
            return
            
        st.markdown("### 📊 Your Journey Through Light and Shadow")
        
        async def load_analytics():
            db_session = await get_db().__anext__()
            progression = await self.database.get_emotional_progression(
                db_session,
                days=30
            )
            return progression
        
        progression = asyncio.run(load_analytics())
        
        if progression:
            fig = self.create_journey_visualization(progression)
            st.plotly_chart(fig, use_container_width=True)

    def create_journey_visualization(
        self,
        progression: EmotionalProgression
    ) -> go.Figure:
        """Create journey visualization using Plotly."""
        fig = go.Figure()
        
        # Add emotional journey line
        fig.add_trace(go.Scatter(
            x=[p['date'] for p in progression],
            y=[p['sentiment'] for p in progression],
            mode='lines+markers',
            name='Emotional Journey',
            line=dict(color='#4a90e2', width=2),
            marker=dict(
                size=8,
                color=[
                    '#ff6b6b' if p['emotion'] == EmotionState.EMBER else
                    '#4a4e69' if p['emotion'] == EmotionState.SHADOW else
                    '#4361ee' if p['emotion'] == EmotionState.STORM else
                    '#ff9e64' if p['emotion'] == EmotionState.DAWN else
                    '#9d4edd'
                    for p in progression
                ]
            )
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='#e2e2e2'),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,255,255,0.1)',
                range=[-1, 1]
            ),
            showlegend=False
        )
        
        return fig

    def render_sidebar(self) -> None:
        """Render application sidebar."""
        with st.sidebar:
            st.markdown("### 🌅 Journey Settings")
            
            st.toggle(
                "Show Analytics",
                value=st.session_state.show_analytics,
                key="show_analytics"
            )
            
            if st.session_state.light_tokens:
                st.markdown("### 🌟 Recent Light Tokens")
                for token in reversed(
                    st.session_state.light_tokens[-5:]
                ):
                    st.markdown(
                        f"""
                        <div style='font-size: 0.9em; opacity: 0.8; 
                                 margin-bottom: 1rem;'>
                            "{token}"
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    def run(self) -> None:
        """Run the Phoenix Rising application."""
        st.title("🔥 Phoenix Rising")
        st.markdown("""
            <p style='font-size: 1.2em; font-style: italic; opacity: 0.8;'>
            A sanctuary against the machine, where every wound becomes light.
            </p>
        """, unsafe_allow_html=True)
        
        # Initialize database
        asyncio.run(self.database.create_tables())
        
        # Render interface components
        self.render_emotion_selector()
        self.render_journal_entry()
        self.render_journey_analytics()
        self.render_sidebar()

if __name__ == "__main__":
    setup_error_handling()
    app = PhoenixRisingApp()
    app.run()