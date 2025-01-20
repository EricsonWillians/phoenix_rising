"""
Phoenix Rising Main Application.

This module serves as the entry point for the Phoenix Rising sanctuary,
creating a serene and professional interface for emotional transformation.
"""

import sys
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
from logging.handlers import RotatingFileHandler

import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.llm_service import (
    LightBearer,
    LightBearerException,
    APIConnectionError,
    APIEndpointUnavailableError
)
from src.database import database, get_db  # Import the Singleton instance
from src.schemas import (
    EmotionState,
    JournalEntryCreate,
    JournalEntryResponse,
    EmotionalProgression
)
from src.utils import Journey, DataProcessor, ApplicationConfig

# Configure logging
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=100000, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def inject_custom_css() -> None:
    """Inject custom CSS for enhanced visual design."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600&family=Raleway:wght@300;400;500&display=swap');
        
        /* Global Styles */
        .stApp {
            background: linear-gradient(
                135deg,
                #0f172a 0%,
                #1e293b 50%,
                #0f172a 100%
            ) !important;
            font-family: 'Raleway', sans-serif;
        }
        
        /* Typography */
        h1, h2, h3 {
            font-family: 'Cinzel', serif !important;
            color: #e2e8f0 !important;
            letter-spacing: 0.05em !important;
        }
        
        .app-title {
            font-size: 3.5rem !important;
            font-weight: 600 !important;
            background: linear-gradient(120deg, #e2e8f0, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem !important;
        }
        
        .subtitle {
            font-family: 'Raleway', sans-serif !important;
            font-size: 1.2rem !important;
            color: #94a3b8 !important;
            font-weight: 300 !important;
            letter-spacing: 0.1em !important;
            line-height: 1.6 !important;
        }
        
        /* Cards and Containers */
        .content-card {
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .content-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }
        
        /* Form Elements */
        .stTextInput > div > div, .stTextArea > div > div {
            background: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            border-radius: 10px !important;
            color: #e2e8f0 !important;
            font-family: 'Raleway', sans-serif !important;
            font-size: 1rem !important;
        }
        
        /* Buttons */
        .stButton > button {
            width: 100%;
            background: linear-gradient(
                135deg,
                #3b82f6 0%,
                #2563eb 100%
            ) !important;
            color: white !important;
            border: none !important;
            padding: 0.75rem 2rem !important;
            border-radius: 25px !important;
            font-family: 'Raleway', sans-serif !important;
            font-weight: 500 !important;
            letter-spacing: 0.05em !important;
            text-transform: uppercase !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(59, 130, 246, 0.5) !important;
        }
        
        /* Light Tokens */
        .light-token {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            position: relative;
            overflow: hidden;
        }
        
        .light-token::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(59, 130, 246, 0.5),
                transparent
            );
        }
        
        .light-token h3 {
            font-family: 'Cinzel', serif !important;
            color: #3b82f6 !important;
            margin-bottom: 1rem !important;
        }
        
        .light-token p {
            font-family: 'Raleway', sans-serif !important;
            color: #e2e8f0 !important;
            font-style: italic !important;
            line-height: 1.6 !important;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: rgba(15, 23, 42, 0.9) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        /* Analytics */
        .analytics-card {
            background: rgba(30, 41, 59, 0.7);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            color: #e2e8f0 !important;
        }
        
        /* Emotion Selector */
        .emotion-selector {
            background: rgba(30, 41, 59, 0.7);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            color: #e2e8f0 !important;
        }
        
        .stSlider > div > div > div {
            background: linear-gradient(
                90deg,
                #ef4444,
                #8b5cf6,
                #3b82f6,
                #22c55e
            ) !important;
        }
                
        .maintenance-card {
            background: linear-gradient(
                135deg,
                rgba(59, 130, 246, 0.1),
                rgba(147, 51, 234, 0.1)
            );
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            text-align: center;
            color: #e2e8f0 !important;
        }

        .maintenance-progress {
            width: 100%;
            height: 4px;
            background: rgba(59, 130, 246, 0.1);
            border-radius: 2px;
            margin-top: 1.5rem;
            overflow: hidden;
        }

        .maintenance-progress .progress-bar {
            width: 30%;
            height: 100%;
            background: linear-gradient(
                90deg,
                #3b82f6,
                #9333ea
            );
            animation: progress 2s infinite ease-in-out;
        }

        @keyframes progress {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(400%); }
        }

        /* Enhanced Contrast for Text and Backgrounds */
        .stTextInput > div > div, .stTextArea > div > div {
            color: #e2e8f0 !important;
        }

        .light-token p, .analytics-card p, .emotion-selector p {
            color: #e2e8f0 !important;
        }

        /* Tooltip Styling */
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted black; /* If you want dots under the hoverable text */
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #1e293b;
            color: #e2e8f0;
            text-align: center;
            border-radius: 6px;
            padding: 5px 0;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position above the text */
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)

class PhoenixRisingUI:
    """Main UI component class for Phoenix Rising."""
    
    def __init__(self):
        """Initialize UI components."""
        self.setup_page_config()
        inject_custom_css()
        self.init_session_state()
        self.config = ApplicationConfig()
    
    def setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Phoenix Rising | Digital Sanctuary",
            page_icon="🔥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def init_session_state(self) -> None:
        """Initialize session state variables."""
        if 'app_state' not in st.session_state:
            st.session_state.app_state = {
                'light_tokens': [],
                'current_emotion': EmotionState.DAWN,
                'show_analytics': False,
                'theme': 'dark',
                'journey_data': []
            }
    
    def render_header(self) -> None:
        """Render application header."""
        st.markdown(
            f"""
            <h1 class="app-title">Phoenix Rising 🔥</h1>
            <p class="subtitle">
                A sanctuary against the machine,<br>where every wound becomes light.
            </p>
            """,
            unsafe_allow_html=True
        )
    
    def render_service_unavailable_message(self) -> None:
        """Render an enhanced service unavailability message with proper HTML rendering."""
        html_content = """
        <div style="
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(147, 51, 234, 0.1));
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            text-align: center;
            color: #e2e8f0;
        ">
            <h3 style="color: #e2e8f0; margin-bottom: 1.5rem; font-size: 1.5rem;">🌅 Sanctuary Renewal in Progress</h3>
            <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem; color: #94a3b8;">
                Our sacred space is currently in a brief period of restoration. This natural cycle 
                helps maintain the sanctuary's strength and resilience.
            </p>
            <div style="
                background: rgba(30, 41, 59, 0.3);
                border-radius: 10px;
                padding: 1.5rem;
                margin: 1rem 0;
            ">
                <h4 style="color: #94a3b8; margin: 1.5rem 0 0.5rem 0; font-size: 1.1rem;">What's Happening?</h4>
                <p style="font-size: 1rem; color: #94a3b8;">
                    To preserve the quality of our sanctuary and ensure sustainable operation, 
                    our service occasionally enters a brief rest state. This process typically lasts 2-3 minutes.
                </p>
                
                <h4 style="color: #94a3b8; margin: 1.5rem 0 0.5rem 0; font-size: 1.1rem;">Steps You Can Take:</h4>
                <ul style="list-style-type: none; padding-left: 0; margin: 0.5rem 0; color: #e2e8f0;">
                    <li style="padding: 0.5rem 0; display: flex; align-items: center;">
                        <strong>💡 Check Your Configuration:</strong> Verify your `.env` file settings and ensure your API tokens are valid.
                    </li>
                    <li style="padding: 0.5rem 0; display: flex; align-items: center;">
                        <strong>🔗 Test Connectivity:</strong> Use tools like `curl` or `Postman` to confirm the endpoints are reachable.
                    </li>
                    <li style="padding: 0.5rem 0; display: flex; align-items: center;">
                        <strong>🌐 Monitor Status:</strong> Visit the <a href="https://huggingface.co/inference-endpoints" target="_blank" style="color: #3b82f6;">Hugging Face dashboard</a> for endpoint health.
                    </li>
                    <li style="padding: 0.5rem 0; display: flex; align-items: center;">
                        <strong>🧘 Pause and Reflect:</strong> Take this moment for a short break or review past entries.
                    </li>
                </ul>
            </div>
            <div style="
                width: 100%;
                height: 4px;
                background: rgba(59, 130, 246, 0.1);
                border-radius: 2px;
                margin-top: 1.5rem;
                overflow: hidden;
            ">
                <div style="
                    width: 30%;
                    height: 100%;
                    background: linear-gradient(90deg, #3b82f6, #9333ea);
                    animation: progress 2s infinite ease-in-out;
                "></div>
            </div>
        </div>
        <style>
            @keyframes progress {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(400%); }
            }
        </style>
        """
        components.html(html_content, height=600)

    
    def render_emotion_selector(self) -> None:
        """Render emotion selection interface."""
        with st.container():
            st.markdown(
                """
                <div class="emotion-selector">
                    <h3>🌟 How does your soul feel today?</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            emotion = st.select_slider(
                "Select Your Emotion",
                options=[e.value for e in EmotionState],
                value=st.session_state.app_state['current_emotion'].value,
                format_func=lambda x: f"{x} - {EmotionState.get_description(x)}",
                help="Choose the emotion that best describes your current state.",
                label_visibility='visible'  # Ensure label is visible
            )
            
            st.session_state.app_state['current_emotion'] = EmotionState(emotion)
    
    def render_journal_section(self) -> None:
        """Render journal entry section."""
        with st.container():
            st.markdown(
                """
                <div class="content-card">
                    <h3>📝 Share your truth</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            with st.form("journal_entry", clear_on_submit=True):
                content = st.text_area(
                    "Your Journal Entry",
                    height=150,
                    placeholder=(
                        "Let your thoughts flow freely in this protected space..."
                    ),
                    max_chars=self.config.max_entry_length,
                    label_visibility='visible'  # Ensure label is visible
                )
                
                submitted = st.form_submit_button(
                    "Transform 🔥",
                    use_container_width=True
                )
                
                if submitted and content.strip():
                    asyncio.run(self.handle_submission(content.strip()))
                elif submitted and not content.strip():
                    st.error("Journal entry cannot be empty. Please share your truth.")
    
    async def handle_submission(self, content: str) -> None:
        """Handle journal entry submission with enhanced error handling and user feedback."""
        try:
            with st.spinner("🕯️ Transmuting experience into light..."):
                async with LightBearer() as light_bearer:
                    token, support, using_fallback = await light_bearer.generate_light_token(
                        entry=content,
                        emotion=st.session_state.app_state['current_emotion']
                    )
                    
                    logger.info(f"Generated Light Token: {token}")
                    
                    if token:
                        st.session_state.app_state['light_tokens'].append(token)
                        
                        if using_fallback:
                            self.render_service_unavailable_message()
                        
                        sentiment_score = light_bearer.last_sentiment_score if light_bearer.last_sentiment_score is not None else 0.0
                        
                        try:
                            # Store in database regardless of fallback status
                            journal_entry = await database.add_journal_entry(
                                content=content,
                                token=token,
                                emotion=st.session_state.app_state['current_emotion'],
                                using_fallback=using_fallback,
                                sentiment_score=sentiment_score
                            )
                            
                            logger.info(f"Journal entry added with ID: {journal_entry.id}")
                            await self.update_journey_data()
                            
                            # Display token after database operation
                            st.markdown(
                                f"""
                                <div class="light-token">
                                    <h3>✨ Your Light Token</h3>
                                    <p>{token}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            if support:
                                st.info(support, icon="🌟")
                            
                        except Exception as e:
                            logger.error(f"Database error: {str(e)}", exc_info=True)
                            st.error(
                                "While your insights have been captured, we encountered a temporary issue "
                                "with storage. Please try again shortly."
                            )
                            
        except APIEndpointUnavailableError:
            self.render_service_unavailable_message()
            logger.warning("Service endpoints unavailable")
            
        except APIConnectionError as e:
            st.error(
                "Our sanctuary is experiencing a moment of connection difficulty. "
                "Like all natural cycles, this too shall pass. Please try again shortly."
            )
            logger.error(f"API Connection Error: {str(e)}")
            
        except LightBearerException as e:
            st.error(
                "An unexpected disturbance affects our sacred space. "
                "We are working to restore harmony."
            )
            logger.error(f"Unexpected Error: {str(e)}")
    
    async def update_journey_data(self) -> None:
        """
        Update journey data for analytics visualization.
        
        This method fetches emotional progression data from the database and updates
        the session state for visualization in the analytics section.
        """
        try:
            # Define the time window for emotional progression
            days = 30
            
            # Fetch progression data from database
            progression = await database.get_emotional_progression(days=days)
            
            # Update session state
            if 'app_state' not in st.session_state:
                st.session_state.app_state = {}
            st.session_state.app_state['journey_data'] = progression
            
            logger.info("Journey data successfully updated for analytics visualization")
            
        except Exception as e:
            logger.error(f"Error updating journey data: {str(e)}", exc_info=True)
            # Don't raise the error - allow the application to continue with stale data
            st.warning(
                "Analytics data may be temporarily outdated. This will not affect your journaling experience."
            )

    
    def map_sentiment_to_emotion(self, score: Optional[float]) -> str:
        """Map sentiment score to emotion category."""
        if score is None:
            return "Unknown"
        if score <= -0.5:
            return "Negative"
        elif score <= 0.5:
            return "Neutral"
        else:
            return "Positive"
    
    def render_sidebar(self) -> None:
        """Render application sidebar."""
        with st.sidebar:
            st.markdown("### 🌅 Journey Settings")
            
            show_analytics = st.checkbox(
                "Show Analytics",
                value=st.session_state.app_state['show_analytics'],
                help="Toggle to display your emotional analytics."
            )
            
            st.session_state.app_state['show_analytics'] = show_analytics
            
            if st.session_state.app_state['light_tokens']:
                st.markdown(
                    """
                    <div class="content-card">
                        <h3>🌟 Recent Light Tokens</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                for token in reversed(
                    st.session_state.app_state['light_tokens'][-5:]
                ):
                    st.markdown(
                        f"""
                        <div class="light-token">
                            <p>{token}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    
    def render_recent_entries(self) -> None:
        """Render recent journal entries."""
        with st.container():
            st.markdown(
                """
                <div class="content-card">
                    <h3>🗒️ Recent Journal Entries</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Fetch recent entries from the database
            try:
                recent_entries = asyncio.run(database.get_recent_entries(limit=5))
                if recent_entries:
                    for entry in recent_entries:
                        st.markdown(
                            f"""
                            <div class="content-card">
                                <h4>{entry.emotion.value} - {entry.created_at.strftime('%Y-%m-%d %H:%M')}</h4>
                                <p>{entry.content}</p>
                                <p><strong>Light Token:</strong> {entry.light_token}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No journal entries found. Start by sharing your truth!")
            except Exception as e:
                logger.error(f"Error fetching recent entries: {str(e)}")
                st.error("An error occurred while fetching recent journal entries.")

    async def update_journey_data(self) -> None:
        """
        Update journey data for analytics visualization.
        
        This method fetches emotional progression data from the database and updates
        the session state for visualization in the analytics section.
        """
        try:
            # Define the time window for emotional progression
            days = 30
            
            # Fetch progression data from database
            progression = await database.get_emotional_progression(days=days)
            
            # Update session state
            if 'app_state' not in st.session_state:
                st.session_state.app_state = {}
            st.session_state.app_state['journey_data'] = progression
            
            logger.info("Journey data successfully updated for analytics visualization")
            
        except Exception as e:
            logger.error(f"Error updating journey data: {str(e)}", exc_info=True)
            # Don't raise the error - allow the application to continue with stale data
            st.warning(
                "Analytics data may be temporarily outdated. This will not affect your journaling experience."
            )

    def render_analytics(self) -> None:
        """
        Render the analytics section with emotional journey visualizations.
        
        This method creates and displays various visualizations of the user's emotional
        journey, including sentiment trends and emotion distribution.
        """
        if not st.session_state.app_state.get('show_analytics', False):
            return

        with st.container():
            st.markdown(
                """
                <div class="analytics-card">
                    <h3>📊 Your Journey Through Light and Shadow</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            journey_data = st.session_state.app_state.get('journey_data', [])
            
            if not journey_data:
                st.info(
                    "Your journey analytics will appear here once you begin sharing your experiences."
                )
                return
            
            try:
                # Create DataFrame for visualization
                import pandas as pd
                df = pd.DataFrame(journey_data, columns=['date', 'sentiment'])
                df['date'] = pd.to_datetime(df['date'])
                
                # Sentiment Trend Visualization
                fig_sentiment = px.line(
                    df,
                    x='date',
                    y='sentiment',
                    title='Your Emotional Journey Over Time',
                    labels={
                        'date': 'Date',
                        'sentiment': 'Emotional State'
                    },
                    template='plotly_dark'
                )
                
                # Customize the appearance
                fig_sentiment.update_traces(
                    line_color='#3b82f6',
                    line_width=2
                )
                fig_sentiment.update_layout(
                    plot_bgcolor='rgba(30, 41, 59, 0.1)',
                    paper_bgcolor='rgba(30, 41, 59, 0.1)',
                    font=dict(color='#e2e8f0'),
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(148, 163, 184, 0.1)'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(148, 163, 184, 0.1)')
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
                
                # Emotion Distribution
                emotions = df['sentiment'].apply(self._map_sentiment_to_emotion)
                emotion_counts = emotions.value_counts()
                
                fig_distribution = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    title='Distribution of Emotional States',
                    color_discrete_sequence=['#3b82f6', '#8b5cf6', '#ec4899']
                )
                fig_distribution.update_layout(
                    plot_bgcolor='rgba(30, 41, 59, 0.1)',
                    paper_bgcolor='rgba(30, 41, 59, 0.1)',
                    font=dict(color='#e2e8f0')
                )
                st.plotly_chart(fig_distribution, use_container_width=True)
                
                # Additional insights
                st.markdown(
                    """
                    <div class="analytics-insights">
                        <h4>Journey Insights</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Calculate and display insights
                avg_sentiment = df['sentiment'].mean()
                sentiment_trend = df['sentiment'].diff().mean()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Average Emotional State",
                        f"{avg_sentiment:.2f}",
                        f"{sentiment_trend:.2f}"
                    )
                with col2:
                    most_common_emotion = emotions.mode()[0]
                    st.metric("Predominant State", most_common_emotion)
                
            except Exception as e:
                logger.error(f"Error rendering analytics: {str(e)}", exc_info=True)
                st.error(
                    "We encountered an issue displaying your analytics. "
                    "This is temporary and doesn't affect your journaling experience."
                )
    
    def _map_sentiment_to_emotion(self, score: float) -> str:
        """
        Map numerical sentiment scores to emotional states.
        
        Args:
            score: Sentiment score between -1 and 1
            
        Returns:
            String representing the emotional state
        """
        if score <= -0.5:
            return "Introspective"
        elif score <= 0:
            return "Balanced"
        else:
            return "Illuminated"
    
    def render(self) -> None:
        """Render the complete interface."""
        self.render_header()
        
        # Initialize the database before rendering UI
        try:
            asyncio.run(database.initialize_database())
            logger.info("Database initialized successfully.")
        except Exception as e:
            st.error(f"Database initialization failed: {e}")
            logger.error(f"Database initialization failed: {e}")
            st.stop()  # Stop further execution if initialization fails
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.render_emotion_selector()
            self.render_journal_section()
            self.render_recent_entries()
            
        with col2:
            self.render_analytics()
            
        self.render_sidebar()

def main() -> None:
    """Main application entry point."""
    ui = PhoenixRisingUI()
    ui.render()

if __name__ == "__main__":
    # Register shutdown handler to close database connections gracefully
    import atexit

    @atexit.register
    def shutdown():
        """Ensure that the database connections are closed on shutdown."""
        asyncio.run(database.close())
    
    main()
