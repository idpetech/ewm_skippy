#!/usr/bin/env python3
"""
Skippy Multi-Coach System

A comprehensive coaching system with multiple specialized coaches:
- EWM Coach: SAP EWM operations and processes
- Business Analyst Coach: Requirements and process analysis
- Support Coach: Technical support and troubleshooting
- Dev Guru Coach: Source code analysis and development guidance

Features:
- Tabbed interface for different coaches
- Mix and match capabilities
- Context-aware conversations
- Specialized knowledge bases
"""

import streamlit as st
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Import our coach system
from src.base_coach import (
    BaseCoach, CoachType, CoachCapabilities, Conversation, 
    ConversationStage, CoachFactory, MixedCoach
)

# =============================================================================
# CONFIGURATION
# =============================================================================

class MultiCoachConfig:
    """Configuration for the multi-coach system"""
    
    def __init__(self):
        # Azure OpenAI Configuration
        self.embedding_endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT", "https://genaiapimna.jnj.com/openai-embeddings/openai")
        self.embedding_api_key = os.getenv("AZURE_EMBEDDING_API_KEY", "f89d10a91b9d4cc989085a495d695eb3")
        self.embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.embedding_api_version = os.getenv("AZURE_EMBEDDING_API_VERSION", "2022-12-01")
        
        self.chat_endpoint = os.getenv("AZURE_CHAT_ENDPOINT", "https://genaiapimna.jnj.com/openai-chat")
        self.chat_api_key = os.getenv("AZURE_CHAT_API_KEY", "acd5a7d2b4d64ea6871aeb4cbc3113dd")
        self.chat_deployment = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o")
        self.chat_api_version = os.getenv("AZURE_CHAT_API_VERSION", "2023-05-15")
        
        # Database Configuration
        self.chroma_db_paths = {
            'ewm': os.getenv("EWM_CHROMA_DB_PATH", "./data/ewm_db"),
            'business_analyst': os.getenv("BA_CHROMA_DB_PATH", "./data/ba_db"),
            'support': os.getenv("SUPPORT_CHROMA_DB_PATH", "./data/support_db"),
            'dev_guru': os.getenv("DEV_CHROMA_DB_PATH", "./data/dev_db")
        }
        
        # Application Settings
        self.max_clarifications = int(os.getenv("MAX_CLARIFICATIONS", "5"))
        self.doc_retrieval_count = int(os.getenv("DOC_RETRIEVAL_COUNT", "3"))
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'embedding_endpoint': self.embedding_endpoint,
            'embedding_api_key': self.embedding_api_key,
            'embedding_deployment': self.embedding_deployment,
            'embedding_api_version': self.embedding_api_version,
            'chat_endpoint': self.chat_endpoint,
            'chat_api_key': self.chat_api_key,
            'chat_deployment': self.chat_deployment,
            'chat_api_version': self.chat_api_version,
            'chroma_db_path': self.chroma_db_paths,  # Will be overridden per coach
            'max_clarifications': self.max_clarifications,
            'doc_retrieval_count': self.doc_retrieval_count,
            'llm_temperature': self.llm_temperature
        }

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Setup structured logging"""
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("skippy_multi_coach")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(logs_dir / "skippy_multi_coach.log")
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class StateManager:
    """Centralized state management for Streamlit"""
    
    @staticmethod
    def init_conversations() -> Dict[str, Conversation]:
        """Initialize conversation states for all coaches"""
        if "conversations" not in st.session_state:
            st.session_state.conversations = {
                'ewm': Conversation(),
                'business_analyst': Conversation(),
                'support': Conversation(),
                'dev_guru': Conversation(),
                'mixed': Conversation()
            }
        return st.session_state.conversations
    
    @staticmethod
    def init_messages() -> Dict[str, List[Dict[str, str]]]:
        """Initialize message history for all coaches"""
        if "messages" not in st.session_state:
            welcome_messages = {
                'ewm': [{
                    "role": "assistant",
                    "content": """🏭 **Welcome to Skippy EWM Coach!**

I specialize in SAP EWM operations and processes:
- 📥 Inbound and putaway operations
- 📤 Outbound picking and shipping
- 📦 Inventory management
- 🔧 Process troubleshooting
- ⚙️ Configuration guidance

**What EWM challenge can I help you with today?**"""
                }],
                'business_analyst': [{
                    "role": "assistant",
                    "content": """📋 **Welcome to Skippy Business Analyst Coach!**

I specialize in business analysis and requirements:
- 📊 Requirements analysis and documentation
- 🔄 Process mapping and optimization
- 👥 Stakeholder communication
- 📈 Business case development
- 🎯 Gap analysis and solutions

**What business analysis challenge can I help you with?**"""
                }],
                'support': [{
                    "role": "assistant",
                    "content": """🔧 **Welcome to Skippy Support Coach!**

I specialize in technical support and troubleshooting:
- 🐛 Error diagnosis and resolution
- 📚 Documentation and guides
- 👥 User support and training
- 🔍 Problem analysis
- 📞 Escalation guidance

**What technical issue can I help you resolve?**"""
                }],
                'dev_guru': [{
                    "role": "assistant",
                    "content": """💻 **Welcome to Skippy Dev Guru Coach!**

I specialize in software development and code analysis:
- 🏗️ Code architecture and design patterns
- 🐛 Bug detection and fixes
- ⚡ Performance optimization
- 🔧 Refactoring and code quality
- 🧪 Testing strategies
- 🗄️ Database design

**What development challenge can I help you with?**"""
                }],
                'mixed': [{
                    "role": "assistant",
                    "content": """🌟 **Welcome to Skippy Mixed Coach!**

I combine expertise from multiple domains:
- 🏭 EWM operations and processes
- 📋 Business analysis and requirements
- 🔧 Technical support and troubleshooting
- 💻 Software development and code analysis

I'll automatically route your questions to the most appropriate specialist!

**What can I help you with today?**"""
                }]
            }
            st.session_state.messages = welcome_messages
        return st.session_state.messages
    
    @staticmethod
    def reset_conversation(coach_type: str):
        """Reset conversation for a specific coach"""
        if hasattr(st.session_state, 'conversations') and coach_type in st.session_state.conversations:
            st.session_state.conversations[coach_type].reset()
        if hasattr(st.session_state, 'messages') and coach_type in st.session_state.messages:
            # Keep only the welcome message
            welcome_msg = st.session_state.messages[coach_type][0]
            st.session_state.messages[coach_type] = [welcome_msg]

# =============================================================================
# COACH MANAGER
# =============================================================================

class CoachManager:
    """Manages multiple coaches and their interactions"""
    
    def __init__(self, config: MultiCoachConfig):
        self.config = config
        self.coaches = {}
        self.mixed_coach = None
        self._initialize_coaches()
    
    def _initialize_coaches(self):
        """Initialize all available coaches"""
        try:
            # Initialize individual coaches
            coach_configs = {
                'ewm': {**self.config.to_dict(), 'chroma_db_path': self.config.chroma_db_paths['ewm']},
                'business_analyst': {**self.config.to_dict(), 'chroma_db_path': self.config.chroma_db_paths['business_analyst']},
                'support': {**self.config.to_dict(), 'chroma_db_path': self.config.chroma_db_paths['support']},
                'dev_guru': {**self.config.to_dict(), 'chroma_db_path': self.config.chroma_db_paths['dev_guru']}
            }
            
            for coach_name, coach_config in coach_configs.items():
                try:
                    coach_type = CoachType(coach_name)
                    self.coaches[coach_name] = CoachFactory.create_coach(coach_type, coach_config)
                    logger.info(f"Initialized {coach_name} coach successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize {coach_name} coach: {e}")
            
            # Initialize mixed coach
            available_coach_types = [CoachType(name) for name in self.coaches.keys()]
            if available_coach_types:
                self.mixed_coach = MixedCoach(available_coach_types, self.config.to_dict())
                logger.info("Initialized mixed coach successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize coaches: {e}")
    
    def get_coach(self, coach_type: str) -> Optional[BaseCoach]:
        """Get a specific coach"""
        if coach_type == 'mixed':
            return self.mixed_coach
        return self.coaches.get(coach_type)
    
    def get_available_coaches(self) -> List[str]:
        """Get list of available coaches"""
        available = list(self.coaches.keys())
        if self.mixed_coach:
            available.append('mixed')
        return available

# =============================================================================
# UI COMPONENTS
# =============================================================================

class UIComponents:
    """UI components for the multi-coach interface"""
    
    @staticmethod
    def setup_page():
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Skippy Multi-Coach System",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Multi-coach CSS
        st.markdown("""
        <style>
        /* Chat container styling */
        .chat-container {
            height: 500px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            background-color: #fafafa;
            margin-bottom: 1rem;
        }
        
        /* User message styling */
        .user-message {
            background: linear-gradient(135deg, #007acc, #005999);
            color: white;
            padding: 0.8rem 1rem;
            border-radius: 18px 18px 4px 18px;
            margin: 0.5rem 0 0.5rem 20%;
            text-align: right;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            word-wrap: break-word;
        }
        
        /* Coach message styling */
        .coach-message {
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            color: #333;
            padding: 0.8rem 1rem;
            border-radius: 18px 18px 18px 4px;
            margin: 0.5rem 20% 0.5rem 0;
            border-left: 4px solid #28a745;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            word-wrap: break-word;
        }
        
        /* Message timestamps */
        .message-time {
            font-size: 0.7rem;
            opacity: 0.7;
            margin-top: 0.3rem;
        }
        
        /* Coach badges */
        .coach-badge {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            border: 1px solid #1976d2;
            border-radius: 8px;
            padding: 0.6rem;
            margin: 0.5rem 0;
            font-size: 0.9rem;
            color: #1976d2;
            font-weight: 500;
        }
        
        /* Input area styling */
        .stTextInput > div > div > input {
            border-radius: 25px;
            border: 2px solid #e0e0e0;
            padding: 0.8rem 1rem;
        }
        
        .stButton > button {
            border-radius: 25px;
            font-weight: 600;
            height: 3rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        
        /* Hide Streamlit elements */
        .stApp > header {
            visibility: hidden;
        }
        
        .stApp > div:first-child {
            padding-top: 1rem;
        }
        
        /* Scrollbar styling */
        .chat-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        .chat-container::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def display_chat_container(messages: List[Dict[str, str]], coach_type: str):
        """Display scrollable chat container"""
        # Create chat container
        chat_html = '<div class="chat-container">'
        
        for message in messages:
            timestamp = datetime.now().strftime("%H:%M")
            if message["role"] == "user":
                chat_html += (
                    '<div class="user-message">'
                    f'<div>{message["content"]}</div>'
                    f'<div class="message-time">{timestamp}</div>'
                    '</div>'
                )
            else:
                # Add coach type indicator
                coach_icons = {
                    'ewm': '🏭',
                    'business_analyst': '📋',
                    'support': '🔧',
                    'dev_guru': '💻',
                    'mixed': '🌟'
                }
                icon = coach_icons.get(coach_type, '🤖')
                
                chat_html += (
                    '<div class="coach-message">'
                    f'<div><strong>{icon} Coach Skippy:</strong><br>{message["content"]}</div>'
                    f'<div class="message-time">{timestamp}</div>'
                    '</div>'
                )
        
        chat_html += '</div>'
        
        # Display chat container
        st.markdown(chat_html, unsafe_allow_html=True)
        
        # Auto-scroll to bottom
        st.markdown(
            '<script>'
            'setTimeout(function() {'
            '    var chatContainer = document.querySelector(".chat-container");'
            '    if (chatContainer) {'
            '        chatContainer.scrollTop = chatContainer.scrollHeight;'
            '    }'
            '}, 100);'
            '</script>',
            unsafe_allow_html=True
        )
    
    @staticmethod
    def create_sidebar(coach_manager: CoachManager, current_coach: str):
        """Create informative sidebar"""
        with st.sidebar:
            st.header("🚀 Skippy Multi-Coach System")
            
            # Coach selection
            st.subheader("🎯 Select Coach")
            available_coaches = coach_manager.get_available_coaches()
            
            coach_descriptions = {
                'ewm': '🏭 EWM Coach - SAP warehouse operations',
                'business_analyst': '📋 Business Analyst - Requirements & processes',
                'support': '🔧 Support Coach - Technical troubleshooting',
                'dev_guru': '💻 Dev Guru - Code analysis & development',
                'mixed': '🌟 Mixed Coach - All capabilities'
            }
            
            for coach in available_coaches:
                if st.button(coach_descriptions.get(coach, coach), key=f"select_{coach}"):
                    st.session_state.current_coach = coach
                    st.rerun()
            
            st.markdown("---")
            
            # Current coach info
            if current_coach in available_coaches:
                coach = coach_manager.get_coach(current_coach)
                if coach:
                    st.subheader(f"Current: {coach_descriptions.get(current_coach, current_coach)}")
                    
                    # Show capabilities
                    capabilities = coach.capabilities
                    st.markdown("**Capabilities:**")
                    if capabilities.can_read_documents:
                        st.markdown("📚 Document Reading")
                    if capabilities.can_analyze_code:
                        st.markdown("💻 Code Analysis")
                    if capabilities.can_generate_code:
                        st.markdown("🔧 Code Generation")
                    if capabilities.can_detect_code_smells:
                        st.markdown("🔍 Code Smell Detection")
                    if capabilities.can_recommend_fixes:
                        st.markdown("🛠️ Fix Recommendations")
                    if capabilities.can_analyze_architecture:
                        st.markdown("🏗️ Architecture Analysis")
                    if capabilities.can_query_database:
                        st.markdown("🗄️ Database Queries")
                    if capabilities.can_access_confluence:
                        st.markdown("📖 Confluence Access")
            
            st.markdown("---")
            
            # Reset conversation
            if st.button("🔄 Reset Conversation", use_container_width=True, type="primary"):
                StateManager.reset_conversation(current_coach)
                st.rerun()
            
            st.markdown("---")
            st.markdown("*Skippy Multi-Coach v1.0*")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

@st.cache_resource
def init_coach_manager() -> CoachManager:
    """Initialize coach manager with caching"""
    try:
        config = MultiCoachConfig()
        return CoachManager(config)
    except Exception as e:
        logger.error(f"Failed to initialize coach manager: {e}")
        st.error("❌ Failed to initialize coaching system. Please check configuration.")
        st.stop()

def main():
    """Main application entry point"""
    UIComponents.setup_page()
    
    # Header
    st.title("🚀 Skippy Multi-Coach System")
    st.markdown("*Comprehensive coaching with specialized expertise*")
    
    # Initialize system
    coach_manager = init_coach_manager()
    conversations = StateManager.init_conversations()
    messages = StateManager.init_messages()
    
    # Set default coach
    if "current_coach" not in st.session_state:
        available_coaches = coach_manager.get_available_coaches()
        st.session_state.current_coach = available_coaches[0] if available_coaches else 'ewm'
    
    current_coach = st.session_state.current_coach
    
    # Create main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat container
        UIComponents.display_chat_container(messages[current_coach], current_coach)
        
        # Input section
        with st.form("coach_form", clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])
            
            with col_input:
                user_input = st.text_input(
                    f"Ask {current_coach.replace('_', ' ').title()} Coach...",
                    placeholder="Type your question here...",
                    key="question_input",
                    label_visibility="collapsed",
                    max_chars=500
                )
            
            with col_send:
                submitted = st.form_submit_button(
                    "Send 🚀", 
                    use_container_width=True,
                    type="primary"
                )
        
        # Process input
        if submitted and user_input:
            messages[current_coach].append({"role": "user", "content": user_input})
            
            with st.spinner(f"🧠 {current_coach.replace('_', ' ').title()} Coach thinking..."):
                coach = coach_manager.get_coach(current_coach)
                if coach:
                    response = coach.ask(user_input, conversations[current_coach], messages[current_coach])
                else:
                    response = "❌ Coach not available. Please try again."
            
            messages[current_coach].append({"role": "assistant", "content": response})
            conversations[current_coach].add_message("assistant", response)
            st.rerun()
    
    with col2:
        # Sidebar
        UIComponents.create_sidebar(coach_manager, current_coach)

if __name__ == "__main__":
    main()
