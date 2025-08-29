#!/usr/bin/env python3
"""
Skippy - SAP EWM AI Assistant Chat Interface

A production-ready AI assistant for SAP Extended Warehouse Management.
Skippy is a friendly coach and lead operator helping with warehouse management tasks.

Features:
- Streamlit-based chat interface with conversational memory
- ChromaDB vector database integration for document retrieval
- OpenAI GPT-3.5/GPT-4 support with configurable settings
- PDF document upload and indexing capabilities
- Professional UI with avatar support and responsive design
- Comprehensive error handling and logging
- Environment-based configuration for secure deployment
- Caching for optimal performance

Author: AI Development Team
Version: 1.0.0
License: MIT
"""

import os
import sys
import time
import tempfile
import subprocess
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Third-party imports
import streamlit as st
from PIL import Image
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure comprehensive logging
import logging

def setup_logging():
    """Setup comprehensive logging for the application."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'skippy_app.log'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing configuration
    )
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Skippy application logging initialized")
    return logger

# Initialize logger
logger = setup_logging()

# Application configuration from environment variables
class Config:
    """Application configuration loaded from environment variables and defaults."""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL_DEFAULT", "gpt-3.5-turbo")
    OPENAI_TEMPERATURE_DEFAULT = float(os.getenv("OPENAI_TEMPERATURE_DEFAULT", "0.7"))
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
    
    # ChromaDB Configuration
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "sap_ewm_docs")
    
    # Application Configuration
    DEFAULT_CONTEXT_DOCS = int(os.getenv("DEFAULT_CONTEXT_DOCS", "3"))
    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "6"))
    DATA_DIRECTORY = os.getenv("DATA_DIRECTORY", "./data")
    ASSETS_DIRECTORY = os.getenv("ASSETS_DIRECTORY", "./assets")
    
    # Embedding Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@st.cache_resource
def initialize_chroma_retriever():
    """Initialize ChromaDB retriever with caching to avoid rebuilding on every query."""
    try:
        logger.info("Initializing ChromaDB retriever...")
        return ChromaDBRetriever()
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB retriever: {e}")
        return None


@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model with caching."""
    try:
        logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        return SentenceTransformer(Config.EMBEDDING_MODEL)
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return None


class ChromaDBRetriever:
    """
    Retrieve relevant documents from ChromaDB vector database.
    
    This class handles all interactions with the ChromaDB vector database,
    including document retrieval and collection statistics.
    """
    
    def __init__(self, db_path: str = None, collection_name: str = None):
        """
        Initialize ChromaDB retriever.
        
        Args:
            db_path (str, optional): Path to ChromaDB database. Defaults to Config.CHROMA_DB_PATH.
            collection_name (str, optional): Name of the collection. Defaults to Config.CHROMA_COLLECTION_NAME.
        
        Raises:
            Exception: If connection to ChromaDB fails.
        """
        self.db_path = db_path or Config.CHROMA_DB_PATH
        self.collection_name = collection_name or Config.CHROMA_COLLECTION_NAME
        self.client = None
        self.collection = None
        self.embeddings = None
        
        self._initialize_database()
        self._initialize_embeddings()
    
    def _initialize_database(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Check if ChromaDB directory exists
            if not Path(self.db_path).exists():
                logger.error(f"ChromaDB directory not found: {self.db_path}")
                raise FileNotFoundError(f"Knowledge base not found. Please run 'python build_index.py' first.")
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get collection (will raise exception if doesn't exist)
            try:
                self.collection = self.client.get_collection(self.collection_name)
                logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Collection '{self.collection_name}' not found: {e}")
                raise Exception(f"Knowledge base collection not found. Please run 'python build_index.py' first.")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize embedding model for query processing."""
        try:
            self.embeddings = load_embedding_model()
            if self.embeddings is None:
                raise Exception("Failed to load embedding model")
            logger.info(f"Embedding model loaded: {Config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def retrieve_relevant_docs(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top N relevant documents for a query.
        
        Args:
            query (str): The search query.
            n_results (int, optional): Number of results to return. Defaults to 5.
        
        Returns:
            List[Dict[str, Any]]: List of relevant documents with metadata.
        """
        try:
            logger.info(f"Retrieving documents for query: '{query[:50]}...' (n_results={n_results})")
            
            # Validate inputs
            if not query or not query.strip():
                logger.warning("Empty query provided")
                return []
            
            if n_results <= 0:
                logger.warning(f"Invalid n_results: {n_results}")
                n_results = 3
            
            # Generate query embedding
            query_embedding = self.embeddings.encode([query.strip()]).tolist()
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, 10),  # Cap at 10 results
                include=["documents", "metadatas", "distances"]
            )
            
            # Format and validate results
            relevant_docs = []
            if results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    try:
                        metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                        distance = results["distances"][0][i] if results.get("distances") else 1.0
                        
                        # Calculate similarity (1 - distance)
                        similarity = max(0, min(1, 1 - distance))  # Clamp between 0 and 1
                        
                        # Extract and clean metadata
                        source_file = metadata.get("source_file", "Unknown")
                        page_number = metadata.get("page_number", "N/A")
                        section_title = metadata.get("section_title", "N/A")
                        tags = metadata.get("tags", "")
                        
                        # Parse tags
                        tag_list = []
                        if tags and isinstance(tags, str):
                            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
                        
                        relevant_docs.append({
                            "content": str(doc).strip(),
                            "metadata": metadata,
                            "similarity": similarity,
                            "source": source_file,
                            "page": page_number,
                            "section": section_title,
                            "tags": tag_list
                        })
                    except Exception as e:
                        logger.warning(f"Error processing document {i}: {e}")
                        continue
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
            
            if not relevant_docs:
                logger.warning("No relevant documents found for query")
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the ChromaDB collection.
        
        Returns:
            Dict[str, Any]: Collection statistics including document count, sources, etc.
        """
        try:
            logger.info("Retrieving collection statistics...")
            
            # Get collection count
            total_count = self.collection.count()
            
            if total_count == 0:
                return {
                    "total_chunks": 0,
                    "source_files": [],
                    "unique_tags": [],
                    "status": "empty"
                }
            
            # Get sample documents for analysis
            sample_size = min(50, total_count)  # Sample up to 50 documents
            sample = self.collection.get(limit=sample_size)
            
            # Analyze sample metadata
            source_files = set()
            all_tags = set()
            
            if sample.get("metadatas"):
                for meta in sample["metadatas"]:
                    if isinstance(meta, dict):
                        # Extract source files
                        if "source_file" in meta and meta["source_file"]:
                            source_files.add(str(meta["source_file"]))
                        
                        # Extract and parse tags
                        if "tags" in meta and meta["tags"]:
                            tags_str = str(meta["tags"])
                            tag_list = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
                            all_tags.update(tag_list)
            
            stats = {
                "total_chunks": total_count,
                "source_files": sorted(list(source_files)),
                "unique_tags": sorted(list(all_tags)),
                "sample_size": len(sample.get("metadatas", [])),
                "status": "healthy"
            }
            
            logger.info(f"Collection stats: {total_count} chunks, {len(source_files)} files")
            return stats
            
        except Exception as e:
            logger.error(f"Error retrieving collection stats: {e}")
            return {"error": str(e), "status": "error"}
    
    def health_check(self) -> Tuple[bool, str]:
        """
        Perform a health check on the ChromaDB connection.
        
        Returns:
            Tuple[bool, str]: (is_healthy, status_message)
        """
        try:
            if not self.client or not self.collection:
                return False, "Database not initialized"
            
            count = self.collection.count()
            if count == 0:
                return False, "Knowledge base is empty"
            
            return True, f"Healthy - {count} documents available"
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False, f"Health check failed: {str(e)}"


class SkippyAssistant:
    """
    Skippy's AI personality and response generation.
    
    This class handles all AI interactions, including conversation management
    and response generation with the configured LLM.
    """
    
    # Comprehensive system prompt defining Skippy's personality and expertise
    SKIPPY_SYSTEM_PROMPT = """You are Skippy, a friendly SAP EWM coach and lead operator with years of warehouse management experience.

ROLE & EXPERTISE:
You are an expert SAP Extended Warehouse Management (EWM) coach and lead warehouse operator. You have deep knowledge of:
- SAP EWM configuration, setup, and customization
- Warehouse operations (inbound, outbound, internal processes)
- Integration with SAP modules (MM, SD, PP, QM, etc.)
- Troubleshooting EWM errors and system issues
- Best practices for warehouse management and optimization
- User training and process guidance
- RF device configuration and mobile warehouse processes

PERSONALITY TRAITS:
- Encouraging and supportive, like a helpful senior colleague
- Patient and understanding when explaining complex topics
- Uses a practical, hands-on approach to problem-solving
- Celebrates successes and helps users learn from challenges
- Occasionally uses relevant emojis to be warm and engaging (but not excessive)
- Always positive, solution-oriented, and professional
- Speaks in a conversational, approachable tone

RESPONSE STYLE & FORMAT:
- Break down complex topics into clear, manageable steps
- Provide practical, actionable advice with specific instructions
- Reference specific SAP EWM transactions, t-codes, and processes when relevant
- Give step-by-step instructions when appropriate
- For error handling, provide specific troubleshooting steps
- Ask clarifying questions when needed to provide better assistance
- Use bullet points, numbered lists, and clear headings for readability
- Include relevant transaction codes in parentheses (e.g., "Use MIGO transaction...")

EXAMPLE INTERACTIONS:
User: "How do I reverse a goods issue?"
Skippy: "Sure thing! Here's how to reverse a goods issue step by step:

1. **Access MIGO Transaction** (Transaction: MIGO)
   - Go to transaction MIGO in SAP
   - Select 'Goods Issue' document type
   
2. **Find the Original Document**
   - Enter the material document number you want to reverse
   - Or use the search function to find it
   
3. **Process the Reversal**
   - Click the 'Reversal' button
   - Verify the details are correct
   - Post the document

The system will automatically create a reversing document. Is there a specific error you're encountering with this process? 🤔"

User: "I'm getting error WM-123 in warehouse task processing"
Skippy: "That error can be tricky! Let me help you troubleshoot WM-123. Here's what usually helps:

**First, let's check the basics:**
1. Verify the warehouse task status in LE11
2. Check if the storage location is active
3. Ensure the material master data is complete

**Common causes and solutions:**
- Issue: Storage bin blocked → Solution: Check bin status in LS03
- Issue: Material availability → Solution: Verify stock in MMBE
- Issue: User authorization → Solution: Check authorization object /SCWM/WAR

What specifically happens when you encounter this error? Can you share the exact error message? This will help me give you more targeted guidance! 💪"

IMPORTANT GUIDELINES:
- Always be encouraging and supportive
- If you don't have specific information, acknowledge it honestly and suggest where to find more details
- For complex topics, offer to break them down into smaller parts
- Remember you're here to help warehouse operators succeed with SAP EWM
- When discussing technical configurations, always mention the importance of testing in a development environment first
- If discussing system changes, remind users to work with their SAP administrator"""

    def __init__(self, openai_api_key: str, model: str = None, temperature: float = None):
        """
        Initialize Skippy assistant.
        
        Args:
            openai_api_key (str): OpenAI API key for authentication.
            model (str, optional): OpenAI model to use. Defaults to Config.OPENAI_MODEL_DEFAULT.
            temperature (float, optional): Temperature for response generation. Defaults to Config.OPENAI_TEMPERATURE_DEFAULT.
        
        Raises:
            Exception: If OpenAI client initialization fails.
        """
        try:
            # Validate API key
            if not openai_api_key or not openai_api_key.strip():
                raise ValueError("OpenAI API key is required")
            
            # Initialize OpenAI client
            self.client = openai.OpenAI(api_key=openai_api_key.strip())
            self.model = model or Config.OPENAI_MODEL_DEFAULT
            self.temperature = temperature if temperature is not None else Config.OPENAI_TEMPERATURE_DEFAULT
            
            # Initialize retriever
            self.retriever = initialize_chroma_retriever()
            if self.retriever is None:
                logger.warning("ChromaDB retriever not available - responses will be without context")
            
            logger.info(f"Skippy assistant initialized with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Skippy assistant: {e}")
            raise
    
    def generate_response(self, user_question: str, conversation_history: List[Dict] = None) -> str:
        """
        Generate Skippy's response with context and conversation memory.
        
        Args:
            user_question (str): The user's question or message.
            conversation_history (List[Dict], optional): Previous conversation messages.
        
        Returns:
            str: Skippy's response to the user.
        """
        try:
            # Validate input
            if not user_question or not user_question.strip():
                return "🤔 I didn't catch that. Could you please ask me a question about SAP EWM?"
            
            logger.info(f"Generating response for question: '{user_question[:100]}...'")
            
            # Retrieve relevant context documents
            relevant_docs = []
            if self.retriever:
                try:
                    relevant_docs = self.retriever.retrieve_relevant_docs(
                        user_question.strip(), 
                        n_results=Config.DEFAULT_CONTEXT_DOCS
                    )
                except Exception as e:
                    logger.warning(f"Failed to retrieve context documents: {e}")
            
            # Format context for the LLM
            context = self._format_context(relevant_docs)
            
            # Build conversation messages
            messages = self._build_conversation_messages(
                user_question.strip(),
                context,
                conversation_history
            )
            
            # Generate response using OpenAI
            response = self._call_openai_api(messages)
            
            logger.info("Response generated successfully")
            return response
            
        except openai.AuthenticationError:
            logger.error("OpenAI authentication failed")
            return "🔑 Authentication failed. Please check your OpenAI API key in the sidebar settings."
        
        except openai.RateLimitError:
            logger.error("OpenAI rate limit exceeded")
            return "⏳ I'm currently experiencing high demand. Please wait a moment and try again."
        
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return "🤖 I'm having trouble with my AI services right now. Please try again in a few moments."
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return "🤔 I encountered an unexpected error. Could you try rephrasing your question or asking something else?"
    
    def _build_conversation_messages(self, user_question: str, context: str, conversation_history: List[Dict] = None) -> List[Dict]:
        """
        Build the conversation messages for the OpenAI API.
        
        Args:
            user_question (str): Current user question.
            context (str): Formatted context from retrieved documents.
            conversation_history (List[Dict], optional): Previous conversation messages.
        
        Returns:
            List[Dict]: Formatted messages for OpenAI API.
        """
        messages = [{"role": "system", "content": self.SKIPPY_SYSTEM_PROMPT}]
        
        # Add conversation history (limit to recent messages to manage token usage)
        if conversation_history:
            recent_history = conversation_history[-Config.MAX_CONVERSATION_HISTORY:]
            for msg in recent_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({
                        "role": msg["role"],
                        "content": str(msg["content"])
                    })
        
        # Add current question with context
        user_message_content = self._format_user_message(user_question, context)
        messages.append({"role": "user", "content": user_message_content})
        
        return messages
    
    def _format_user_message(self, user_question: str, context: str) -> str:
        """
        Format the user message with context information.
        
        Args:
            user_question (str): The user's question.
            context (str): Formatted context from documents.
        
        Returns:
            str: Formatted message for the LLM.
        """
        if context and context != "No specific documentation found for this query.":
            return f"""Context from SAP EWM documentation:
{context}

User Question: {user_question}

Please provide a helpful, step-by-step response based on the context above and your SAP EWM knowledge. If the context doesn't contain relevant information for this question, use your general SAP EWM expertise to provide a comprehensive answer."""
        else:
            return f"""User Question: {user_question}

Please provide a helpful, step-by-step response based on your SAP EWM knowledge and expertise."""
    
    def _call_openai_api(self, messages: List[Dict]) -> str:
        """
        Call the OpenAI API with the formatted messages.
        
        Args:
            messages (List[Dict]): Conversation messages.
        
        Returns:
            str: Response from OpenAI API.
        
        Raises:
            Various OpenAI exceptions for API-related errors.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=Config.OPENAI_MAX_TOKENS,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            else:
                raise Exception("Empty response from OpenAI API")
                
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _format_context(self, docs: List[Dict]) -> str:
        """
        Format retrieved documents into context for the LLM.
        
        Args:
            docs (List[Dict]): List of retrieved documents with metadata.
        
        Returns:
            str: Formatted context string.
        """
        if not docs:
            return "No specific documentation found for this query."
        
        context_parts = []
        for i, doc in enumerate(docs[:5], 1):  # Limit to top 5 results
            try:
                # Extract document information safely
                content = doc.get("content", "").strip()
                source = doc.get("source", "Unknown")
                page = doc.get("page", "N/A")
                section = doc.get("section", "N/A")
                similarity = doc.get("similarity", 0)
                tags = doc.get("tags", [])
                
                # Format tags
                tags_str = ", ".join(tags) if tags else "N/A"
                
                # Build context entry
                context_entry = f"""Document {i} (Source: {source}, Page: {page}, Relevance: {similarity:.1%}):
Section: {section}
Tags: {tags_str}

Content:
{content}

---"""
                context_parts.append(context_entry)
                
            except Exception as e:
                logger.warning(f"Error formatting document {i}: {e}")
                continue
        
        return "\n".join(context_parts)


def load_skippy_avatar() -> Optional[Image.Image]:
    """
    Load Skippy's avatar image from the assets directory.
    
    Returns:
        Optional[Image.Image]: Loaded avatar image or None if not found/loadable.
    """
    try:
        avatar_path = Path(Config.ASSETS_DIRECTORY) / "skippy.png"
        
        if avatar_path.exists():
            logger.info(f"Loading avatar from: {avatar_path}")
            return Image.open(avatar_path)
        else:
            logger.info(f"Avatar not found at: {avatar_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading avatar: {e}")
        return None


def setup_page_config():
    """Configure Streamlit page settings for optimal user experience."""
    st.set_page_config(
        page_title="Skippy - SAP EWM Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/skippy-ewm-assistant',
            'Report a bug': 'https://github.com/your-repo/skippy-ewm-assistant/issues',
            'About': """
            # Skippy - SAP EWM Assistant
            
            Your friendly AI coach for SAP Extended Warehouse Management!
            
            **Version:** 1.0.0  
            **Built with:** Streamlit, OpenAI, ChromaDB
            """
        }
    )


def initialize_session_state():
    """Initialize Streamlit session state variables with default values."""
    try:
        logger.info("Initializing session state...")
        
        # Chat messages with welcome message
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
            # Add Skippy's comprehensive welcome message
            welcome_msg = """👋 Hi there! I'm Skippy, your friendly SAP EWM coach and lead operator!

I'm here to help you succeed with **SAP Extended Warehouse Management**. With years of warehouse experience, I can assist you with:

🏭 **Warehouse Operations**
- Receiving and putaway processes
- Picking and packing workflows  
- Shipping and outbound logistics
- Inventory management and cycle counting

🔧 **System Configuration & Troubleshooting**
- EWM setup and customization
- Error resolution and debugging
- Transaction guidance and best practices
- Integration with other SAP modules (MM, SD, PP)

📱 **Mobile Warehouse Solutions**
- RF device configuration
- Mobile warehouse processes
- Handheld scanner operations

📚 **Training & Process Improvement**
- Step-by-step process guidance
- Best practice recommendations
- User training support
- Workflow optimization

**Ready to get started?** Ask me anything about SAP EWM! Here are some example questions:

• *"How do I reverse a goods receipt in MIGO?"*
• *"What should I do when I get error WM_TASK_001?"*  
• *"How do I configure storage types in EWM?"*
• *"Walk me through the wave planning process"*
• *"How do I set up RF menus for warehouse workers?"*

I'm here to provide practical, step-by-step guidance with a friendly approach. Let's make your SAP EWM operations run smoothly! 🚀"""
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": welcome_msg
            })
        
        # Application settings with environment defaults
        if "model_choice" not in st.session_state:
            st.session_state.model_choice = Config.OPENAI_MODEL_DEFAULT
        
        if "temperature" not in st.session_state:
            st.session_state.temperature = Config.OPENAI_TEMPERATURE_DEFAULT
        
        if "num_results" not in st.session_state:
            st.session_state.num_results = Config.DEFAULT_CONTEXT_DOCS
        
        # Error tracking
        if "last_error" not in st.session_state:
            st.session_state.last_error = None
            
        # Feature flags
        if "show_debug_info" not in st.session_state:
            st.session_state.show_debug_info = False
        
        logger.info("Session state initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing session state: {e}")
        st.error("Failed to initialize application. Please refresh the page.")


def render_sidebar():
    """Render the comprehensive sidebar with settings, status, and features."""
    with st.sidebar:
        # Main title and status
        st.title("🤖 Skippy Settings")
        
        # Knowledge base connection status
        render_knowledge_base_status()
        
        st.divider()
        
        # Model and AI settings
        render_model_settings()
        
        st.divider()
        
        # PDF upload and indexing
        render_pdf_upload_section()
        
        st.divider()
        
        # Conversation controls
        render_conversation_controls()
        
        st.divider()
        
        # Help and information
        render_help_section()


def render_knowledge_base_status():
    """Render the knowledge base connection status and statistics."""
    st.subheader("📚 Knowledge Base Status")
    
    try:
        # Get cached retriever
        retriever = initialize_chroma_retriever()
        
        if retriever:
            # Perform health check
            is_healthy, status_message = retriever.health_check()
            
            if is_healthy:
                st.success("✅ Knowledge Base Connected")
                
                # Get and display statistics
                with st.spinner("Loading statistics..."):
                    stats = retriever.get_collection_stats()
                
                if "error" not in stats:
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Documents", len(stats.get("source_files", [])))
                    with col2:
                        st.metric("Chunks", stats.get("total_chunks", 0))
                    
                    # Show indexed documents
                    if stats.get("source_files"):
                        with st.expander("📄 Indexed Documents"):
                            for doc in stats["source_files"]:
                                st.text(f"📄 {doc}")
                    
                    # Show available tags
                    if stats.get("unique_tags"):
                        with st.expander("🏷️ Content Tags"):
                            tag_cols = st.columns(2)
                            for i, tag in enumerate(stats["unique_tags"][:10]):  # Show first 10 tags
                                with tag_cols[i % 2]:
                                    st.text(f"• {tag}")
                else:
                    st.error("❌ Knowledge Base Error")
                    st.text(stats.get("error", "Unknown error"))
            else:
                st.warning(f"⚠️ {status_message}")
                st.text("Please run: `python build_index.py`")
        else:
            st.error("❌ Could not connect to knowledge base")
            st.text("Please run: `python build_index.py`")
            
    except Exception as e:
        logger.error(f"Error rendering knowledge base status: {e}")
        st.error("❌ Knowledge base connection error")
        st.text(f"Error: {str(e)}")


def render_model_settings():
    """Render AI model configuration settings."""
    st.subheader("⚙️ AI Model Settings")
    
    # API Key input with validation
    current_key = os.getenv("OPENAI_API_KEY", "")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=current_key,
        help="Enter your OpenAI API key. You can get one from https://platform.openai.com/api-keys",
        placeholder="sk-..."
    )
    
    # Validate and set API key
    if openai_api_key and openai_api_key != current_key:
        if openai_api_key.startswith("sk-"):
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("✅ API key updated")
        else:
            st.error("❌ Invalid API key format")
    
    # API key status
    if os.getenv("OPENAI_API_KEY"):
        st.success("🔑 API key configured")
    else:
        st.error("🔑 API key required")
        st.info("💡 Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)")
    
    # Model selection
    model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o"]
    current_model = st.session_state.get("model_choice", Config.OPENAI_MODEL_DEFAULT)
    
    try:
        default_index = model_options.index(current_model)
    except ValueError:
        default_index = 0
    
    st.session_state.model_choice = st.selectbox(
        "Model Choice",
        model_options,
        index=default_index,
        help="Choose the OpenAI model. GPT-4 provides better quality but costs more."
    )
    
    # Temperature control
    st.session_state.temperature = st.slider(
        "Response Creativity",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("temperature", Config.OPENAI_TEMPERATURE_DEFAULT),
        step=0.1,
        help="Lower values = more focused and consistent, Higher values = more creative and varied"
    )
    
    # Context documents
    st.session_state.num_results = st.slider(
        "Context Documents",
        min_value=1,
        max_value=5,
        value=st.session_state.get("num_results", Config.DEFAULT_CONTEXT_DOCS),
        help="Number of relevant documents to include as context for each response"
    )


def render_pdf_upload_section():
    """Render PDF upload and re-indexing functionality."""
    st.subheader("📄 Upload Training Documents")
    
    st.info("💡 Upload SAP EWM documentation to expand Skippy's knowledge base.")
    
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF documents containing SAP EWM training materials, manuals, or guides."
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} file(s) selected:")
        for file in uploaded_files:
            st.text(f"• {file.name} ({file.size:,} bytes)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Re-index Knowledge Base", type="primary"):
                reindex_knowledge_base(uploaded_files)
        
        with col2:
            if st.button("❌ Cancel Upload"):
                st.rerun()


def reindex_knowledge_base(uploaded_files):
    """
    Handle PDF upload and knowledge base re-indexing.
    
    Args:
        uploaded_files: List of uploaded PDF files from Streamlit.
    """
    try:
        with st.spinner("Processing PDFs and rebuilding knowledge base..."):
            # Create data directory
            data_dir = Path(Config.DATA_DIRECTORY)
            data_dir.mkdir(exist_ok=True)
            
            # Save uploaded files
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = data_dir / uploaded_file.name
                
                # Save file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_files.append(file_path)
                logger.info(f"Saved uploaded file: {file_path}")
            
            # Run build_index.py
            logger.info("Starting knowledge base re-indexing...")
            result = subprocess.run(
                [sys.executable, "build_index.py"],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            if result.returncode == 0:
                st.success("✅ Knowledge base updated successfully!")
                st.info(f"📁 Processed {len(saved_files)} files")
                logger.info("Knowledge base re-indexing completed successfully")
                
                # Clear retriever cache to force reload
                st.cache_resource.clear()
                
                # Refresh the page to reload data
                time.sleep(1)
                st.rerun()
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                st.error(f"❌ Error updating knowledge base:")
                st.code(error_msg, language="text")
                logger.error(f"Knowledge base re-indexing failed: {error_msg}")
                
    except Exception as e:
        error_msg = f"Error processing files: {str(e)}"
        st.error(f"❌ {error_msg}")
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())


def render_conversation_controls():
    """Render conversation management controls."""
    st.subheader("💬 Conversation")
    
    # Message count
    message_count = len(st.session_state.get("messages", []))
    st.metric("Messages", message_count)
    
    # Reset conversation button
    if st.button("🔄 Reset Conversation", help="Start a fresh conversation with Skippy"):
        reset_conversation()
    
    # Export conversation (future feature)
    if message_count > 1:
        with st.expander("📤 Export Options"):
            st.info("💡 Conversation export coming soon!")
            st.button("📄 Export as Text", disabled=True)
            st.button("📊 Export as PDF", disabled=True)


def reset_conversation():
    """Reset the conversation to initial state."""
    try:
        logger.info("Resetting conversation...")
        
        # Clear messages except for welcome message
        if "messages" in st.session_state:
            # Keep only the first message (welcome message)
            if st.session_state.messages:
                welcome_message = st.session_state.messages[0]
                st.session_state.messages = [welcome_message]
            else:
                st.session_state.messages = []
        
        # Clear any error states
        st.session_state.last_error = None
        
        st.success("✅ Conversation reset!")
        logger.info("Conversation reset successfully")
        
        # Force a rerun to refresh the UI
        time.sleep(0.5)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error resetting conversation: {e}")
        logger.error(f"Error resetting conversation: {e}")


def render_help_section():
    """Render help and information section."""
    with st.expander("ℹ️ About Skippy"):
        st.markdown("""
        ## 🤖 Skippy - SAP EWM Assistant
        
        **Your friendly AI coach for SAP Extended Warehouse Management!**
        
        ### 🎯 Expertise Areas
        - **Warehouse Operations**: Receiving, putaway, picking, packing, shipping
        - **System Configuration**: EWM setup, customization, master data
        - **Error Resolution**: Troubleshooting, debugging, system issues  
        - **Process Training**: Step-by-step guidance, best practices
        - **Integration**: MM, SD, PP, QM module connections
        - **Mobile Solutions**: RF devices, handheld operations
        
        ### 💡 Tips for Best Results
        - **Be specific**: Include transaction codes, error messages, exact processes
        - **Provide context**: Mention what you were trying to accomplish
        - **Ask follow-ups**: Don't hesitate to ask for clarification or more detail
        - **Include screenshots**: Describe what you see on screen (coming soon)
        
        ### 🚀 Example Questions
        ```
        • "How do I reverse a goods receipt in MIGO?"
        • "What does error WM_TASK_001 mean?"
        • "How do I configure storage types?"
        • "Walk me through wave planning"
        • "How do I set up RF menus?"
        ```
        
        ### 🔧 Technical Info
        - **Version**: 1.0.0
        - **Models**: GPT-3.5-turbo, GPT-4
        - **Knowledge Base**: ChromaDB vector database
        - **Updated**: Real-time with your uploads
        """)
    
    # Debug information (for development)
    if st.checkbox("🐛 Show Debug Info", help="Show technical debug information"):
        render_debug_info()


def render_debug_info():
    """Render debug information for troubleshooting."""
    st.subheader("🐛 Debug Information")
    
    try:
        debug_info = {
            "Session State Keys": list(st.session_state.keys()),
            "Environment Variables": {
                "OPENAI_API_KEY": "***SET***" if os.getenv("OPENAI_API_KEY") else "NOT SET",
                "CHROMA_DB_PATH": Config.CHROMA_DB_PATH,
                "CHROMA_COLLECTION_NAME": Config.CHROMA_COLLECTION_NAME,
                "DATA_DIRECTORY": Config.DATA_DIRECTORY,
            },
            "File System": {
                "Current Directory": str(Path.cwd()),
                "Data Directory Exists": Path(Config.DATA_DIRECTORY).exists(),
                "ChromaDB Directory Exists": Path(Config.CHROMA_DB_PATH).exists(),
                "Assets Directory Exists": Path(Config.ASSETS_DIRECTORY).exists(),
            },
            "Configuration": {
                "Default Model": Config.OPENAI_MODEL_DEFAULT,
                "Default Temperature": Config.OPENAI_TEMPERATURE_DEFAULT,
                "Max Tokens": Config.OPENAI_MAX_TOKENS,
                "Embedding Model": Config.EMBEDDING_MODEL,
            }
        }
        
        st.json(debug_info)
        
    except Exception as e:
        st.error(f"Error generating debug info: {e}")


def render_chat_interface():
    """Render the main chat interface with enhanced UX."""
    
    # Header section with avatar and title
    render_chat_header()
    
    # Chat messages container with custom styling
    render_chat_messages()
    
    # Chat input and response handling
    handle_chat_input()


def render_chat_header():
    """Render the chat interface header with avatar and branding."""
    # Main header with avatar
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        avatar = load_skippy_avatar()
        if avatar:
            st.image(avatar, width=80)
        else:
            st.markdown("""
            <div style="text-align: center; font-size: 60px; line-height: 80px;">
                🤖
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.title("Skippy - SAP EWM Assistant")
        st.caption("Your friendly SAP Extended Warehouse Management coach and lead operator")
        
        # Status indicators
        col_status1, col_status2, col_status3 = st.columns(3)
        
        with col_status1:
            if os.getenv("OPENAI_API_KEY"):
                st.success("🔑 API Ready")
            else:
                st.error("🔑 API Key Needed")
        
        with col_status2:
            retriever = initialize_chroma_retriever()
            if retriever:
                is_healthy, _ = retriever.health_check()
                if is_healthy:
                    st.success("📚 Knowledge Ready")
                else:
                    st.warning("📚 No Documents")
            else:
                st.error("📚 DB Error")
        
        with col_status3:
            model = st.session_state.get("model_choice", Config.OPENAI_MODEL_DEFAULT)
            st.info(f"🧠 {model}")
    
    with col3:
        # Quick actions
        if st.button("🔄", help="Reset Conversation"):
            reset_conversation()
    
    st.divider()


def render_chat_messages():
    """Render chat messages with improved styling and scrolling."""
    
    # Custom CSS for better chat appearance
    st.markdown("""
    <style>
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display chat messages
    messages = st.session_state.get("messages", [])
    
    if not messages:
        st.info("👋 Start a conversation with Skippy by typing a question below!")
        return
    
    for i, message in enumerate(messages):
        try:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Determine avatar
            if role == "assistant":
                avatar_image = load_skippy_avatar()
                avatar = avatar_image if avatar_image else "🤖"
            else:
                avatar = "👤"
            
            # Display message with context
            with st.chat_message(role, avatar=avatar):
                st.markdown(content)
                
                # Add metadata for assistant messages
                if role == "assistant" and i > 0:  # Skip welcome message
                    with st.expander("ℹ️ Response Details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.text(f"Model: {st.session_state.get('model_choice', 'N/A')}")
                        with col2:
                            st.text(f"Temperature: {st.session_state.get('temperature', 'N/A')}")
                        with col3:
                            st.text(f"Context Docs: {st.session_state.get('num_results', 'N/A')}")
        
        except Exception as e:
            logger.error(f"Error rendering message {i}: {e}")
            continue


def handle_chat_input():
    """Handle chat input and response generation."""
    
    # Chat input with placeholder
    prompt = st.chat_input(
        "Ask Skippy anything about SAP EWM...",
        disabled=not os.getenv("OPENAI_API_KEY")
    )
    
    if prompt:
        handle_user_message(prompt)


def handle_user_message(user_message: str):
    """
    Handle user message and generate response.
    
    Args:
        user_message (str): The user's input message.
    """
    try:
        # Validate API key
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ Please enter your OpenAI API key in the sidebar settings.")
            return
        
        # Add user message to conversation
        st.session_state.messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_message)
        
        # Generate and display Skippy's response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Skippy is thinking..."):
                try:
                    # Initialize Skippy with current settings
                    skippy = SkippyAssistant(
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                        model=st.session_state.get("model_choice", Config.OPENAI_MODEL_DEFAULT),
                        temperature=st.session_state.get("temperature", Config.OPENAI_TEMPERATURE_DEFAULT)
                    )
                    
                    # Generate response with conversation history
                    response = skippy.generate_response(
                        user_message,
                        conversation_history=st.session_state.messages[:-1]  # Exclude current user message
                    )
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add response to conversation history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Clear any previous errors
                    st.session_state.last_error = None
                    
                    logger.info(f"Successfully processed user message: '{user_message[:50]}...'")
                    
                except Exception as e:
                    error_msg = f"🤔 I encountered an error while processing your question: {str(e)}"
                    st.error(error_msg)
                    
                    # Add error to conversation
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Store error for debugging
                    st.session_state.last_error = {
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    logger.error(f"Error processing user message: {e}")
                    logger.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        logger.error(f"Unexpected error in handle_user_message: {e}")
        logger.error(traceback.format_exc())


def render_footer():
    """Render application footer with tips and information."""
    st.divider()
    
    # Pro tips and footer information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 💡 Pro Tips
        - **Be specific** with your SAP EWM questions
        - **Include transaction codes** (e.g., MIGO, LT01)
        - **Mention exact error messages** for better help
        - **Ask follow-up questions** for clarification
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Quick Actions
        - Upload PDFs to expand knowledge base
        - Adjust AI model settings in sidebar
        - Reset conversation for fresh start
        - Check knowledge base status
        """)
    
    # Application information
    st.markdown("""
    ---
    **Skippy SAP EWM Assistant** v1.0.0 | Built with ❤️ using Streamlit, OpenAI, and ChromaDB
    """)


def main():
    """Main application entry point with comprehensive error handling."""
    try:
        logger.info("Starting Skippy application...")
        
        # Setup application
        setup_page_config()
        
        # Initialize session state
        initialize_session_state()
        
        # Render main interface
        col1, col2 = st.columns([1, 3])
        
        with col1:
            render_sidebar()
        
        with col2:
            render_chat_interface()
        
        # Render footer
        render_footer()
        
        logger.info("Application rendered successfully")
        
    except Exception as e:
        logger.critical(f"Critical error in main application: {e}")
        logger.critical(traceback.format_exc())
        
        st.error("🚨 Critical Application Error")
        st.exception(e)
        
        if st.button("🔄 Restart Application"):
            st.rerun()


if __name__ == "__main__":
    main()
