#!/usr/bin/env python3
"""
Skippy - SAP EWM Coach (Clean & Maintainable)

A simplified, maintainable coaching-style SAP EWM assistant that provides guided assistance
with clean architecture and readable code structure.

Features:
- Simplified class structure
- Single parameterized prompt template
- Clean function decomposition
- Proper error handling
- Minimal UI bloat

Author: AI Development Team
Version: 3.0.0 (Clean Edition)
License: MIT
"""

import streamlit as st
import os
import re
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Core imports from your working setup
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings 
from langchain_chroma import Chroma 
from langchain_core.documents import Document 
from langchain_core.prompts import ChatPromptTemplate 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains import create_retrieval_chain 
from langchain_core.callbacks import CallbackManagerForRetrieverRun 
from langchain_core.retrievers import BaseRetriever

# Constants
MAX_CLARIFICATIONS = 5
DOC_RETRIEVAL_COUNT = 3
LLM_TEMPERATURE = 0.1

class ConversationStage(Enum):
    INITIAL = "initial"
    CLARIFYING = "clarifying" 
    ANSWERING = "answering"

class IntentType(Enum):
    LEARNING = "learning"
    NAVIGATION = "navigation"
    ERROR = "error"

# Simplified conversation state
@dataclass
class Conversation:
    original_question: str = ""
    clarifications_asked: List[str] = field(default_factory=list)
    detected_intent: str = ""
    detected_role: str = ""
    stage: ConversationStage = ConversationStage.INITIAL
    
    def add_clarification(self, clarification: str):
        """Add a clarification question we've asked"""
        if clarification not in self.clarifications_asked:
            self.clarifications_asked.append(clarification)
    
    def has_asked_about(self, topic: str) -> bool:
        """Check if we've asked about a specific topic"""
        return any(topic.lower() in q.lower() for q in self.clarifications_asked)
    
    def get_context_summary(self) -> str:
        """Get conversation context for LLM"""
        parts = [f"Original Question: {self.original_question}"]
        if self.clarifications_asked:
            parts.append(f"Clarifications asked: {len(self.clarifications_asked)}")
        return " | ".join(parts)
    
    def reset(self):
        """Reset for new conversation"""
        self.original_question = ""
        self.clarifications_asked = []
        self.detected_intent = ""
        self.detected_role = ""
        self.stage = ConversationStage.INITIAL

# Error handling decorator
def handle_errors(fallback_message: str = "I encountered an issue. Please try rephrasing your question."):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"Error in {func.__name__}: {str(e)}")
                traceback.print_exc()
                return fallback_message
        return wrapper
    return decorator

# Custom retriever (from your working setup)
class SimpleRetriever(BaseRetriever):
    documents: List[Document]
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.documents

# Simplified coaching system
class SkippyCoach:
    """Main coaching system - simplified and maintainable"""
    
    def __init__(self):
        self.embeddings = self._init_embeddings()
        self.llm = self._init_llm()
        self.vector_db = self._init_vector_db()
        self.prompt_template = self._create_unified_prompt()
    
    def _init_embeddings(self):
        """Initialize embeddings with your working config"""
        return AzureOpenAIEmbeddings(
            base_url="https://genaiapimna.jnj.com/openai-embeddings/openai",
            openai_api_key='f89d10a91b9d4cc989085a495d695eb3',
            api_version="2022-12-01",
            model="text-embedding-ada-002",
            openai_api_type="azure"
        )
    
    def _init_llm(self):
        """Initialize LLM with your working config"""
        os.environ["OPENAI_API_KEY"] = '3c44ff2f28874e6f9d74dd79e89a8093'
        return AzureChatOpenAI(
            azure_endpoint="https://genaiapimna.jnj.com/openai-chat",
            api_key='acd5a7d2b4d64ea6871aeb4cbc3113dd',
            api_version="2023-05-15",
            deployment_name="gpt-4o",
            temperature=LLM_TEMPERATURE,
            streaming=True
        )
    
    def _init_vector_db(self):
        """Initialize vector database"""
        if not Path('./data/eWMDB').exists():
            st.error("❌ ChromaDB not found at /data/eWMDB")
            st.info("Please run the index builder first.")
            st.stop()
        
        return Chroma(
            persist_directory='./data/eWMDB', 
            embedding_function=self.embeddings
        )
    
    def _create_unified_prompt(self):
        """Single parameterized prompt template - no duplication"""
        return ChatPromptTemplate.from_template("""
You are Skippy, an SAP EWM Coach providing {coaching_mode} guidance.

Coaching Style for {coaching_mode}:
{coaching_instructions}

Conversation Context: {conversation_context}
User Role: {user_role}
User Input: {input}

SAP EWM Documentation Context:
{context}

Provide focused, step-by-step guidance:
""")
    
    @handle_errors("I'm having trouble processing your question. Could you rephrase it?")
    def ask(self, question: str, conversation: Conversation) -> str:
        """Main entry point - simplified pipeline"""
        # Step 1: Set up conversation if new
        if conversation.stage == ConversationStage.INITIAL:
            self._setup_new_conversation(question, conversation)
        
        # Step 2: Get relevant documents
        docs = self._retrieve_documents(conversation.original_question or question)
        
        # Step 3: Check for clarification need
        clarification = self._check_clarification_need(question, docs, conversation)
        if clarification:
            conversation.stage = ConversationStage.CLARIFYING
            return clarification
        
        # Step 4: Generate coaching response
        conversation.stage = ConversationStage.ANSWERING
        return self._generate_response(question, docs, conversation)
    
    def _setup_new_conversation(self, question: str, conversation: Conversation):
        """Initialize new conversation context"""
        conversation.original_question = question
        conversation.detected_intent = self._classify_intent(question)
        conversation.detected_role = self._detect_role(question)
        conversation.stage = ConversationStage.INITIAL
    
    def _classify_intent(self, text: str) -> str:
        """Simple intent classification using regex patterns"""
        text_lower = text.lower()
        
        error_patterns = [r'\b(error|issue|problem|failed|stuck|cannot)\b']
        navigation_patterns = [r'\b(next step|what now|where|currently|completed)\b']
        learning_patterns = [r'\b(what is|how does|explain|tell me|learn)\b']
        
        for pattern in error_patterns:
            if re.search(pattern, text_lower):
                return IntentType.ERROR.value
        
        for pattern in navigation_patterns:
            if re.search(pattern, text_lower):
                return IntentType.NAVIGATION.value
        
        for pattern in learning_patterns:
            if re.search(pattern, text_lower):
                return IntentType.LEARNING.value
        
        return IntentType.LEARNING.value  # Default
    
    def _detect_role(self, text: str) -> str:
        """Simple role detection using keyword scoring"""
        text_lower = text.lower()
        
        config_keywords = ['configure', 'setup', 'customize', 'define', 'maintain']
        supervisor_keywords = ['monitor', 'check', 'verify', 'review', 'approve']
        execution_keywords = ['execute', 'perform', 'do', 'create', 'process']
        
        config_score = sum(1 for word in config_keywords if word in text_lower)
        supervisor_score = sum(1 for word in supervisor_keywords if word in text_lower)
        execution_score = sum(1 for word in execution_keywords if word in text_lower)
        
        if config_score > max(supervisor_score, execution_score):
            return "configuration"
        elif supervisor_score > execution_score:
            return "supervisor"
        else:
            return "execution"
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents from vector store"""
        return self.vector_db.similarity_search(query, k=DOC_RETRIEVAL_COUNT)
    
    def _check_clarification_need(self, question: str, docs: List[Document], conversation: Conversation) -> Optional[str]:
        """Check if clarification is needed - simplified logic"""
        if len(conversation.clarifications_asked) >= MAX_CLARIFICATIONS or not docs:
            return None
        
        # Extract process areas from documents
        process_areas = set()
        for doc in docs:
            content = doc.page_content.lower()
            areas = ['inbound', 'outbound', 'picking', 'putaway', 'packing', 'shipping']
            for area in areas:
                if area in content:
                    process_areas.add(area)
        
        # Intent-specific clarifications
        intent = conversation.detected_intent
        original = conversation.original_question
        
        if intent == IntentType.LEARNING.value and len(process_areas) > 1 and not conversation.has_asked_about("process area"):
            areas_list = ', '.join(sorted(process_areas))
            clarification = f"I found information about multiple areas: {areas_list}. Which area do you want to focus on for '{original}'?"
            conversation.add_clarification("process area")
            return f"🎯 **Coaching Question:** {clarification}"
        
        if intent == IntentType.NAVIGATION.value and 'transaction' not in question.lower() and not conversation.has_asked_about("current location"):
            clarification = f"To guide your next step for '{original}', which transaction or screen are you currently in?"
            conversation.add_clarification("current location")
            return f"🎯 **Coaching Question:** {clarification}"
        
        if intent == IntentType.ERROR.value and not any(word in question.lower() for word in ['error', 'message']) and not conversation.has_asked_about("error details"):
            clarification = f"To help resolve '{original}', what specific error message or code are you seeing?"
            conversation.add_clarification("error details")
            return f"🎯 **Coaching Question:** {clarification}"
        
        return None
    
    def _generate_response(self, question: str, docs: List[Document], conversation: Conversation) -> str:
        """Generate coaching response using unified prompt"""
        if not docs:
            return self._generate_no_context_response(conversation)
        
        # Get coaching instructions based on intent
        coaching_instructions = self._get_coaching_instructions(conversation.detected_intent)
        
        # Create retriever and chains
        retriever = SimpleRetriever(documents=docs)
        document_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Generate response with context
        response = retrieval_chain.invoke({
            "input": question,
            "coaching_mode": conversation.detected_intent,
            "coaching_instructions": coaching_instructions,
            "conversation_context": conversation.get_context_summary(),
            "user_role": conversation.detected_role
        })
        
        return self._format_response(response["answer"], conversation, docs)
    
    def _get_coaching_instructions(self, intent: str) -> str:
        """Get coaching instructions for different intents"""
        instructions = {
            IntentType.LEARNING.value: "Give ONE clear explanation at a time. Use simple language. Provide concrete examples.",
            IntentType.NAVIGATION.value: "Focus ONLY on the immediate next step. Be specific about which screen/transaction to use.",
            IntentType.ERROR.value: "Provide ONE troubleshooting step at a time. Explain what each step accomplishes."
        }
        return instructions.get(intent, instructions[IntentType.LEARNING.value])
    
    def _generate_no_context_response(self, conversation: Conversation) -> str:
        """Fallback response when no context is found"""
        intent = conversation.detected_intent
        role = conversation.detected_role
        original = conversation.original_question
        
        base_message = f"🤔 I don't have specific information about that topic"
        if original:
            base_message += f" regarding '{original}'"
        
        suggestions = {
            IntentType.LEARNING.value: f"Could you ask about a specific SAP EWM process or transaction?",
            IntentType.NAVIGATION.value: f"Please tell me which transaction you're in and your last completed step.",
            IntentType.ERROR.value: f"Please share the specific error message and which transaction you were using."
        }
        
        suggestion = suggestions.get(intent, suggestions[IntentType.LEARNING.value])
        return f"{base_message}.\n\nFor {role} tasks: {suggestion}"
    
    def _format_response(self, answer: str, conversation: Conversation, docs: List[Document]) -> str:
        """Format final response with context info"""
        # Add mode and role badges
        mode_icons = {
            IntentType.LEARNING.value: "📚 **Learning Mode**",
            IntentType.NAVIGATION.value: "🧭 **Process Navigation**",
            IntentType.ERROR.value: "🚨 **Problem Resolution**"
        }
        
        role_icons = {
            "execution": "👩‍💼",
            "supervisor": "👨‍💼", 
            "configuration": "⚙️"
        }
        
        mode_badge = mode_icons.get(conversation.detected_intent, "🤖")
        role_badge = role_icons.get(conversation.detected_role, "👤")
        
        response = f"{mode_badge} | {role_badge} **{conversation.detected_role.title()}**\n\n{answer}"
        
        # Add memory note if different from original
        if conversation.original_question and conversation.original_question not in answer:
            response += f"\n\n*💭 Context: {conversation.original_question}*"
        
        # Add source info (minimal)
        if len(docs) > 1:
            sources = list(set([Path(doc.metadata.get('source', 'Unknown')).name for doc in docs[:2]]))
            if sources:
                response += f"\n\n*📄 Sources: {', '.join(sources)}*"
        
        return response

# Simplified UI functions
def setup_page():
    """Configure Streamlit page with minimal styling"""
    st.set_page_config(
        page_title="Skippy - SAP EWM Coach (Clean)",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Minimal CSS - only essential styling
    st.markdown("""
    <style>
    .user-message {
        background-color: #007acc; color: white; padding: 0.8rem;
        border-radius: 10px; margin: 0.5rem 0; text-align: right;
    }
    .coach-message {
        background-color: #f8f9fa; color: #333; padding: 0.8rem;
        border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #28a745;
    }
    .memory-badge {
        background-color: #e3f2fd; border: 1px solid #1976d2;
        border-radius: 8px; padding: 0.5rem; margin: 0.5rem 0;
        font-size: 0.85rem; color: #1976d2;
    }
    </style>
    """, unsafe_allow_html=True)

def show_memory_indicator(conversation: Conversation):
    """Show current conversation memory"""
    if conversation.original_question:
        clarification_info = f" | 🎯 {len(conversation.clarifications_asked)}/{MAX_CLARIFICATIONS}" if conversation.clarifications_asked else ""
        st.markdown(f"""
        <div class="memory-badge">
        🧠 <strong>Coaching:</strong> {conversation.original_question}{clarification_info}
        </div>
        """, unsafe_allow_html=True)

def display_chat_message(message: Dict[str, str]):
    """Display a single chat message"""
    if message["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="coach-message"><strong>🧠 Coach Skippy:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)

def create_sidebar(conversation: Conversation):
    """Create simplified sidebar"""
    with st.sidebar:
        st.header("🧠 Smart Coaching")
        st.markdown("""
        **Memory Features:**
        - Remembers original questions
        - Avoids repetitive clarifications
        - Tracks conversation context
        """)
        
        if conversation.original_question:
            st.header("🎯 Current Session")
            st.write(f"**Question:** {conversation.original_question[:30]}...")
            st.write(f"**Intent:** {conversation.detected_intent.title()}")
            st.write(f"**Role:** {conversation.detected_role.title()}")
            st.write(f"**Stage:** {conversation.stage.value.title()}")
        
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.conversation.reset()
            st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome
            st.rerun()

# Main application
@st.cache_resource
def init_coach():
    """Initialize the coaching system"""
    return SkippyCoach()

def main():
    """Main application - clean and focused"""
    setup_page()
    
    st.title("🧠 Skippy - SAP EWM Coach (Clean Edition)")
    st.markdown("*Clean, maintainable coaching with intelligent memory*")
    
    # Initialize coach and conversation
    coach = init_coach()
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = Conversation()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant", 
            "content": """👋 **Hello! I'm Skippy, your Clean SAP EWM Coach!**

I'm built with clean, maintainable code and smart memory:

🧠 **Smart Memory:** Remembers context throughout our conversation
🎯 **Focused Coaching:** One clear direction at a time  
📚 **Progressive Guidance:** Builds on previous exchanges
🚀 **Clean Architecture:** Maintainable and readable code

**What SAP EWM challenge can I help you with today?**"""
        }]
    
    conversation = st.session_state.conversation
    
    # Show memory indicator
    show_memory_indicator(conversation)
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message)
    
    # Chat input
    st.markdown("---")
    with st.form("coach_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Your SAP EWM question...",
                placeholder="e.g., I'm stuck in putaway confirmation",
                key="question",
                label_visibility="collapsed"
            )
        
        with col2:
            submit = st.form_submit_button("Ask Coach 🧠", use_container_width=True)
    
    # Process input
    if submit and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("🧠 Coaching..."):
            response = coach.ask(user_input, conversation)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Sidebar
    create_sidebar(conversation)

if __name__ == "__main__":
    main()
