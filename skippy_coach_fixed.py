#!/usr/bin/env python3
"""
Skippy - SAP EWM Coach (Fixed - Memory & Context Aware)

A coaching-style SAP EWM assistant that maintains conversation context and avoids
repetitive clarification questions. This version properly tracks the original question
and previous clarifications to provide intelligent, progressive guidance.

Features:
- Conversation context tracking
- Smart clarification question detection (no repeats)
- Original question retention throughout clarification loops
- Progressive coaching based on accumulated context
- Intent classification with memory
- Role-aware guidance with context continuity

Author: AI Development Team
Version: 2.1.0 (Memory-Fixed Coach Edition)
License: MIT
"""

import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
import re

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

import traceback

# Custom retriever from your working setup 
class MyRetriever(BaseRetriever):
    """Custom retriever that returns documents as-is"""
        
    documents: List[Document]
    """List of documents to retrieve from."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        return self.documents

# Configuration class with your exact working values 
class SkippyConfig:
    """Skippy configuration using your proven Azure OpenAI setup"""

    # Embedding configuration - your exact working values
    EMBEDDING_ENDPOINT = "https://genaiapimna.jnj.com/openai-embeddings/openai"
    EMBEDDING_API_KEY = "f89d10a91b9d4cc989085a495d695eb3"
    EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"
    EMBEDDING_API_VERSION = "2022-12-01"

    # Chat LLM configuration - your exact working values
    CHAT_ENDPOINT = "https://genaiapimna.jnj.com/openai-chat"
    CHAT_API_KEY = "acd5a7d2b4d64ea6871aeb4cbc3113dd"
    CHAT_DEPLOYMENT = "gpt-4o"
    CHAT_API_VERSION = "2023-05-15"

    # Database configuration
    DB_PATH = "./data/eWMDB"

# Conversation Context Manager
class ConversationContext:
    """Manages conversation context and memory across clarification loops"""
    
    def __init__(self):
        self.original_question = ""
        self.clarifications_asked = []
        self.clarifications_received = {}
        self.detected_intent = ""
        self.detected_role = ""
        self.relevant_docs = []
        self.conversation_stage = "initial"  # initial, clarifying, answering
        
    def set_original_question(self, question: str, intent: str, role: str):
        """Set the original question and context"""
        self.original_question = question
        self.detected_intent = intent
        self.detected_role = role
        self.conversation_stage = "initial"
        
    def add_clarification_asked(self, clarification: str):
        """Track clarification questions we've asked"""
        if clarification not in self.clarifications_asked:
            self.clarifications_asked.append(clarification)
            
    def add_clarification_received(self, key: str, value: str):
        """Store clarification answers"""
        self.clarifications_received[key] = value
        
    def get_full_context(self) -> str:
        """Build full context string for LLM"""
        context_parts = [f"Original Question: {self.original_question}"]
        
        if self.clarifications_received:
            context_parts.append("Clarifications provided:")
            for key, value in self.clarifications_received.items():
                context_parts.append(f"- {key}: {value}")
                
        return "\n".join(context_parts)
        
    def has_asked_clarification(self, clarification_type: str) -> bool:
        """Check if we've already asked a specific type of clarification"""
        return any(clarification_type.lower() in asked.lower() for asked in self.clarifications_asked)
        
    def reset(self):
        """Reset context for new conversation"""
        self.original_question = ""
        self.clarifications_asked = []
        self.clarifications_received = {}
        self.detected_intent = ""
        self.detected_role = ""
        self.relevant_docs = []
        self.conversation_stage = "initial"

# Intent Classification System (Enhanced)
class IntentClassifier:
    """Classifies user intent to determine coaching approach"""
    
    def __init__(self):
        # Learning mode patterns
        self.learning_patterns = [
            r'\b(what is|what are|how does|how do|explain|definition|overview)\b',
            r'\b(tell me about|describe|understand|learn)\b',
            r'\b(difference between|compare|versus)\b'
        ]
        
        # Process navigation patterns
        self.navigation_patterns = [
            r'\b(stuck|help|next step|what now|where do|how to proceed)\b',
            r'\b(I am at|I am in|currently in|transaction)\b',
            r'\b(completed|finished|done with)\b'
        ]
        
        # Error/blocker patterns
        self.error_patterns = [
            r'\b(error|issue|problem|failed|not working)\b',
            r'\b(blocked|stuck|cannot|can\'t|unable)\b',
            r'\b(message|warning|alert|exception)\b'
        ]
    
    def classify_intent(self, user_input: str, context: ConversationContext) -> str:
        """Classify user intent based on input patterns and context"""
        user_lower = user_input.lower()
        
        # If we're in clarification stage and user is answering, maintain original intent
        if context.conversation_stage == "clarifying" and context.detected_intent:
            return context.detected_intent
        
        # Check for error patterns first (highest priority)
        for pattern in self.error_patterns:
            if re.search(pattern, user_lower):
                return "error"
        
        # Check for navigation patterns
        for pattern in self.navigation_patterns:
            if re.search(pattern, user_lower):
                return "navigation"
        
        # Check for learning patterns
        for pattern in self.learning_patterns:
            if re.search(pattern, user_lower):
                return "learning"
        
        # Default to learning mode
        return "learning"

# Role Detection System (Enhanced)
class RoleDetector:
    """Detects user role to adapt coaching style"""
    
    def __init__(self):
        self.execution_keywords = ['execute', 'perform', 'do', 'create', 'process', 'scan', 'confirm']
        self.supervisor_keywords = ['monitor', 'check', 'verify', 'review', 'approve', 'status', 'overview']
        self.config_keywords = ['configure', 'setup', 'customize', 'define', 'maintain', 'assign', 'table']
    
    def detect_role(self, user_input: str, chat_history: List[Dict], context: ConversationContext) -> str:
        """Detect user role from input, chat history, and context"""
        # If we already detected role in this conversation, maintain it
        if context.conversation_stage in ["clarifying", "answering"] and context.detected_role:
            return context.detected_role
            
        user_lower = user_input.lower()
        
        # Check recent chat history for role context
        recent_messages = chat_history[-5:] if len(chat_history) >= 5 else chat_history
        full_context = " ".join([msg.get("content", "") for msg in recent_messages]) + " " + user_input
        full_context_lower = full_context.lower()
        
        config_score = sum(1 for word in self.config_keywords if word in full_context_lower)
        supervisor_score = sum(1 for word in self.supervisor_keywords if word in full_context_lower)
        execution_score = sum(1 for word in self.execution_keywords if word in full_context_lower)
        
        if config_score > max(supervisor_score, execution_score):
            return "configuration"
        elif supervisor_score > execution_score:
            return "supervisor"
        else:
            return "execution"

# Smart Context Analyzer (Enhanced)
class ContextAnalyzer:
    """Analyzes retrieved documents and conversation context for intelligent clarification"""
    
    def needs_clarification(self, user_input: str, relevant_docs: List[Document], 
                          intent: str, context: ConversationContext) -> Tuple[bool, Optional[str]]:
        """Determine if clarification is needed based on context and conversation history"""
        
        # Don't ask for clarification if we're at max limit
        if len(context.clarifications_asked) >= 5:
            return False, None
            
        if not relevant_docs:
            return False, None
        
        # Extract unique process areas from documents
        process_areas = set()
        transaction_codes = set()
        
        for doc in relevant_docs:
            content = doc.page_content.lower()
            
            # Extract process areas
            if 'inbound' in content:
                process_areas.add('inbound')
            if 'outbound' in content:
                process_areas.add('outbound')
            if 'picking' in content:
                process_areas.add('picking')
            if 'putaway' in content:
                process_areas.add('putaway')
            if 'packing' in content:
                process_areas.add('packing')
            if 'shipping' in content:
                process_areas.add('shipping')
            
            # Extract transaction codes
            tcode_matches = re.findall(r'/n\w+|/o\w+|\b\w{2,4}\b(?=\s*[:-].*transaction)', content)
            transaction_codes.update(tcode_matches)
        
        # Learning mode clarifications
        if intent == "learning":
            if len(process_areas) > 1 and not context.has_asked_clarification("process area"):
                areas_list = ', '.join(sorted(process_areas))
                clarification = f"I found information about multiple warehouse areas: {areas_list}. Which specific area would you like to focus on for your question about '{context.original_question or user_input}'?"
                context.add_clarification_asked("process area")
                return True, clarification
        
        # Navigation mode clarifications
        elif intent == "navigation":
            if ('transaction' not in user_input.lower() and 'tcode' not in user_input.lower() 
                and not context.has_asked_clarification("current location")):
                clarification = f"To guide you from where you are with '{context.original_question or user_input}', I need to know: Which transaction or Fiori app are you currently working in?"
                context.add_clarification_asked("current location")
                return True, clarification
        
        # Error mode clarifications
        elif intent == "error":
            if (not any(keyword in user_input.lower() for keyword in ['error', 'message', 'exception', 'code'])
                and not context.has_asked_clarification("error details")):
                clarification = f"To help resolve the issue you mentioned: '{context.original_question or user_input}', could you share the specific error message or code you're seeing?"
                context.add_clarification_asked("error details")
                return True, clarification
        
        return False, None

# Enhanced Coaching Response Generator
class CoachingResponseGenerator:
    """Generates coaching-style responses with full conversation context"""
    
    def __init__(self, llm, prompt_template):
        self.llm = llm
        self.base_prompt = prompt_template
        
        # Enhanced coaching prompts with context awareness
        self.coaching_prompts = {
            "learning": ChatPromptTemplate.from_template("""
You are Skippy, an SAP EWM Coach. The user wants to learn about something, and you have their full conversation context.

Coaching Style:
- Give ONE clear explanation or concept at a time
- Use simple, practical language
- Focus on what they need to know for their role: {user_role}
- Reference their original question and any clarifications they provided
- Provide concrete examples when helpful

Conversation Context:
{conversation_context}

User Role: {user_role}
Current Input: {input}
Context from SAP EWM documentation:

{context}

Provide a focused, coaching-style response that explains the concept clearly, keeping in mind their original question and clarifications:
"""),

            "navigation": ChatPromptTemplate.from_template("""
You are Skippy, an SAP EWM Coach. The user needs guidance on their next step in a process, and you have their conversation history.

Coaching Style:
- Focus ONLY on the immediate next step
- Be specific about which screen/transaction/button to use
- Reference their original question and any clarifications they provided
- Ask for confirmation they completed the step before continuing
- Adapt guidance for their role: {user_role}

Conversation Context:
{conversation_context}

User Role: {user_role}
Current Input: {input}
Context from SAP EWM documentation:

{context}

Provide the specific next step they should take, keeping their original question and situation in mind:
"""),

            "error": ChatPromptTemplate.from_template("""
You are Skippy, an SAP EWM Coach. The user has encountered an issue or error, and you have their full context.

Coaching Style:
- Provide ONE troubleshooting step at a time
- Explain what each step will accomplish
- Reference their original problem and any additional details they provided
- Adapt solutions for their role: {user_role}

Conversation Context:
{conversation_context}

User Role: {user_role}
Current Input: {input}
Context from SAP EWM documentation:

{context}

Provide focused troubleshooting guidance, addressing their original problem with the context they've provided:
""")
        }
    
    def generate_response(self, intent: str, user_input: str, relevant_docs: List[Document], 
                         user_role: str, context: ConversationContext) -> str:
        """Generate coaching response with full conversation context"""
        
        if not relevant_docs:
            return self._generate_no_context_response(intent, user_input, user_role, context)
        
        # Use intent-specific prompt
        prompt = self.coaching_prompts.get(intent, self.base_prompt)
        
        # Create retriever with documents
        retriever = MyRetriever(documents=relevant_docs)
        
        # Create chains
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Generate response with full context
        response = retrieval_chain.invoke({
            "input": user_input,
            "user_role": user_role,
            "conversation_context": context.get_full_context()
        })
        
        return response["answer"]
    
    def _generate_no_context_response(self, intent: str, user_input: str, user_role: str, 
                                    context: ConversationContext) -> str:
        """Generate response when no relevant context is found"""
        
        original_ref = f" about '{context.original_question}'" if context.original_question else ""
        
        responses = {
            "learning": f"""🤔 I don't have specific information about that topic in my knowledge base{original_ref}. 

As your SAP EWM coach, let me help you find what you need:

**For {user_role} role**, I can guide you with:
• Specific warehouse processes (inbound, outbound, picking)
• Transaction codes and their usage
• Step-by-step procedures

Could you try asking about a specific SAP EWM process or transaction related to your original question{original_ref}?""",

            "navigation": f"""🧭 I want to help guide your next step{original_ref}, but I need more context.

**For {user_role} tasks**, please tell me:
1. Which transaction or screen are you currently in?
2. What was the last step you completed?

This will help me give you the exact next action to take for your situation{original_ref}.""",

            "error": f"""🚨 I'm here to help resolve your issue{original_ref}!

**For {user_role} troubleshooting**, I need:
1. What specific error message are you seeing?
2. Which transaction or process were you performing?
3. What were you trying to accomplish?

Share these details and I'll guide you through the solution step by step for your problem{original_ref}."""
        }
        
        return responses.get(intent, responses["learning"])

# Initialize Skippy components with coaching enhancements
@st.cache_resource
def initialize_skippy():
    """Initialize Skippy's AI components with coaching capabilities"""
    
    # Initialize embeddings - your exact working configuration
    embeddings = AzureOpenAIEmbeddings(
        base_url="https://genaiapimna.jnj.com/openai-embeddings/openai",
        openai_api_key='f89d10a91b9d4cc989085a495d695eb3',
        api_version= "2022-12-01",
        model="text-embedding-ada-002",
        openai_api_type="azure"
    )
    
    # Initialize LLM - your exact working configuration
    os.environ["OPENAI_API_KEY"] ='3c44ff2f28874e6f9d74dd79e89a8093'
    llm = AzureChatOpenAI(
        azure_endpoint= "https://genaiapimna.jnj.com/openai-chat",
        api_key='acd5a7d2b4d64ea6871aeb4cbc3113dd',
        api_version="2023-05-15",
        deployment_name="gpt-4o",
        temperature=0.1,  # Low temperature for consistent coaching
        streaming=True
    )
    
    # Load ChromaDB - your exact working pattern
    if not Path('./data/eWMDB').exists():
        st.error(f"❌ ChromaDB not found at /data/eWMDB")
        st.info("Please run the index builder first to create the knowledge base.")
        st.stop()
    
    vector_db = Chroma(
        persist_directory='./data/eWMDB', 
        embedding_function=embeddings
    )
    
    # Base prompt template (for fallback)
    base_prompt = ChatPromptTemplate.from_template("""
You are Skippy, an SAP EWM Coach providing guided assistance.

Conversation Context:
{conversation_context}

<context>
{context}
</context>

User Question: {input}

Provide focused, coaching-style guidance:""")
    
    # Initialize coaching components
    intent_classifier = IntentClassifier()
    role_detector = RoleDetector()
    context_analyzer = ContextAnalyzer()
    response_generator = CoachingResponseGenerator(llm, base_prompt)
    
    return embeddings, llm, vector_db, intent_classifier, role_detector, context_analyzer, response_generator

# Skippy's enhanced coaching brain with memory
def ask_skippy_coach_smart(question: str, chat_history: List[Dict], components) -> str:
    """Ask Skippy Coach with intelligent conversation memory"""
    
    embeddings, llm, vector_db, intent_classifier, role_detector, context_analyzer, response_generator = components
    
    # Get or create conversation context
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = ConversationContext()
    
    context = st.session_state.conversation_context
    
    try:
        # Step 1: Classify user intent (context-aware)
        intent = intent_classifier.classify_intent(question, context)
        
        # Step 2: Detect user role (context-aware)
        user_role = role_detector.detect_role(question, chat_history, context)
        
        # Step 3: If this is a new conversation, set original question
        if context.conversation_stage == "initial":
            context.set_original_question(question, intent, user_role)
            context.conversation_stage = "processing"
        elif context.conversation_stage == "clarifying":
            # User is providing clarification - extract the info
            context.conversation_stage = "answering"
            # You could parse specific clarification answers here
        
        # Step 4: Search for relevant documents
        search_query = context.original_question if context.original_question else question
        relevant_docs = vector_db.similarity_search(search_query, k=3)
        context.relevant_docs = relevant_docs
        
        # Step 5: Check if clarification is needed
        needs_clarification, clarification_question = context_analyzer.needs_clarification(
            question, relevant_docs, intent, context
        )
        
        # Step 6: Handle clarification or generate response
        if needs_clarification and len(context.clarifications_asked) < 5:
            context.conversation_stage = "clarifying"
            return f"🎯 **Coaching Question:** {clarification_question}\n\n*This helps me give you the most relevant guidance for your original question.*"
        
        # Step 7: Generate coaching response with full context
        context.conversation_stage = "answering"
        response = response_generator.generate_response(
            intent, question, relevant_docs, user_role, context
        )
        
        # Step 8: Add coaching context display
        coaching_context = {
            "learning": "📚 **Learning Mode**",
            "navigation": "🧭 **Process Navigation**", 
            "error": "🚨 **Problem Resolution**"
        }
        
        role_emoji = {
            "execution": "👩‍💼",
            "supervisor": "👨‍💼", 
            "configuration": "⚙️"
        }
        
        # Format final response with context reference
        context_note = f"\n\n*💭 Remembering: {context.original_question}*" if context.original_question != question else ""
        final_response = f"{coaching_context.get(intent, '🤖')} | {role_emoji.get(user_role, '👤')} **{user_role.title()}**\n\n{response}{context_note}"
        
        # Add minimal source info
        if len(relevant_docs) > 1:
            unique_sources = list(set([
                Path(doc.metadata.get('source', 'Unknown')).name 
                for doc in relevant_docs[:2]
            ]))
            if unique_sources:
                final_response += f"\n\n*📄 Referenced: {', '.join(unique_sources)}*"
        
        return final_response
        
    except Exception as e:
        traceback.print_exc()
        return f"""
        🤔 I encountered a technical issue, but let me still try to coach you!
        
        **Remembering your question**: {context.original_question if context.original_question else question}
        
        Could you provide more specific details about:
        • Which SAP EWM process you're working with
        • Your current step or transaction
        • What you're trying to accomplish
        """

# Streamlit UI Configuration (enhanced for memory-aware coaching)
def configure_streamlit():
    """Configure Streamlit page settings for memory-aware coaching interface"""
    st.set_page_config(
        page_title="Skippy - SAP EWM Coach (Smart)",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Enhanced CSS for coaching interface
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    .memory-indicator {
        background-color: #e3f2fd;
        border: 1px solid #1976d2;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #1976d2;
    }
    
    .user-message {
        background-color: #007acc;
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: right;
    }
    
    .skippy-message {
        background-color: #f8f9fa;
        color: #333;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .skippy-header {
        color: #28a745;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .context-reminder {
        background-color: #fff8e1;
        border: 1px solid #ffb74d;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main Skippy Coach application with intelligent memory"""
    
    configure_streamlit()
    
    # Header with memory emphasis
    st.title("🧠 Skippy - Your Smart SAP EWM Coach")
    st.markdown("*I remember our conversation and guide you step-by-step through SAP EWM processes, configurations, and troubleshooting.*")
    
    # Initialize Skippy Coach
    try:
        with st.spinner("🧠 Initializing Skippy's smart coaching capabilities..."):
            components = initialize_skippy()
        
        st.success("✅ Skippy Coach is ready to guide you intelligently!")
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Skippy Coach: {str(e)}")
        st.info("Please make sure the ChromaDB index has been built and is available.")
        st.stop()
    
    # Initialize session state for smart coaching
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": """👋 **Hello! I'm Skippy, your Smart SAP EWM Coach!**

I'm **memory-aware** and will remember our conversation throughout our coaching session:

🧠 **I remember** your original questions during clarification loops  
🎯 **I track** what we've discussed to avoid repeating questions  
🧭 **I guide** you progressively based on our conversation history  
📚 **I learn** your role and adapt my coaching style accordingly  

**What do you need guidance with today?**
• Working through a specific SAP EWM process?
• Stuck on a particular step or transaction?  
• Encountering an error or issue?
• Learning about EWM concepts?

*I'll remember your question and guide you intelligently through clarifications and solutions!*"""
            }
        ]
    
    # Initialize conversation context
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = ConversationContext()
    
    # Display current conversation memory (if active)
    context = st.session_state.conversation_context
    if context.original_question:
        st.markdown(f"""
        <div class="memory-indicator">
        🧠 <strong>Coaching Context:</strong> {context.original_question}
        {f" | 🎯 Clarifications: {len(context.clarifications_asked)}/5" if context.clarifications_asked else ""}
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history with memory context
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="skippy-message">
                    <div class="skippy-header">🧠 Smart Coach Skippy:</div>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input - enhanced for smart coaching
    st.markdown("---")
    
    with st.form("smart_coaching_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Tell me about your SAP EWM situation...",
                placeholder="e.g., I'm stuck at the putaway confirmation screen in /nLT12",
                key="user_question",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("Get Smart Guidance 🧠", use_container_width=True)
    
    # Handle user input with smart coaching logic
    if submit_button and user_input.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show smart coaching indicator
        with st.spinner("🧠 Skippy is thinking with full context awareness..."):
            # Get smart coaching response
            coaching_response = ask_skippy_coach_smart(
                user_input, 
                st.session_state.messages[:-1],  # Exclude current message
                components
            )
        
        # Add coaching response
        st.session_state.messages.append({"role": "assistant", "content": coaching_response})
        
        # Rerun to update display
        st.rerun()
    
    # Enhanced sidebar for smart coaching
    with st.sidebar:
        st.header("🧠 Smart Coaching Features")
        st.markdown("""
        **Memory-Aware Coaching:**
        - Remembers your original question
        - Tracks clarifications already asked
        - Builds progressive understanding
        - Avoids repetitive questions
        
        **Context Intelligence:**
        - Maintains conversation thread
        - References previous exchanges
        - Adapts based on your responses
        - Learns your working style
        """)
        
        # Show current conversation state
        if st.session_state.conversation_context.original_question:
            st.header("🎯 Current Session")
            ctx = st.session_state.conversation_context
            st.markdown(f"""
            **Original Question:** {ctx.original_question[:50]}...
            
            **Intent:** {ctx.detected_intent.title()}
            **Role:** {ctx.detected_role.title()}
            **Clarifications:** {len(ctx.clarifications_asked)}/5
            **Stage:** {ctx.conversation_stage.title()}
            """)
        
        st.header("💡 Smart Coaching Tips")
        st.markdown("""
        **I work best when you:**
        - Answer my clarifying questions directly
        - Provide specific details when asked
        - Let me know if I understood correctly
        - Tell me when you've completed a step
        
        **I remember:**
        - Your original question throughout our chat
        - What I've already asked you
        - Your role and working context
        - The process we're working through
        """)
        
        # Reset coaching session
        if st.button("🔄 Start Fresh Coaching Session", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            st.session_state.conversation_context = ConversationContext()
            st.rerun()
        
        st.markdown("---")
        st.markdown("*Skippy Smart Coach v2.1 - Memory-Aware SAP EWM Guidance*")

if __name__ == "__main__":
    main()
