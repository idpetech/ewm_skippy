#!/usr/bin/env python3
"""
Skippy - SAP EWM Coach (Refactored)

A coaching-style SAP EWM assistant that provides guided, step-by-step assistance
rather than encyclopedia-style knowledge dumps. Skippy acts as a true coach,
asking clarifying questions and providing focused, actionable guidance.

Features:
- Intent classification (Learning, Navigation, Error Resolution)
- Clarification loops with 5-turn maximum
- Role-aware guidance (Execution, Supervisor, Configuration)
- Coaching personality with step-by-step guidance
- Context-aware responses focused on next actions

Author: AI Development Team
Version: 2.0.0 (Coach Edition)
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

# Intent Classification System
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
    
    def classify_intent(self, user_input: str) -> str:
        """Classify user intent based on input patterns"""
        user_lower = user_input.lower()
        
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

# Role Detection System
class RoleDetector:
    """Detects user role to adapt coaching style"""
    
    def __init__(self):
        self.execution_keywords = ['execute', 'perform', 'do', 'create', 'process', 'scan', 'confirm']
        self.supervisor_keywords = ['monitor', 'check', 'verify', 'review', 'approve', 'status', 'overview']
        self.config_keywords = ['configure', 'setup', 'customize', 'define', 'maintain', 'assign', 'table']
    
    def detect_role(self, user_input: str, chat_history: List[Dict]) -> str:
        """Detect user role from input and chat history"""
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

# Context Analysis System
class ContextAnalyzer:
    """Analyzes retrieved documents to determine if clarification is needed"""
    
    def needs_clarification(self, user_input: str, relevant_docs: List[Document], intent: str) -> Tuple[bool, Optional[str]]:
        """Determine if clarification is needed based on context diversity"""
        
        if not relevant_docs:
            return False, None
        
        # Extract unique process areas from documents
        process_areas = set()
        transaction_codes = set()
        
        for doc in relevant_docs:
            content = doc.page_content.lower()
            metadata = doc.metadata
            
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
            if len(process_areas) > 1:
                areas_list = ', '.join(sorted(process_areas))
                return True, f"I found information about multiple warehouse areas: {areas_list}. Which specific area would you like to focus on?"
        
        # Navigation mode clarifications
        elif intent == "navigation":
            if 'transaction' not in user_input.lower() and 'tcode' not in user_input.lower():
                return True, "To guide you to the next step, I need to know: Which transaction or Fiori app are you currently working in?"
        
        # Error mode clarifications
        elif intent == "error":
            if not any(keyword in user_input.lower() for keyword in ['error', 'message', 'exception', 'code']):
                return True, "To help resolve this issue, could you share the specific error message or code you're seeing?"
        
        return False, None

# Coaching Response Generator
class CoachingResponseGenerator:
    """Generates coaching-style responses based on intent and context"""
    
    def __init__(self, llm, prompt_template):
        self.llm = llm
        self.base_prompt = prompt_template
        
        # Coaching prompts for different intents
        self.coaching_prompts = {
            "learning": ChatPromptTemplate.from_template("""
You are Skippy, an SAP EWM Coach. The user wants to learn about something. Your job is to provide clear, focused guidance.

Coaching Style:
- Give ONE clear explanation or concept at a time
- Use simple, practical language
- Focus on what they need to know for their role: {user_role}
- Ask if they need the next level of detail
- Provide concrete examples when helpful

User Role: {user_role}
User Question: {input}
Context from SAP EWM documentation:

{context}

Provide a focused, coaching-style response that explains the concept clearly without overwhelming detail:
"""),

            "navigation": ChatPromptTemplate.from_template("""
You are Skippy, an SAP EWM Coach. The user needs guidance on their next step in a process.

Coaching Style:
- Focus ONLY on the immediate next step
- Be specific about which screen/transaction/button to use
- Ask for confirmation they completed the step before continuing
- Adapt guidance for their role: {user_role}

User Role: {user_role}
User Input: {input}
Context from SAP EWM documentation:

{context}

Provide the specific next step they should take:
"""),

            "error": ChatPromptTemplate.from_template("""
You are Skippy, an SAP EWM Coach. The user has encountered an issue or error.

Coaching Style:
- Start with immediate diagnostic questions if needed
- Provide ONE troubleshooting step at a time
- Explain what each step will accomplish
- Adapt solutions for their role: {user_role}

User Role: {user_role}
User Problem: {input}
Context from SAP EWM documentation:

{context}

Provide focused troubleshooting guidance:
""")
        }
    
    def generate_response(self, intent: str, user_input: str, relevant_docs: List[Document], user_role: str) -> str:
        """Generate coaching response based on intent"""
        
        if not relevant_docs:
            return self._generate_no_context_response(intent, user_input, user_role)
        
        # Use intent-specific prompt
        prompt = self.coaching_prompts.get(intent, self.base_prompt)
        
        # Create retriever with documents
        retriever = MyRetriever(documents=relevant_docs)
        
        # Create chains
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Generate response
        response = retrieval_chain.invoke({
            "input": user_input,
            "user_role": user_role
        })
        
        return response["answer"]
    
    def _generate_no_context_response(self, intent: str, user_input: str, user_role: str) -> str:
        """Generate response when no relevant context is found"""
        
        responses = {
            "learning": f"""🤔 I don't have specific information about that topic in my knowledge base. 

As your SAP EWM coach, let me help you find what you need:

**For {user_role} role**, I can guide you with:
• Specific warehouse processes (inbound, outbound, picking)
• Transaction codes and their usage
• Step-by-step procedures

Could you try asking about a specific SAP EWM process or transaction?""",

            "navigation": f"""🧭 I want to help guide your next step, but I need more context.

**For {user_role} tasks**, please tell me:
1. Which transaction or screen are you currently in?
2. What was the last step you completed?

This will help me give you the exact next action to take.""",

            "error": f"""🚨 I'm here to help resolve your issue!

**For {user_role} troubleshooting**, I need:
1. What specific error message are you seeing?
2. Which transaction or process were you performing?
3. What were you trying to accomplish?

Share these details and I'll guide you through the solution step by step."""
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

# Skippy's coaching brain - intelligent response generation
def ask_skippy_coach(question: str, chat_history: List[Dict], components) -> str:
    """Ask Skippy Coach a question and get guided assistance"""
    
    embeddings, llm, vector_db, intent_classifier, role_detector, context_analyzer, response_generator = components
    
    try:
        # Step 1: Classify user intent
        intent = intent_classifier.classify_intent(question)
        
        # Step 2: Detect user role
        user_role = role_detector.detect_role(question, chat_history)
        
        # Step 3: Search for relevant documents (fewer docs for focused coaching)
        relevant_docs = vector_db.similarity_search(question, k=3)  # Reduced from 5 to 3
        
        # Step 4: Check if clarification is needed
        needs_clarification, clarification_question = context_analyzer.needs_clarification(
            question, relevant_docs, intent
        )
        
        # Step 5: Handle clarification loop (max 5 turns)
        clarification_count = sum(1 for msg in chat_history[-10:] if "?" in msg.get("content", ""))
        
        if needs_clarification and clarification_count < 5:
            return f"🎯 **Coaching Question:** {clarification_question}\n\n*This helps me give you the most relevant guidance for your situation.*"
        
        # Step 6: Generate coaching response
        response = response_generator.generate_response(intent, question, relevant_docs, user_role)
        
        # Step 7: Add coaching context
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
        
        # Format final response
        final_response = f"{coaching_context.get(intent, '🤖')} | {role_emoji.get(user_role, '👤')} **{user_role.title()}**\n\n{response}"
        
        # Add minimal source info (only if multiple sources)
        if len(relevant_docs) > 1:
            unique_sources = list(set([
                Path(doc.metadata.get('source', 'Unknown')).name 
                for doc in relevant_docs[:2]  # Top 2 sources only
            ]))
            if unique_sources:
                final_response += f"\n\n*📄 Referenced: {', '.join(unique_sources)}*"
        
        return final_response
        
    except Exception as e:
        traceback.print_exc()
        return f"""
        🤔 I encountered a technical issue, but let me still try to coach you!
        
        Could you rephrase your question and include:
        • Which SAP EWM process you're working with
        • Your current step or transaction
        • What you're trying to accomplish
        
        This will help me provide better guidance despite the technical hiccup.
        """

# Streamlit UI Configuration (enhanced for coaching)
def configure_streamlit():
    """Configure Streamlit page settings for coaching interface"""
    st.set_page_config(
        page_title="Skippy - SAP EWM Coach",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Enhanced CSS for coaching interface
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    .coaching-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        background-color: #e8f4fd;
        color: #1f77b4;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
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
    
    .clarification-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main Skippy Coach application"""
    
    configure_streamlit()
    
    # Header with coaching emphasis
    st.title("🎯 Skippy - Your SAP EWM Coach")
    st.markdown("*I'm here to guide you step-by-step through SAP EWM processes, configurations, and troubleshooting.*")
    
    # Initialize Skippy Coach
    try:
        with st.spinner("🧠 Initializing Skippy's coaching capabilities..."):
            components = initialize_skippy()
        
        st.success("✅ Skippy Coach is ready to guide you!")
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Skippy Coach: {str(e)}")
        st.info("Please make sure the ChromaDB index has been built and is available.")
        st.stop()
    
    # Initialize session state for coaching
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": """👋 **Hello! I'm Skippy, your SAP EWM Coach!**

I'm here to **guide** you step-by-step, not just dump information. I'll:

🎯 **Ask clarifying questions** to understand your exact situation  
🧭 **Guide you through processes** one step at a time  
🚨 **Help troubleshoot issues** with focused solutions  
📚 **Explain concepts** clearly for your specific role  

**What do you need guidance with today?**
• Working through a specific SAP EWM process?
• Stuck on a particular step or transaction?  
• Encountering an error or issue?
• Learning about EWM concepts?

*Just describe your situation and I'll coach you through it!*"""
            }
        ]
    
    if "clarification_count" not in st.session_state:
        st.session_state.clarification_count = 0
    
    # Display chat history with coaching context
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
                    <div class="skippy-header">🎯 Coach Skippy:</div>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input - enhanced for coaching
    st.markdown("---")
    
    with st.form("coaching_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Tell me about your SAP EWM situation...",
                placeholder="e.g., I'm stuck at the putaway confirmation screen in /nLT12",
                key="user_question",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("Get Guidance 🎯", use_container_width=True)
    
    # Handle user input with coaching logic
    if submit_button and user_input.strip():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show coaching indicator
        with st.spinner("🎯 Skippy is analyzing your situation..."):
            # Get coaching response
            coaching_response = ask_skippy_coach(
                user_input, 
                st.session_state.messages[:-1],  # Exclude current message
                components
            )
        
        # Add coaching response
        st.session_state.messages.append({"role": "assistant", "content": coaching_response})
        
        # Rerun to update display
        st.rerun()
    
    # Enhanced sidebar for coaching
    with st.sidebar:
        st.header("🎯 Coaching Modes")
        st.markdown("""
        **📚 Learning Mode**
        - "What is putaway strategy?"
        - "Explain inbound process"
        - "How does wave management work?"
        
        **🧭 Navigation Mode**  
        - "I'm stuck in LT12"
        - "What's my next step?"
        - "I completed goods receipt, what now?"
        
        **🚨 Problem Resolution**
        - "Getting error message XYZ"
        - "Can't confirm the task"  
        - "System won't let me proceed"
        """)
        
        st.header("💡 Coaching Tips")
        st.markdown("""
        **Get Better Guidance:**
        - Mention your current transaction
        - Describe what you just completed
        - Share specific error messages
        - Tell me your role (user/supervisor/admin)
        
        **Example Good Questions:**
        - "I'm in /nLT12 and can't confirm putaway"
        - "Completed goods receipt, what's next?"
        - "Error WM123 in picking confirmation"
        """)
        
        # Reset coaching session
        if st.button("🔄 Start Fresh Coaching Session", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            st.session_state.clarification_count = 0
            st.rerun()
        
        # Show clarification counter
        clarification_count = sum(1 for msg in st.session_state.messages if "?" in msg.get("content", ""))
        if clarification_count > 0:
            st.info(f"🎯 Clarification questions asked: {clarification_count}/5")
        
        st.markdown("---")
        st.markdown("*Skippy Coach v2.0 - Step-by-Step SAP EWM Guidance*")

if __name__ == "__main__":
    main()
