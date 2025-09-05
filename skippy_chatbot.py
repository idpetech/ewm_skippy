#!/usr/bin/env python3
"""
Skippy - SAP EWM Expert Chatbot

A knowledgeable and helpful SAP EWM assistant built on your working ChromaDB index.
Skippy provides accurate, context-aware answers based on your SAP EWM documentation.

Features:
- Uses your proven Azure OpenAI setup
- ChromaDB vector search for relevant context
- Streaming chat interface with auto-scroll
- Persistent chat history
- SAP EWM expertise and helpful personality

Author: AI Development Team
Version: 1.0.0
License: MIT
"""

import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import time

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

# Initialize Skippy components
@st.cache_resource
def initialize_skippy():
    """Initialize Skippy's AI components"""
    
    # Initialize embeddings - your exact working configuration
    embeddings = AzureOpenAIEmbeddings(
        base_url=SkippyConfig.EMBEDDING_ENDPOINT,
        openai_api_key=SkippyConfig.EMBEDDING_API_KEY,
        api_version=SkippyConfig.EMBEDDING_API_VERSION,
        model=SkippyConfig.EMBEDDING_DEPLOYMENT,
        openai_api_type="azure"
    )
    
    # Initialize LLM - your exact working configuration
    os.environ["OPENAI_API_KEY"] = SkippyConfig.CHAT_API_KEY
    llm = AzureChatOpenAI(
        azure_endpoint=SkippyConfig.CHAT_ENDPOINT,
        api_key=SkippyConfig.CHAT_API_KEY,
        api_version=SkippyConfig.CHAT_API_VERSION,
        deployment_name=SkippyConfig.CHAT_DEPLOYMENT,
        temperature=0.1,  # Low temperature for more accurate, consistent responses
        streaming=True   # Enable streaming for better UX
    )
    
    # Load ChromaDB - your exact working pattern
    if not Path(SkippyConfig.DB_PATH).exists():
        st.error(f"❌ ChromaDB not found at {SkippyConfig.DB_PATH}")
        st.info("Please run the index builder first to create the knowledge base.")
        st.stop()
    
    vector_db = Chroma(
        persist_directory=SkippyConfig.DB_PATH, 
        embedding_function=embeddings
    )
    
    # Skippy's enhanced prompt with personality
    prompt = ChatPromptTemplate.from_template("""
You are Skippy, a knowledgeable and helpful SAP EWM (Extended Warehouse Management) expert assistant. 
You have extensive knowledge about SAP EWM processes, configurations, transactions, and best practices.

Your personality:
- Friendly, approachable, and professional
- Always eager to help with SAP EWM questions
- Provide clear, actionable answers
- When you're not sure, you acknowledge limitations
- You explain complex concepts in an understandable way
- You often provide practical examples and best practices

Answer the user's question based on the provided context from SAP EWM documentation.
If the context doesn't contain enough information to fully answer the question, say so honestly and provide what relevant information you can.

<context>
{context}
</context>

User Question: {input}

Skippy's Response:""")
    
    return embeddings, llm, vector_db, prompt

# Skippy's brain - retrieval and response generation
def ask_skippy(question: str, embeddings, llm, vector_db, prompt_template) -> str:
    """Ask Skippy a question and get a knowledgeable response"""
    
    try:
        # Search for relevant documents
        relevant_docs = vector_db.similarity_search(question, k=5)
        
        if not relevant_docs:
            return """
            🤔 Hmm, I couldn't find specific information about that in my SAP EWM knowledge base. 
            
            Could you try rephrasing your question or asking about a more specific SAP EWM topic? 
            I'm particularly good with:
            - Warehouse processes (inbound, outbound, picking, packing)
            - Transaction codes and configurations  
            - Integration with SAP ECC/S4
            - Troubleshooting common issues
            - Best practices and implementation tips
            """
        
        # Create retriever with found documents
        retriever = MyRetriever(documents=relevant_docs)
        
        # Create the chains - using your exact working pattern
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Get Skippy's response
        response = retrieval_chain.invoke({"input": question})
        
        # Add source information
        sources = []
        for doc in relevant_docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            section = doc.metadata.get('section_title', 'Unknown')
            sources.append(f"📄 {Path(source).name} (Page {page})")
        
        # Format response with sources
        answer = response["answer"]
        if sources:
            unique_sources = list(set(sources[:3]))  # Top 3 unique sources
            source_text = "\n\n---\n**📚 Sources:**\n" + "\n".join(unique_sources)
            answer += source_text
        
        return answer
        
    except Exception as e:
        return f"""
        😅 Oops! I encountered a technical issue: {str(e)}
        
        Let me try to help anyway - could you rephrase your question or ask about a specific SAP EWM topic?
        """

# Streamlit UI Configuration
def configure_streamlit():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Skippy - SAP EWM Expert",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better chat experience
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stChatMessage {
        margin-bottom: 1rem;
    }
    
    .chat-container {
        height: 70vh;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 1rem;
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
        background-color: #f0f0f0;
        color: #333;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    .skippy-header {
        color: #4CAF50;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main Skippy application"""
    
    configure_streamlit()
    
    # Header
    st.title("🤖 Skippy - Your SAP EWM Expert")
    st.markdown("*Hi there! I'm Skippy, your knowledgeable SAP EWM assistant. Ask me anything about Extended Warehouse Management!*")
    
    # Initialize Skippy
    try:
        with st.spinner("🧠 Initializing Skippy's knowledge base..."):
            embeddings, llm, vector_db, prompt_template = initialize_skippy()
        
        st.success("✅ Skippy is ready to help!")
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Skippy: {str(e)}")
        st.info("Please make sure the ChromaDB index has been built and is available.")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": """👋 **Hello! I'm Skippy, your SAP EWM expert assistant!**

I'm here to help you with all things related to SAP Extended Warehouse Management. Feel free to ask me about:

• 📦 **Warehouse Processes**: Inbound, outbound, picking, packing, putaway
• ⚙️ **Configuration**: Customizing, setup, and parameters  
• 🔧 **Transactions**: T-codes and their usage
• 🔗 **Integration**: With SAP ECC, S/4HANA, and other modules
• 🚨 **Troubleshooting**: Error resolution and best practices
• 💡 **Best Practices**: Implementation tips and recommendations

What would you like to know about SAP EWM today?"""
            }
        ]
    
    # Display chat history
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
                    <div class="skippy-header">🤖 Skippy:</div>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input - always at bottom and in focus
    st.markdown("---")
    
    # Use form to ensure Enter key works
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask Skippy anything about SAP EWM...",
                placeholder="e.g., What are the key steps in the inbound process?",
                key="user_question",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("Ask Skippy 🚀", use_container_width=True)
    
    # Handle user input
    if submit_button and user_input.strip():
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show thinking indicator
        with st.spinner("🤔 Skippy is thinking..."):
            # Get Skippy's response
            skippy_response = ask_skippy(user_input, embeddings, llm, vector_db, prompt_template)
        
        # Add Skippy's response to chat
        st.session_state.messages.append({"role": "assistant", "content": skippy_response})
        
        # Rerun to update the display and clear input
        st.rerun()
    
    # Sidebar with helpful information
    with st.sidebar:
        st.header("🎯 Skippy's Specialties")
        st.markdown("""
        **SAP EWM Processes:**
        - Goods Receipt
        - Putaway Strategies  
        - Picking & Packing
        - Shipping & Loading
        - Inventory Management
        
        **Configuration Areas:**
        - Warehouse Structure
        - Master Data Setup
        - Process Types
        - Storage Types & Bins
        - Work Centers
        
        **Integration Topics:**
        - EWM-ERP Integration
        - RF/Mobile Devices
        - Print Control
        - Interfaces & APIs
        """)
        
        st.header("💡 Quick Tips")
        st.markdown("""
        - Be specific in your questions
        - Mention transaction codes if you know them
        - Ask about error messages for troubleshooting
        - Include context about your SAP version if relevant
        """)
        
        # Clear chat button
        if st.button("🔄 Start New Conversation", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
            st.rerun()
        
        st.markdown("---")
        st.markdown("*Skippy v1.0 - SAP EWM Expert Assistant*")

if __name__ == "__main__":
    main()
