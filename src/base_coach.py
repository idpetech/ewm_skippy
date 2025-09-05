#!/usr/bin/env python3
"""
Base Coach Architecture for Skippy Multi-Coach System

This module provides the foundational classes and interfaces for creating
specialized coaches with different capabilities and data sources.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class CoachType(Enum):
    """Types of specialized coaches"""
    EWM_COACH = "ewm_coach"
    BUSINESS_ANALYST = "business_analyst"
    SUPPORT_COACH = "support_coach"
    DEV_GURU = "dev_guru"

class DataSourceType(Enum):
    """Types of data sources coaches can handle"""
    PDF = "pdf"
    WORD = "word"
    CONFLUENCE = "confluence"
    SOURCE_CODE = "source_code"
    DATABASE_SCHEMA = "database_schema"

class ConversationStage(Enum):
    """Conversation stages"""
    INITIAL = "initial"
    CLARIFYING = "clarifying"
    ANSWERING = "answering"

# =============================================================================
# BASE DATA MODELS
# =============================================================================

@dataclass
class CoachCapabilities:
    """Defines what a coach can do"""
    data_sources: List[DataSourceType] = field(default_factory=list)
    can_analyze_code: bool = False
    can_read_documents: bool = False
    can_access_confluence: bool = False
    can_query_database: bool = False
    can_generate_code: bool = False
    can_analyze_architecture: bool = False
    can_detect_code_smells: bool = False
    can_recommend_fixes: bool = False

@dataclass
class Conversation:
    """Enhanced conversation state management"""
    original_question: str = ""
    clarifications_asked: List[str] = field(default_factory=list)
    stage: ConversationStage = ConversationStage.INITIAL
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    previous_responses: List[str] = field(default_factory=list)
    coach_context: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str):
        """Add a message to chat history"""
        self.chat_history.append({"role": role, "content": content})
        if role == "assistant":
            self.previous_responses.append(content)
        logging.debug(f"Added {role} message to conversation history")
    
    def get_context_summary(self) -> str:
        """Get comprehensive conversation context for LLM"""
        parts = [f"Original Question: {self.original_question}"]
        
        if self.clarifications_asked:
            parts.append(f"Clarifications Asked: {len(self.clarifications_asked)}")
        
        # Add recent conversation history
        if self.chat_history:
            recent_messages = self.chat_history[-4:]  # Last 4 messages
            conversation_context = []
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Coach"
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                conversation_context.append(f"{role}: {content}")
            parts.append(f"Recent Conversation: {' | '.join(conversation_context)}")
        
        return " | ".join(parts)
    
    def reset(self):
        """Reset for new conversation"""
        logging.info("Resetting conversation context")
        self.original_question = ""
        self.clarifications_asked = []
        self.stage = ConversationStage.INITIAL
        self.chat_history = []
        self.previous_responses = []
        self.coach_context = {}

# =============================================================================
# BASE COACH INTERFACE
# =============================================================================

class BaseCoach(ABC):
    """Abstract base class for all Skippy coaches"""
    
    def __init__(self, coach_type: CoachType, config: Dict[str, Any]):
        self.coach_type = coach_type
        self.config = config
        self.capabilities = self._define_capabilities()
        self.embeddings = self._init_embeddings()
        self.llm = self._init_llm()
        self.vector_db = self._init_vector_db()
        self.prompt_template = self._create_prompt_template()
        logging.info(f"{self.coach_type.value} coach initialized successfully")
    
    @abstractmethod
    def _define_capabilities(self) -> CoachCapabilities:
        """Define what this coach can do"""
        pass
    
    @abstractmethod
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for this coach"""
        pass
    
    @abstractmethod
    def _get_system_instructions(self) -> str:
        """Get system instructions specific to this coach"""
        pass
    
    def _init_embeddings(self) -> AzureOpenAIEmbeddings:
        """Initialize embeddings with configuration"""
        return AzureOpenAIEmbeddings(
            base_url=self.config.get("embedding_endpoint", ""),
            openai_api_key=self.config.get("embedding_api_key", ""),
            api_version=self.config.get("embedding_api_version", "2022-12-01"),
            model=self.config.get("embedding_deployment", "text-embedding-ada-002"),
            openai_api_type="azure"
        )
    
    def _init_llm(self) -> AzureChatOpenAI:
        """Initialize LLM with configuration"""
        os.environ["OPENAI_API_KEY"] = self.config.get("chat_api_key", "")
        return AzureChatOpenAI(
            azure_endpoint=self.config.get("chat_endpoint", ""),
            api_key=self.config.get("chat_api_key", ""),
            api_version=self.config.get("chat_api_version", "2023-05-15"),
            deployment_name=self.config.get("chat_deployment", "gpt-4o"),
            temperature=self.config.get("llm_temperature", 0.1),
            streaming=True
        )
    
    def _init_vector_db(self) -> Optional[Chroma]:
        """Initialize vector database"""
        db_path = self.config.get("chroma_db_path")
        if not db_path or not Path(db_path).exists():
            logging.warning(f"ChromaDB not found at {db_path}")
            return None
        
        return Chroma(
            persist_directory=str(db_path),
            embedding_function=self.embeddings
        )
    
    def ask(self, question: str, conversation: Conversation, chat_history: List[Dict[str, str]] = None) -> str:
        """Main coaching entry point"""
        logging.info(f"Processing question with {self.coach_type.value}: {question[:50]}...")
        
        # Update conversation with new message
        conversation.add_message("user", question)
        
        # Initialize conversation if new
        if conversation.stage == ConversationStage.INITIAL:
            conversation.original_question = question
            conversation.stage = ConversationStage.ANSWERING
        
        # Get relevant documents
        docs = self._retrieve_documents(conversation.original_question or question)
        
        # Generate response
        return self._generate_response(question, docs, conversation, chat_history)
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents"""
        if not self.vector_db:
            return []
        
        try:
            docs = self.vector_db.similarity_search(query, k=self.config.get("doc_retrieval_count", 3))
            logging.debug(f"Retrieved {len(docs)} documents for query")
            return docs
        except Exception as e:
            logging.error(f"Document retrieval failed: {e}")
            return []
    
    def _generate_response(self, question: str, docs: List[Document], 
                         conversation: Conversation, chat_history: List[Dict[str, str]] = None) -> str:
        """Generate coaching response"""
        if not docs:
            return self._format_no_context_response(conversation)
        
        try:
            # Build conversation history for prompt
            conversation_history = ""
            if chat_history and len(chat_history) > 1:
                recent_history = chat_history[-6:]  # Last 6 messages
                history_parts = []
                for msg in recent_history[:-1]:  # Exclude current question
                    role = "User" if msg["role"] == "user" else "Coach"
                    content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
                    history_parts.append(f"{role}: {content}")
                if history_parts:
                    conversation_history = f"Previous Conversation:\n" + "\n".join(history_parts) + "\n"
            
            # Create retrieval chain
            retriever = SkippyRetriever(documents=docs)
            document_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Generate response
            response = retrieval_chain.invoke({
                "input": question,
                "coach_type": self.coach_type.value,
                "system_instructions": self._get_system_instructions(),
                "conversation_context": conversation.get_context_summary(),
                "conversation_history": conversation_history,
                "capabilities": self._format_capabilities()
            })
            
            return response["answer"]
            
        except Exception as e:
            logging.error(f"Response generation failed: {e}")
            return self._format_no_context_response(conversation)
    
    def _format_capabilities(self) -> str:
        """Format capabilities for the prompt"""
        caps = []
        if self.capabilities.can_analyze_code:
            caps.append("Code Analysis")
        if self.capabilities.can_read_documents:
            caps.append("Document Reading")
        if self.capabilities.can_access_confluence:
            caps.append("Confluence Access")
        if self.capabilities.can_query_database:
            caps.append("Database Queries")
        if self.capabilities.can_generate_code:
            caps.append("Code Generation")
        if self.capabilities.can_analyze_architecture:
            caps.append("Architecture Analysis")
        if self.capabilities.can_detect_code_smells:
            caps.append("Code Smell Detection")
        if self.capabilities.can_recommend_fixes:
            caps.append("Fix Recommendations")
        
        return ", ".join(caps) if caps else "Basic Q&A"
    
    def _format_no_context_response(self, conversation: Conversation) -> str:
        """Format fallback response when no relevant context is found"""
        return f"🤔 I don't have specific information about that topic regarding '{conversation.original_question}'.\n\nPlease try rephrasing your question or ask about topics within my expertise."

# =============================================================================
# RETRIEVER COMPONENT
# =============================================================================

class SkippyRetriever(BaseRetriever):
    """Simple document retriever for Skippy coaches"""
    documents: List[Document]
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.documents

# =============================================================================
# COACH FACTORY
# =============================================================================

class CoachFactory:
    """Factory for creating different types of coaches"""
    
    @staticmethod
    def create_coach(coach_type: CoachType, config: Dict[str, Any]) -> BaseCoach:
        """Create a coach of the specified type"""
        if coach_type == CoachType.EWM_COACH:
            from src.ewm_coach import EWMCoach
            return EWMCoach(config)
        elif coach_type == CoachType.BUSINESS_ANALYST:
            from src.business_analyst_coach import BusinessAnalystCoach
            return BusinessAnalystCoach(config)
        elif coach_type == CoachType.SUPPORT_COACH:
            from src.support_coach import SupportCoach
            return SupportCoach(config)
        elif coach_type == CoachType.DEV_GURU:
            from src.dev_guru_coach import DevGuruCoach
            return DevGuruCoach(config)
        else:
            raise ValueError(f"Unknown coach type: {coach_type}")

# =============================================================================
# MIXED COACH CAPABILITIES
# =============================================================================

class MixedCoach(BaseCoach):
    """Coach that combines capabilities from multiple specialized coaches"""
    
    def __init__(self, coach_types: List[CoachType], config: Dict[str, Any]):
        self.coach_types = coach_types
        self.specialized_coaches = {}
        
        # Initialize specialized coaches
        for coach_type in coach_types:
            self.specialized_coaches[coach_type] = CoachFactory.create_coach(coach_type, config)
        
        # Combine capabilities
        combined_capabilities = CoachCapabilities()
        for coach in self.specialized_coaches.values():
            combined_capabilities.data_sources.extend(coach.capabilities.data_sources)
            combined_capabilities.can_analyze_code |= coach.capabilities.can_analyze_code
            combined_capabilities.can_read_documents |= coach.capabilities.can_read_documents
            combined_capabilities.can_access_confluence |= coach.capabilities.can_access_confluence
            combined_capabilities.can_query_database |= coach.capabilities.can_query_database
            combined_capabilities.can_generate_code |= coach.capabilities.can_generate_code
            combined_capabilities.can_analyze_architecture |= coach.capabilities.can_analyze_architecture
            combined_capabilities.can_detect_code_smells |= coach.capabilities.can_detect_code_smells
            combined_capabilities.can_recommend_fixes |= coach.capabilities.can_recommend_fixes
        
        # Remove duplicates
        combined_capabilities.data_sources = list(set(combined_capabilities.data_sources))
        
        super().__init__(CoachType.EWM_COACH, config)  # Use base type
        self.capabilities = combined_capabilities
    
    def _define_capabilities(self) -> CoachCapabilities:
        return self.capabilities
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template("""
You are a specialized Skippy coach with combined capabilities from multiple domains.

System Instructions:
{system_instructions}

Coach Type: {coach_type}
Capabilities: {capabilities}
Conversation Context: {conversation_context}
Current Input: {input}

{conversation_history}

Relevant Documentation:
{context}

Provide focused, expert guidance based on your combined capabilities:
""")
    
    def _get_system_instructions(self) -> str:
        return f"Combined expertise from: {', '.join([ct.value for ct in self.coach_types])}"
    
    def ask(self, question: str, conversation: Conversation, chat_history: List[Dict[str, str]] = None) -> str:
        """Route question to most appropriate specialized coach"""
        # Simple routing logic - can be enhanced with ML
        if any(word in question.lower() for word in ['code', 'function', 'class', 'method', 'bug', 'fix']):
            if CoachType.DEV_GURU in self.specialized_coaches:
                return self.specialized_coaches[CoachType.DEV_GURU].ask(question, conversation, chat_history)
        
        if any(word in question.lower() for word in ['business', 'requirement', 'process', 'workflow']):
            if CoachType.BUSINESS_ANALYST in self.specialized_coaches:
                return self.specialized_coaches[CoachType.BUSINESS_ANALYST].ask(question, conversation, chat_history)
        
        if any(word in question.lower() for word in ['support', 'issue', 'problem', 'help']):
            if CoachType.SUPPORT_COACH in self.specialized_coaches:
                return self.specialized_coaches[CoachType.SUPPORT_COACH].ask(question, conversation, chat_history)
        
        # Default to EWM coach
        if CoachType.EWM_COACH in self.specialized_coaches:
            return self.specialized_coaches[CoachType.EWM_COACH].ask(question, conversation, chat_history)
        
        # Fallback to first available coach
        if self.specialized_coaches:
            first_coach = next(iter(self.specialized_coaches.values()))
            return first_coach.ask(question, conversation, chat_history)
        
        return "I'm not sure how to help with that question. Please try rephrasing."
