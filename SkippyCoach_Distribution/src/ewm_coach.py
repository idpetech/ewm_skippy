#!/usr/bin/env python3
"""
Skippy EWM Coach

Specialized coach for SAP EWM (Extended Warehouse Management) operations,
processes, and troubleshooting. This is the original coach refactored to use
the new base architecture.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .base_coach import BaseCoach, CoachType, CoachCapabilities, DataSourceType, Conversation

# =============================================================================
# EWM COACH
# =============================================================================

class EWMCoach(BaseCoach):
    """Specialized coach for SAP EWM operations and processes"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(CoachType.EWM_COACH, config)
        self.ewm_processes = self._initialize_ewm_processes()
        self.ewm_roles = self._initialize_ewm_roles()
    
    def _define_capabilities(self) -> CoachCapabilities:
        """Define EWM coach capabilities"""
        return CoachCapabilities(
            data_sources=[
                DataSourceType.PDF,  # EWM documentation
                DataSourceType.WORD  # Process documentation
            ],
            can_read_documents=True,
            can_analyze_code=False,
            can_generate_code=False,
            can_detect_code_smells=False,
            can_recommend_fixes=False,
            can_analyze_architecture=False,
            can_query_database=False,
            can_access_confluence=False
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create EWM coach specific prompt template"""
        return ChatPromptTemplate.from_template("""
You are Skippy, an expert SAP EWM Coach specializing in warehouse management,
logistics operations, and process optimization.

{system_instructions}

Your Expertise:
- SAP EWM Process Guidance
- Warehouse Operations
- Logistics Optimization
- Process Troubleshooting
- Configuration Management
- User Training and Support

Conversation Context: {conversation_context}
Current Input: {input}

{conversation_history}

Relevant SAP EWM Documentation:
{context}

Provide focused EWM guidance following these principles:
1. Focus on practical, actionable steps
2. Consider the user's role and experience level
3. Provide clear process guidance
4. Include troubleshooting steps when relevant
5. Think about process optimization opportunities

EWM Coaching Response:
""")
    
    def _get_system_instructions(self) -> str:
        """Get EWM coach specific system instructions"""
        return """
As an SAP EWM Coach, you excel at:

🏭 **Warehouse Operations**
- Inbound and outbound processes
- Putaway and picking strategies
- Inventory management and control
- Quality management processes

📋 **Process Guidance**
- Step-by-step process instructions
- Transaction codes and navigation
- Configuration and customization
- Integration with other SAP modules

🔧 **Troubleshooting**
- Error resolution and problem solving
- Process optimization recommendations
- Performance improvement suggestions
- Best practice implementation

👥 **User Support**
- Role-based guidance (execution, supervisor, configuration)
- Training and knowledge transfer
- Process documentation and standards
- Change management support

When responding:
- Provide specific transaction codes and screen names
- Include step-by-step instructions
- Consider the user's role and experience level
- Suggest process improvements when relevant
- Include troubleshooting steps for common issues
"""
    
    def _initialize_ewm_processes(self) -> Dict[str, List[str]]:
        """Initialize EWM process areas and keywords"""
        return {
            'inbound': [
                'inbound', 'receiving', 'goods receipt', 'putaway', 'confirmation',
                'inbound delivery', 'purchase order', 'vendor', 'supplier'
            ],
            'outbound': [
                'outbound', 'shipping', 'picking', 'packing', 'delivery',
                'sales order', 'customer', 'wave', 'shipment'
            ],
            'inventory': [
                'inventory', 'stock', 'cycle count', 'physical inventory',
                'transfer', 'movement', 'adjustment', 'valuation'
            ],
            'quality': [
                'quality', 'inspection', 'hold', 'release', 'quarantine',
                'quality notification', 'defect', 'rework'
            ],
            'configuration': [
                'configuration', 'setup', 'customize', 'maintain', 'assign',
                'table', 'parameter', 'setting', 'profile'
            ],
            'monitoring': [
                'monitor', 'check', 'verify', 'review', 'approve', 'status',
                'overview', 'report', 'dashboard', 'alert'
            ]
        }
    
    def _initialize_ewm_roles(self) -> Dict[str, List[str]]:
        """Initialize EWM user roles and keywords"""
        return {
            'execution': [
                'execute', 'perform', 'do', 'process', 'scan', 'confirm',
                'complete', 'finish', 'operate', 'run'
            ],
            'supervisor': [
                'monitor', 'check', 'verify', 'review', 'approve', 'status',
                'overview', 'report', 'manage', 'supervise'
            ],
            'configuration': [
                'configure', 'setup', 'customize', 'define', 'maintain',
                'assign', 'create', 'table', 'parameter', 'setting'
            ]
        }
    
    def _analyze_ewm_context(self, question: str) -> Dict[str, Any]:
        """Analyze the EWM context of the question"""
        context = {
            'process_area': 'general',
            'user_role': 'execution',
            'intent_type': 'learning',
            'complexity_level': 'medium',
            'has_error': False,
            'needs_clarification': False
        }
        
        question_lower = question.lower()
        
        # Detect process area
        for process, keywords in self.ewm_processes.items():
            if any(keyword in question_lower for keyword in keywords):
                context['process_area'] = process
                break
        
        # Detect user role
        for role, keywords in self.ewm_roles.items():
            if any(keyword in question_lower for keyword in keywords):
                context['user_role'] = role
                break
        
        # Detect intent type
        if any(word in question_lower for word in ['error', 'issue', 'problem', 'failed', 'stuck', 'cannot']):
            context['intent_type'] = 'error'
            context['has_error'] = True
        elif any(word in question_lower for word in ['next step', 'what now', 'where', 'currently', 'completed']):
            context['intent_type'] = 'navigation'
        elif any(word in question_lower for word in ['what is', 'how does', 'explain', 'tell me', 'describe']):
            context['intent_type'] = 'learning'
        
        # Detect complexity level
        if any(word in question_lower for word in ['simple', 'basic', 'easy']):
            context['complexity_level'] = 'low'
        elif any(word in question_lower for word in ['complex', 'advanced', 'sophisticated']):
            context['complexity_level'] = 'high'
        
        # Check if clarification is needed
        if context['process_area'] == 'general' or context['user_role'] == 'execution':
            context['needs_clarification'] = True
        
        return context
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """Enhanced document retrieval for EWM"""
        docs = super()._retrieve_documents(query)
        
        # EWM specific filtering
        ewm_keywords = [
            'ewm', 'warehouse', 'inbound', 'outbound', 'putaway', 'picking',
            'inventory', 'delivery', 'shipment', 'wave', 'confirmation',
            'process', 'transaction', 'configuration', 'setup'
        ]
        
        # Filter documents based on EWM relevance
        relevant_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            if any(keyword in content_lower for keyword in ewm_keywords):
                relevant_docs.append(doc)
        
        # If no EWM-relevant docs found, return all docs
        return relevant_docs if relevant_docs else docs
    
    def _generate_response(self, question: str, docs: List[Document], 
                         conversation: Conversation, chat_history: List[Dict[str, str]] = None) -> str:
        """Generate EWM coach specific response"""
        if not docs:
            return self._format_no_context_response(conversation)
        
        try:
            # Analyze EWM context
            ewm_context = self._analyze_ewm_context(question)
            
            # Build conversation history for prompt
            conversation_history = ""
            if chat_history and len(chat_history) > 1:
                recent_history = chat_history[-6:]
                history_parts = []
                for msg in recent_history[:-1]:
                    role = "User" if msg["role"] == "user" else "Coach"
                    content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
                    history_parts.append(f"{role}: {content}")
                if history_parts:
                    conversation_history = f"Previous Conversation:\n" + "\n".join(history_parts) + "\n"
            
            # Create retrieval chain
            from .base_coach import SkippyRetriever
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain.chains import create_retrieval_chain
            
            retriever = SkippyRetriever(documents=docs)
            document_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Generate response with EWM context
            response = retrieval_chain.invoke({
                "input": question,
                "coach_type": self.coach_type.value,
                "system_instructions": self._get_system_instructions(),
                "conversation_context": conversation.get_context_summary(),
                "conversation_history": conversation_history,
                "capabilities": self._format_capabilities(),
                "ewm_context": f"Process: {ewm_context['process_area']}, Role: {ewm_context['user_role']}, Intent: {ewm_context['intent_type']}, Complexity: {ewm_context['complexity_level']}"
            })
            
            # Enhance response with EWM insights
            enhanced_response = self._enhance_ewm_response(response["answer"], ewm_context, question)
            
            return enhanced_response
            
        except Exception as e:
            logging.error(f"EWM coach response generation failed: {e}")
            return self._format_no_context_response(conversation)
    
    def _enhance_ewm_response(self, base_response: str, ewm_context: Dict[str, Any], question: str) -> str:
        """Enhance response with EWM insights"""
        enhancements = []
        
        # Add process area insights
        if ewm_context['process_area'] != 'general':
            process_insights = {
                'inbound': "📥 **Inbound Process** - Focus on receiving and putaway operations",
                'outbound': "📤 **Outbound Process** - Focus on picking and shipping operations",
                'inventory': "📦 **Inventory Management** - Focus on stock control and movements",
                'quality': "🔍 **Quality Management** - Focus on inspection and hold processes",
                'configuration': "⚙️ **Configuration** - Focus on system setup and customization",
                'monitoring': "📊 **Monitoring** - Focus on status checks and reporting"
            }
            if ewm_context['process_area'] in process_insights:
                enhancements.append(process_insights[ewm_context['process_area']])
        
        # Add role-specific guidance
        if ewm_context['user_role'] != 'execution':
            role_insights = {
                'supervisor': "👨‍💼 **Supervisor Role** - Consider approval and monitoring aspects",
                'configuration': "⚙️ **Configuration Role** - Focus on system setup and maintenance"
            }
            if ewm_context['user_role'] in role_insights:
                enhancements.append(role_insights[ewm_context['user_role']])
        
        # Add intent-specific guidance
        if ewm_context['intent_type'] == 'error':
            enhancements.append("🚨 **Error Resolution** - Follow systematic troubleshooting steps")
        elif ewm_context['intent_type'] == 'navigation':
            enhancements.append("🧭 **Process Navigation** - Focus on next steps and current location")
        elif ewm_context['intent_type'] == 'learning':
            enhancements.append("📚 **Learning Mode** - Comprehensive explanation provided")
        
        # Add complexity considerations
        if ewm_context['complexity_level'] == 'high':
            enhancements.append("🧠 **Complex Process** - Consider breaking down into smaller steps")
        elif ewm_context['complexity_level'] == 'low':
            enhancements.append("✨ **Simple Process** - Straightforward steps provided")
        
        # Add clarification prompt if needed
        if ewm_context['needs_clarification']:
            enhancements.append("❓ **Clarification** - Please provide more specific details if needed")
        
        # Combine base response with enhancements
        if enhancements:
            enhanced_response = base_response + "\n\n" + " | ".join(enhancements)
        else:
            enhanced_response = base_response
        
        return enhanced_response
    
    def _format_no_context_response(self, conversation: Conversation) -> str:
        """Format EWM coach specific no-context response"""
        return f"""🤔 I don't have specific SAP EWM documentation about '{conversation.original_question}'.

**As your EWM Coach, I can help with:**
- 🏭 **Warehouse Operations** - Inbound, outbound, and inventory processes
- 📋 **Process Guidance** - Step-by-step instructions and navigation
- 🔧 **Troubleshooting** - Error resolution and problem solving
- ⚙️ **Configuration** - System setup and customization
- 👥 **User Support** - Role-based guidance and training

**Please try:**
- Mentioning specific EWM processes (inbound, outbound, putaway, picking)
- Including transaction codes or screen names
- Describing the specific error message or issue
- Specifying your role (execution, supervisor, configuration)

**Common EWM Topics:**
- Inbound delivery processing and putaway
- Outbound picking and shipping
- Inventory management and cycle counting
- Quality management and holds
- System configuration and customization

Could you provide more details about your EWM question?"""
