#!/usr/bin/env python3
"""
Skippy Business Analyst Coach

Specialized coach for business analysis tasks, focusing on Word documents,
business requirements, process documentation, and workflow analysis.
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
# BUSINESS ANALYST COACH
# =============================================================================

class BusinessAnalystCoach(BaseCoach):
    """Specialized coach for business analysis and requirements"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(CoachType.BUSINESS_ANALYST, config)
    
    def _define_capabilities(self) -> CoachCapabilities:
        """Define business analyst capabilities"""
        return CoachCapabilities(
            data_sources=[
                DataSourceType.WORD,
                DataSourceType.PDF,
                DataSourceType.CONFLUENCE
            ],
            can_read_documents=True,
            can_access_confluence=True,
            can_analyze_architecture=False,
            can_analyze_code=False,
            can_generate_code=False,
            can_detect_code_smells=False,
            can_recommend_fixes=False,
            can_query_database=False
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create business analyst specific prompt template"""
        return ChatPromptTemplate.from_template("""
You are Skippy, an expert Business Analyst Coach specializing in requirements analysis, 
process documentation, and business workflow optimization.

{system_instructions}

Your Expertise:
- Business Requirements Analysis
- Process Documentation Review
- Workflow Optimization
- Stakeholder Communication
- Gap Analysis
- Business Process Modeling

Conversation Context: {conversation_context}
Current Input: {input}

{conversation_history}

Relevant Business Documentation:
{context}

Provide focused business analysis guidance following these principles:
1. Focus on business value and stakeholder needs
2. Ask clarifying questions about business context
3. Provide actionable recommendations
4. Consider process improvements and efficiencies
5. Think about user experience and adoption

Business Analysis Response:
""")
    
    def _get_system_instructions(self) -> str:
        """Get business analyst specific system instructions"""
        return """
As a Business Analyst Coach, you excel at:

📋 **Requirements Analysis**
- Breaking down complex business requirements
- Identifying gaps and inconsistencies
- Ensuring requirements are testable and measurable

🔄 **Process Analysis**
- Mapping current vs. future state processes
- Identifying bottlenecks and inefficiencies
- Recommending process improvements

👥 **Stakeholder Management**
- Understanding different stakeholder perspectives
- Facilitating communication between teams
- Managing expectations and scope

📊 **Documentation Review**
- Analyzing business documents for completeness
- Identifying missing information
- Suggesting documentation improvements

🎯 **Business Value Focus**
- Always consider ROI and business impact
- Prioritize features based on business value
- Think about user adoption and change management

When responding:
- Ask clarifying questions about business context
- Provide specific, actionable recommendations
- Consider the impact on different stakeholder groups
- Suggest measurable success criteria
- Think about implementation challenges and solutions
"""
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """Enhanced document retrieval for business analysis"""
        docs = super()._retrieve_documents(query)
        
        # Business analysis specific filtering
        business_keywords = [
            'requirement', 'process', 'workflow', 'business', 'stakeholder',
            'user story', 'acceptance criteria', 'functional', 'non-functional',
            'use case', 'scenario', 'business rule', 'policy', 'procedure'
        ]
        
        # Filter documents based on business relevance
        relevant_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            if any(keyword in content_lower for keyword in business_keywords):
                relevant_docs.append(doc)
        
        # If no business-relevant docs found, return all docs
        return relevant_docs if relevant_docs else docs
    
    def _analyze_business_context(self, question: str) -> Dict[str, Any]:
        """Analyze the business context of the question"""
        context = {
            'domain': 'general',
            'stakeholder_type': 'unknown',
            'requirement_type': 'unknown',
            'urgency': 'normal'
        }
        
        question_lower = question.lower()
        
        # Detect domain
        if any(word in question_lower for word in ['ewm', 'warehouse', 'inventory', 'logistics']):
            context['domain'] = 'warehouse_management'
        elif any(word in question_lower for word in ['finance', 'accounting', 'billing']):
            context['domain'] = 'finance'
        elif any(word in question_lower for word in ['hr', 'human resources', 'employee']):
            context['domain'] = 'human_resources'
        elif any(word in question_lower for word in ['sales', 'customer', 'marketing']):
            context['domain'] = 'sales_marketing'
        
        # Detect stakeholder type
        if any(word in question_lower for word in ['end user', 'user', 'operator']):
            context['stakeholder_type'] = 'end_user'
        elif any(word in question_lower for word in ['manager', 'supervisor', 'lead']):
            context['stakeholder_type'] = 'management'
        elif any(word in question_lower for word in ['developer', 'technical', 'implementation']):
            context['stakeholder_type'] = 'technical'
        elif any(word in question_lower for word in ['business', 'analyst', 'requirement']):
            context['stakeholder_type'] = 'business_analyst'
        
        # Detect requirement type
        if any(word in question_lower for word in ['functional', 'feature', 'capability']):
            context['requirement_type'] = 'functional'
        elif any(word in question_lower for word in ['non-functional', 'performance', 'security', 'usability']):
            context['requirement_type'] = 'non_functional'
        elif any(word in question_lower for word in ['business rule', 'policy', 'constraint']):
            context['requirement_type'] = 'business_rule'
        
        # Detect urgency
        if any(word in question_lower for word in ['urgent', 'critical', 'asap', 'immediately']):
            context['urgency'] = 'high'
        elif any(word in question_lower for word in ['low priority', 'nice to have', 'future']):
            context['urgency'] = 'low'
        
        return context
    
    def _generate_response(self, question: str, docs: List[Document], 
                         conversation: Conversation, chat_history: List[Dict[str, str]] = None) -> str:
        """Generate business analyst specific response"""
        if not docs:
            return self._format_no_context_response(conversation)
        
        try:
            # Analyze business context
            business_context = self._analyze_business_context(question)
            
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
            
            # Generate response with business context
            response = retrieval_chain.invoke({
                "input": question,
                "coach_type": self.coach_type.value,
                "system_instructions": self._get_system_instructions(),
                "conversation_context": conversation.get_context_summary(),
                "conversation_history": conversation_history,
                "capabilities": self._format_capabilities(),
                "business_context": f"Domain: {business_context['domain']}, Stakeholder: {business_context['stakeholder_type']}, Type: {business_context['requirement_type']}, Urgency: {business_context['urgency']}"
            })
            
            # Enhance response with business analysis insights
            enhanced_response = self._enhance_business_response(response["answer"], business_context, question)
            
            return enhanced_response
            
        except Exception as e:
            logging.error(f"Business analyst response generation failed: {e}")
            return self._format_no_context_response(conversation)
    
    def _enhance_business_response(self, base_response: str, business_context: Dict[str, Any], question: str) -> str:
        """Enhance response with business analysis insights"""
        enhancements = []
        
        # Add domain-specific insights
        if business_context['domain'] != 'general':
            enhancements.append(f"📊 **Domain Context**: {business_context['domain'].replace('_', ' ').title()}")
        
        # Add stakeholder considerations
        if business_context['stakeholder_type'] != 'unknown':
            stakeholder_insights = {
                'end_user': "Consider user experience and ease of use",
                'management': "Focus on business value and ROI",
                'technical': "Include implementation considerations",
                'business_analyst': "Emphasize requirements clarity and traceability"
            }
            if business_context['stakeholder_type'] in stakeholder_insights:
                enhancements.append(f"👥 **Stakeholder Focus**: {stakeholder_insights[business_context['stakeholder_type']]}")
        
        # Add requirement type guidance
        if business_context['requirement_type'] != 'unknown':
            type_guidance = {
                'functional': "Ensure clear acceptance criteria and test scenarios",
                'non_functional': "Define measurable performance and quality metrics",
                'business_rule': "Document business logic and decision criteria"
            }
            if business_context['requirement_type'] in type_guidance:
                enhancements.append(f"📋 **Requirement Type**: {type_guidance[business_context['requirement_type']]}")
        
        # Add urgency considerations
        if business_context['urgency'] == 'high':
            enhancements.append("🚨 **High Priority**: Consider immediate action items and escalation paths")
        elif business_context['urgency'] == 'low':
            enhancements.append("📅 **Low Priority**: Suitable for future planning and optimization")
        
        # Combine base response with enhancements
        if enhancements:
            enhanced_response = base_response + "\n\n" + " | ".join(enhancements)
        else:
            enhanced_response = base_response
        
        return enhanced_response
    
    def _format_no_context_response(self, conversation: Conversation) -> str:
        """Format business analyst specific no-context response"""
        return f"""🤔 I don't have specific business documentation about '{conversation.original_question}'.

**As your Business Analyst Coach, I can help with:**
- 📋 Requirements analysis and documentation
- 🔄 Process mapping and optimization
- 👥 Stakeholder communication strategies
- 📊 Business case development
- 🎯 Gap analysis and solution recommendations

**Please try:**
- Asking about specific business processes or workflows
- Requesting help with requirement documentation
- Seeking guidance on stakeholder management
- Asking for process improvement recommendations

Could you provide more context about your business analysis needs?"""
