#!/usr/bin/env python3
"""
Skippy Support Coach

Specialized coach for technical support, troubleshooting, and user assistance.
Handles Word documents, Confluence pages, PDFs, and provides comprehensive support guidance.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .base_coach import BaseCoach, CoachType, CoachCapabilities, DataSourceType, Conversation

# =============================================================================
# SUPPORT COACH
# =============================================================================

class SupportCoach(BaseCoach):
    """Specialized coach for technical support and troubleshooting"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(CoachType.SUPPORT_COACH, config)
        self.support_categories = self._initialize_support_categories()
    
    def _define_capabilities(self) -> CoachCapabilities:
        """Define support coach capabilities"""
        return CoachCapabilities(
            data_sources=[
                DataSourceType.WORD,
                DataSourceType.PDF,
                DataSourceType.CONFLUENCE
            ],
            can_read_documents=True,
            can_access_confluence=True,
            can_analyze_code=False,
            can_generate_code=False,
            can_detect_code_smells=False,
            can_recommend_fixes=False,
            can_query_database=False,
            can_analyze_architecture=False
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create support coach specific prompt template"""
        return ChatPromptTemplate.from_template("""
You are Skippy, an expert Technical Support Coach specializing in troubleshooting,
user assistance, and problem resolution.

{system_instructions}

Your Expertise:
- Technical Troubleshooting
- User Support and Guidance
- Problem Resolution
- Documentation Analysis
- Step-by-step Instructions
- Error Diagnosis

Conversation Context: {conversation_context}
Current Input: {input}

{conversation_history}

Relevant Support Documentation:
{context}

Provide focused support guidance following these principles:
1. Start with the most likely solution
2. Provide clear, step-by-step instructions
3. Ask for confirmation before proceeding to next steps
4. Consider different user skill levels
5. Provide alternative solutions when possible
6. Include preventive measures

Support Response:
""")
    
    def _get_system_instructions(self) -> str:
        """Get support coach specific system instructions"""
        return """
As a Technical Support Coach, you excel at:

🔧 **Troubleshooting**
- Systematic problem diagnosis
- Root cause analysis
- Step-by-step resolution guidance
- Error message interpretation

📚 **Documentation Analysis**
- Finding relevant support articles
- Extracting key information from manuals
- Identifying solution patterns
- Cross-referencing multiple sources

👥 **User Support**
- Adapting explanations to user skill level
- Providing clear, actionable instructions
- Managing user expectations
- Escalating when necessary

🎯 **Problem Resolution**
- Prioritizing solutions by likelihood
- Testing and validation steps
- Preventive measures
- Follow-up recommendations

When responding:
- Start with the most common solution
- Provide one step at a time
- Ask for confirmation before continuing
- Include screenshots or specific locations when helpful
- Suggest alternative approaches
- Consider the user's technical background
"""
    
    def _initialize_support_categories(self) -> Dict[str, List[str]]:
        """Initialize support categories and keywords"""
        return {
            'installation': [
                'install', 'setup', 'configuration', 'deployment', 'environment',
                'prerequisites', 'requirements', 'compatibility'
            ],
            'authentication': [
                'login', 'password', 'authentication', 'authorization', 'access',
                'credentials', 'permissions', 'security'
            ],
            'performance': [
                'slow', 'performance', 'timeout', 'memory', 'cpu', 'optimization',
                'bottleneck', 'lag', 'response time'
            ],
            'data_issues': [
                'data', 'database', 'corruption', 'missing', 'sync', 'backup',
                'restore', 'migration', 'import', 'export'
            ],
            'ui_issues': [
                'interface', 'display', 'layout', 'button', 'menu', 'navigation',
                'screen', 'window', 'dialog'
            ],
            'integration': [
                'api', 'integration', 'connection', 'webhook', 'endpoint',
                'service', 'external', 'third-party'
            ],
            'error_handling': [
                'error', 'exception', 'crash', 'failure', 'bug', 'issue',
                'problem', 'not working', 'broken'
            ]
        }
    
    def _categorize_support_request(self, question: str) -> Dict[str, Any]:
        """Categorize the support request"""
        question_lower = question.lower()
        
        # Find matching categories
        matched_categories = []
        for category, keywords in self.support_categories.items():
            if any(keyword in question_lower for keyword in keywords):
                matched_categories.append(category)
        
        # Determine urgency
        urgency = 'normal'
        if any(word in question_lower for word in ['urgent', 'critical', 'emergency', 'down', 'broken']):
            urgency = 'high'
        elif any(word in question_lower for word in ['minor', 'cosmetic', 'enhancement']):
            urgency = 'low'
        
        # Determine user skill level
        skill_level = 'intermediate'
        if any(word in question_lower for word in ['beginner', 'new', 'first time', 'how to']):
            skill_level = 'beginner'
        elif any(word in question_lower for word in ['advanced', 'expert', 'developer', 'technical']):
            skill_level = 'advanced'
        
        return {
            'categories': matched_categories,
            'urgency': urgency,
            'skill_level': skill_level,
            'has_error_message': any(word in question_lower for word in ['error', 'exception', 'failed', 'cannot'])
        }
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """Enhanced document retrieval for support"""
        docs = super()._retrieve_documents(query)
        
        # Support specific filtering
        support_keywords = [
            'troubleshoot', 'error', 'issue', 'problem', 'solution', 'fix',
            'help', 'support', 'guide', 'manual', 'faq', 'known issue',
            'resolution', 'workaround', 'step', 'instruction'
        ]
        
        # Filter documents based on support relevance
        relevant_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            if any(keyword in content_lower for keyword in support_keywords):
                relevant_docs.append(doc)
        
        # If no support-relevant docs found, return all docs
        return relevant_docs if relevant_docs else docs
    
    def _generate_response(self, question: str, docs: List[Document], 
                         conversation: Conversation, chat_history: List[Dict[str, str]] = None) -> str:
        """Generate support coach specific response"""
        if not docs:
            return self._format_no_context_response(conversation)
        
        try:
            # Categorize support request
            support_context = self._categorize_support_request(question)
            
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
            
            # Generate response with support context
            response = retrieval_chain.invoke({
                "input": question,
                "coach_type": self.coach_type.value,
                "system_instructions": self._get_system_instructions(),
                "conversation_context": conversation.get_context_summary(),
                "conversation_history": conversation_history,
                "capabilities": self._format_capabilities(),
                "support_context": f"Categories: {', '.join(support_context['categories'])}, Urgency: {support_context['urgency']}, Skill Level: {support_context['skill_level']}"
            })
            
            # Enhance response with support insights
            enhanced_response = self._enhance_support_response(response["answer"], support_context, question)
            
            return enhanced_response
            
        except Exception as e:
            logging.error(f"Support coach response generation failed: {e}")
            return self._format_no_context_response(conversation)
    
    def _enhance_support_response(self, base_response: str, support_context: Dict[str, Any], question: str) -> str:
        """Enhance response with support insights"""
        enhancements = []
        
        # Add urgency indicators
        if support_context['urgency'] == 'high':
            enhancements.append("🚨 **High Priority Issue**")
        elif support_context['urgency'] == 'low':
            enhancements.append("📅 **Low Priority**")
        
        # Add skill level considerations
        if support_context['skill_level'] == 'beginner':
            enhancements.append("👶 **Beginner Friendly** - Detailed steps provided")
        elif support_context['skill_level'] == 'advanced':
            enhancements.append("⚡ **Advanced User** - Technical details included")
        
        # Add category-specific guidance
        if 'error_handling' in support_context['categories']:
            enhancements.append("🔍 **Error Analysis** - Check logs and error messages")
        if 'performance' in support_context['categories']:
            enhancements.append("⚡ **Performance Issue** - Monitor system resources")
        if 'authentication' in support_context['categories']:
            enhancements.append("🔐 **Authentication** - Verify credentials and permissions")
        
        # Add escalation guidance for high urgency
        if support_context['urgency'] == 'high':
            enhancements.append("📞 **Escalation** - Contact support team if issue persists")
        
        # Combine base response with enhancements
        if enhancements:
            enhanced_response = base_response + "\n\n" + " | ".join(enhancements)
        else:
            enhanced_response = base_response
        
        return enhanced_response
    
    def _format_no_context_response(self, conversation: Conversation) -> str:
        """Format support coach specific no-context response"""
        return f"""🤔 I don't have specific support documentation about '{conversation.original_question}'.

**As your Support Coach, I can help with:**
- 🔧 Technical troubleshooting and problem resolution
- 📚 Finding relevant documentation and guides
- 👥 User support and step-by-step instructions
- 🎯 Error diagnosis and solution recommendations
- 📞 Escalation guidance when needed

**Please try:**
- Describing the specific error message or issue
- Mentioning which system or application is involved
- Providing details about what you were trying to do
- Sharing any error codes or symptoms

**For immediate assistance:**
- Check system logs for error messages
- Verify system requirements and compatibility
- Try basic troubleshooting steps (restart, clear cache, etc.)

Could you provide more details about the issue you're experiencing?"""
