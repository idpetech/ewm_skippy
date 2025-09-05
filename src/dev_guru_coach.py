#!/usr/bin/env python3
"""
Skippy Dev Guru Coach

Specialized coach for software development, source code analysis, architecture review,
and technical guidance. Handles large codebases with 7000+ classes and 400+ tables.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import re
import ast
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from .base_coach import BaseCoach, CoachType, CoachCapabilities, DataSourceType, Conversation

# =============================================================================
# DEV GURU COACH
# =============================================================================

class DevGuruCoach(BaseCoach):
    """Specialized coach for software development and code analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(CoachType.DEV_GURU, config)
        self.code_patterns = self._initialize_code_patterns()
        self.architecture_components = self._initialize_architecture_components()
    
    def _define_capabilities(self) -> CoachCapabilities:
        """Define dev guru capabilities"""
        return CoachCapabilities(
            data_sources=[
                DataSourceType.SOURCE_CODE,
                DataSourceType.DATABASE_SCHEMA,
                DataSourceType.PDF,  # For technical documentation
                DataSourceType.WORD  # For technical specs
            ],
            can_read_documents=True,
            can_analyze_code=True,
            can_generate_code=True,
            can_detect_code_smells=True,
            can_recommend_fixes=True,
            can_analyze_architecture=True,
            can_query_database=True,
            can_access_confluence=False
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create dev guru specific prompt template"""
        return ChatPromptTemplate.from_template("""
You are Skippy, an expert Dev Guru Coach specializing in software development,
code analysis, architecture review, and technical guidance.

{system_instructions}

Your Expertise:
- Source Code Analysis and Review
- Architecture Design and Patterns
- Code Quality and Best Practices
- Bug Detection and Fixes
- Performance Optimization
- Database Design and Queries
- Technical Documentation

Conversation Context: {conversation_context}
Current Input: {input}

{conversation_history}

Relevant Code and Documentation:
{context}

Provide focused development guidance following these principles:
1. Analyze code structure and patterns
2. Identify potential issues and improvements
3. Suggest best practices and optimizations
4. Provide specific code examples when helpful
5. Consider scalability and maintainability
6. Think about testing and quality assurance

Development Response:
""")
    
    def _get_system_instructions(self) -> str:
        """Get dev guru specific system instructions"""
        return """
As a Dev Guru Coach, you excel at:

💻 **Code Analysis**
- Reviewing code structure and architecture
- Identifying design patterns and anti-patterns
- Analyzing code complexity and maintainability
- Detecting potential bugs and vulnerabilities

🏗️ **Architecture Guidance**
- System design and component relationships
- Database schema design and optimization
- API design and integration patterns
- Scalability and performance considerations

🔧 **Code Quality**
- Best practices and coding standards
- Refactoring recommendations
- Performance optimization techniques
- Testing strategies and quality assurance

🐛 **Problem Solving**
- Bug diagnosis and root cause analysis
- Debugging techniques and tools
- Error handling and logging strategies
- Troubleshooting complex issues

When responding:
- Provide specific code examples and snippets
- Explain the reasoning behind recommendations
- Consider different programming paradigms
- Think about long-term maintainability
- Include testing and validation approaches
- Suggest tools and frameworks when relevant
"""
    
    def _initialize_code_patterns(self) -> Dict[str, List[str]]:
        """Initialize code patterns and anti-patterns"""
        return {
            'design_patterns': [
                'singleton', 'factory', 'observer', 'strategy', 'decorator',
                'adapter', 'facade', 'proxy', 'command', 'state'
            ],
            'anti_patterns': [
                'god class', 'spaghetti code', 'copy paste', 'dead code',
                'magic numbers', 'long parameter list', 'feature envy',
                'shotgun surgery', 'divergent change', 'primitive obsession'
            ],
            'code_smells': [
                'duplicate code', 'long method', 'large class', 'long parameter list',
                'data clumps', 'switch statements', 'temporary field',
                'refused bequest', 'inappropriate intimacy', 'alternative classes'
            ],
            'performance_issues': [
                'n+1 query', 'memory leak', 'inefficient loop', 'unnecessary object creation',
                'string concatenation', 'database connection leak', 'blocking call',
                'synchronous processing', 'large object graph', 'inefficient algorithm'
            ]
        }
    
    def _initialize_architecture_components(self) -> Dict[str, List[str]]:
        """Initialize architecture component keywords"""
        return {
            'layers': [
                'presentation', 'business', 'data access', 'service', 'repository',
                'controller', 'model', 'view', 'middleware', 'gateway'
            ],
            'patterns': [
                'mvc', 'mvp', 'mvvm', 'microservices', 'monolith', 'soa',
                'event driven', 'cqs', 'cqrs', 'ddd', 'hexagonal'
            ],
            'technologies': [
                'rest', 'graphql', 'grpc', 'message queue', 'cache', 'database',
                'api', 'webservice', 'microservice', 'container', 'kubernetes'
            ]
        }
    
    def _analyze_code_context(self, question: str) -> Dict[str, Any]:
        """Analyze the code context of the question"""
        context = {
            'code_type': 'general',
            'analysis_type': 'general',
            'complexity_level': 'medium',
            'focus_area': 'general',
            'has_code_snippet': False,
            'patterns_detected': [],
            'issues_detected': []
        }
        
        question_lower = question.lower()
        
        # Detect code type
        if any(word in question_lower for word in ['class', 'method', 'function', 'object']):
            context['code_type'] = 'object_oriented'
        elif any(word in question_lower for word in ['database', 'table', 'query', 'sql']):
            context['code_type'] = 'database'
        elif any(word in question_lower for word in ['api', 'endpoint', 'service', 'rest']):
            context['code_type'] = 'api'
        elif any(word in question_lower for word in ['frontend', 'ui', 'component', 'react', 'angular']):
            context['code_type'] = 'frontend'
        elif any(word in question_lower for word in ['backend', 'server', 'microservice']):
            context['code_type'] = 'backend'
        
        # Detect analysis type
        if any(word in question_lower for word in ['review', 'analyze', 'examine']):
            context['analysis_type'] = 'review'
        elif any(word in question_lower for word in ['bug', 'error', 'issue', 'problem']):
            context['analysis_type'] = 'debugging'
        elif any(word in question_lower for word in ['optimize', 'performance', 'improve']):
            context['analysis_type'] = 'optimization'
        elif any(word in question_lower for word in ['refactor', 'restructure', 'clean']):
            context['analysis_type'] = 'refactoring'
        elif any(word in question_lower for word in ['design', 'architecture', 'structure']):
            context['analysis_type'] = 'design'
        
        # Detect complexity level
        if any(word in question_lower for word in ['simple', 'basic', 'easy']):
            context['complexity_level'] = 'low'
        elif any(word in question_lower for word in ['complex', 'advanced', 'sophisticated']):
            context['complexity_level'] = 'high'
        
        # Detect focus area
        if any(word in question_lower for word in ['security', 'vulnerability', 'safe']):
            context['focus_area'] = 'security'
        elif any(word in question_lower for word in ['test', 'testing', 'unit test']):
            context['focus_area'] = 'testing'
        elif any(word in question_lower for word in ['deploy', 'deployment', 'ci/cd']):
            context['focus_area'] = 'deployment'
        elif any(word in question_lower for word in ['maintain', 'maintainability', 'legacy']):
            context['focus_area'] = 'maintainability'
        
        # Check for code snippets
        if any(char in question for char in ['{', '}', '(', ')', ';', '=', 'def', 'class', 'function']):
            context['has_code_snippet'] = True
        
        # Detect patterns
        for pattern_type, patterns in self.code_patterns.items():
            for pattern in patterns:
                if pattern in question_lower:
                    context['patterns_detected'].append(f"{pattern_type}: {pattern}")
        
        return context
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """Enhanced document retrieval for code analysis"""
        docs = super()._retrieve_documents(query)
        
        # Code analysis specific filtering
        code_keywords = [
            'class', 'method', 'function', 'interface', 'implementation',
            'database', 'table', 'query', 'api', 'service', 'component',
            'architecture', 'pattern', 'design', 'code', 'bug', 'fix',
            'performance', 'optimization', 'refactor', 'test'
        ]
        
        # Filter documents based on code relevance
        relevant_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            if any(keyword in content_lower for keyword in code_keywords):
                relevant_docs.append(doc)
        
        # If no code-relevant docs found, return all docs
        return relevant_docs if relevant_docs else docs
    
    def _generate_response(self, question: str, docs: List[Document], 
                         conversation: Conversation, chat_history: List[Dict[str, str]] = None) -> str:
        """Generate dev guru specific response"""
        if not docs:
            return self._format_no_context_response(conversation)
        
        try:
            # Analyze code context
            code_context = self._analyze_code_context(question)
            
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
            
            # Generate response with code context
            response = retrieval_chain.invoke({
                "input": question,
                "coach_type": self.coach_type.value,
                "system_instructions": self._get_system_instructions(),
                "conversation_context": conversation.get_context_summary(),
                "conversation_history": conversation_history,
                "capabilities": self._format_capabilities(),
                "code_context": f"Type: {code_context['code_type']}, Analysis: {code_context['analysis_type']}, Complexity: {code_context['complexity_level']}, Focus: {code_context['focus_area']}"
            })
            
            # Enhance response with code insights
            enhanced_response = self._enhance_code_response(response["answer"], code_context, question)
            
            return enhanced_response
            
        except Exception as e:
            logging.error(f"Dev guru response generation failed: {e}")
            return self._format_no_context_response(conversation)
    
    def _enhance_code_response(self, base_response: str, code_context: Dict[str, Any], question: str) -> str:
        """Enhance response with code analysis insights"""
        enhancements = []
        
        # Add code type insights
        if code_context['code_type'] != 'general':
            type_insights = {
                'object_oriented': "🏗️ **OOP Focus** - Consider inheritance, polymorphism, and encapsulation",
                'database': "🗄️ **Database Design** - Review schema, indexes, and query optimization",
                'api': "🌐 **API Design** - Consider REST principles, versioning, and documentation",
                'frontend': "🎨 **Frontend Architecture** - Think about component design and state management",
                'backend': "⚙️ **Backend Services** - Consider scalability, security, and performance"
            }
            if code_context['code_type'] in type_insights:
                enhancements.append(type_insights[code_context['code_type']])
        
        # Add analysis type guidance
        if code_context['analysis_type'] != 'general':
            analysis_insights = {
                'review': "🔍 **Code Review** - Check for best practices and potential issues",
                'debugging': "🐛 **Debugging** - Focus on error handling and logging",
                'optimization': "⚡ **Performance** - Consider algorithms and resource usage",
                'refactoring': "🔧 **Refactoring** - Improve code structure and maintainability",
                'design': "📐 **Architecture** - Consider design patterns and system structure"
            }
            if code_context['analysis_type'] in analysis_insights:
                enhancements.append(analysis_insights[code_context['analysis_type']])
        
        # Add complexity considerations
        if code_context['complexity_level'] == 'high':
            enhancements.append("🧠 **Complex System** - Consider breaking down into smaller components")
        elif code_context['complexity_level'] == 'low':
            enhancements.append("✨ **Simple Solution** - Keep it straightforward and maintainable")
        
        # Add focus area recommendations
        if code_context['focus_area'] != 'general':
            focus_insights = {
                'security': "🔒 **Security** - Review authentication, authorization, and data protection",
                'testing': "🧪 **Testing** - Ensure comprehensive test coverage",
                'deployment': "🚀 **Deployment** - Consider CI/CD and environment management",
                'maintainability': "🔧 **Maintainability** - Focus on code clarity and documentation"
            }
            if code_context['focus_area'] in focus_insights:
                enhancements.append(focus_insights[code_context['focus_area']])
        
        # Add pattern detection insights
        if code_context['patterns_detected']:
            enhancements.append(f"🎯 **Patterns Detected**: {', '.join(code_context['patterns_detected'])}")
        
        # Combine base response with enhancements
        if enhancements:
            enhanced_response = base_response + "\n\n" + " | ".join(enhancements)
        else:
            enhanced_response = base_response
        
        return enhanced_response
    
    def _format_no_context_response(self, conversation: Conversation) -> str:
        """Format dev guru specific no-context response"""
        return f"""🤔 I don't have specific code or technical documentation about '{conversation.original_question}'.

**As your Dev Guru Coach, I can help with:**
- 💻 **Code Analysis** - Review code structure, patterns, and quality
- 🏗️ **Architecture** - Design patterns, system structure, and scalability
- 🐛 **Debugging** - Bug detection, root cause analysis, and fixes
- ⚡ **Performance** - Optimization techniques and best practices
- 🔧 **Refactoring** - Code improvement and maintainability
- 🧪 **Testing** - Test strategies and quality assurance
- 🗄️ **Database** - Schema design, queries, and optimization

**Please try:**
- Sharing specific code snippets or error messages
- Describing the architecture or system you're working with
- Mentioning the programming language or framework
- Providing context about the problem or requirement

**For code analysis:**
- Include relevant code snippets
- Describe the expected vs. actual behavior
- Mention any error messages or logs
- Specify the technology stack and environment

Could you provide more technical details about your development question?"""
