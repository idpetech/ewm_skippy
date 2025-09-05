# 🚀 Skippy Multi-Coach System

**A comprehensive AI coaching system with specialized expertise across multiple domains.**

## 🌟 Overview

The Skippy Multi-Coach System provides specialized AI coaches for different aspects of your work:

- **🏭 EWM Coach** - SAP EWM operations and warehouse management
- **📋 Business Analyst Coach** - Requirements analysis and process optimization  
- **🔧 Support Coach** - Technical support and troubleshooting
- **💻 Dev Guru Coach** - Source code analysis and development guidance
- **🌟 Mixed Coach** - Combines all capabilities with intelligent routing

## ✨ Key Features

### 🧠 **Specialized Expertise**
- Each coach has domain-specific knowledge and capabilities
- Context-aware responses tailored to your role and needs
- Advanced pattern recognition and analysis

### 💬 **Modern Chat Interface**
- Scrollable chat containers with message bubbles
- Real-time conversation history
- Auto-scroll and responsive design
- Coach-specific styling and indicators

### 🔄 **Mix & Match Capabilities**
- Combine multiple coaches for comprehensive assistance
- Intelligent question routing to the most appropriate specialist
- Seamless switching between different expertise areas

### 📚 **Comprehensive Knowledge Bases**
- Document processing (PDF, Word, Confluence)
- Source code analysis (7000+ classes, 400+ tables)
- Database schema understanding
- Business process documentation

## 🚀 Quick Start

### 1. Setup the System
```bash
# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup_multi_coach.py
```

### 2. Launch the Application
```bash
# Launch multi-coach system
python launch_multi_coach.py
```

### 3. Open in Browser
The application will automatically open at `http://localhost:8501`

## 🎯 Coach Specializations

### 🏭 EWM Coach
**Specializes in SAP EWM operations and processes**

**Capabilities:**
- Inbound and outbound process guidance
- Putaway and picking strategies
- Inventory management and control
- Quality management processes
- Configuration and customization
- Error troubleshooting

**Data Sources:** PDF documentation, Word documents, process guides

**Example Questions:**
- "How do I process an inbound delivery in EWM?"
- "What's the next step in putaway confirmation?"
- "I'm getting an error in picking, how do I resolve it?"

### 📋 Business Analyst Coach
**Specializes in business analysis and requirements**

**Capabilities:**
- Requirements analysis and documentation
- Process mapping and optimization
- Stakeholder communication strategies
- Business case development
- Gap analysis and solution recommendations
- Workflow optimization

**Data Sources:** Word documents, business requirements, process documentation

**Example Questions:**
- "How do I write effective user stories?"
- "What's the best way to document business processes?"
- "How can I improve this workflow?"

### 🔧 Support Coach
**Specializes in technical support and troubleshooting**

**Capabilities:**
- Technical troubleshooting and problem resolution
- User support and step-by-step instructions
- Error diagnosis and solution recommendations
- Documentation analysis and guidance
- Escalation procedures
- Performance optimization

**Data Sources:** Word documents, Confluence pages, PDFs, support documentation

**Example Questions:**
- "I'm getting a connection error, how do I fix it?"
- "The system is running slowly, what should I check?"
- "How do I configure the API endpoint?"

### 💻 Dev Guru Coach
**Specializes in software development and code analysis**

**Capabilities:**
- Source code analysis and review
- Architecture design and patterns
- Code quality and best practices
- Bug detection and fixes
- Performance optimization
- Database design and queries
- Technical documentation

**Data Sources:** Source code files, database schemas, technical documentation

**Example Questions:**
- "How can I refactor this code to improve maintainability?"
- "What's the best design pattern for this scenario?"
- "I found a bug in this function, how do I fix it?"
- "How can I optimize this database query?"

### 🌟 Mixed Coach
**Combines all capabilities with intelligent routing**

**Capabilities:**
- Automatically routes questions to the most appropriate specialist
- Combines insights from multiple domains
- Provides comprehensive solutions
- Handles cross-domain questions

**Example Questions:**
- "I need to implement a new EWM feature, what should I consider?"
- "How do I document the business requirements for this code change?"
- "What's the best approach to troubleshoot this integration issue?"

## 📁 Data Organization

### Directory Structure
```
data/
├── ewm_db/           # EWM Coach vector database
├── ba_db/            # Business Analyst Coach database
├── support_db/       # Support Coach database
├── dev_db/           # Dev Guru Coach database
├── pdfs/             # PDF documents
├── source_code/      # Source code files
└── schemas/          # Database schema files
```

### Adding Your Data

**For EWM Coach:**
- Add SAP EWM documentation PDFs to `./data/pdfs/`
- Add process guides and manuals
- Run setup script to rebuild index

**For Business Analyst Coach:**
- Add business requirements documents to `./data/pdfs/`
- Add process documentation and workflows
- Include stakeholder communication templates

**For Support Coach:**
- Add support documentation to `./data/pdfs/`
- Include troubleshooting guides
- Add user manuals and FAQs

**For Dev Guru Coach:**
- Add source code files to `./data/source_code/`
- Include database schema files in `./data/schemas/`
- Add technical documentation

## ⚙️ Configuration

### Environment Variables
Create a `.env` file with your Azure OpenAI credentials:

```env
# Azure OpenAI Configuration
AZURE_EMBEDDING_ENDPOINT=https://your-embedding-endpoint.openai.azure.com/
AZURE_EMBEDDING_API_KEY=your-embedding-api-key
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_EMBEDDING_API_VERSION=2022-12-01

AZURE_CHAT_ENDPOINT=https://your-chat-endpoint.openai.azure.com/
AZURE_CHAT_API_KEY=your-chat-api-key
AZURE_CHAT_DEPLOYMENT=gpt-4o
AZURE_CHAT_API_VERSION=2023-05-15

# Database Paths (Optional)
EWM_CHROMA_DB_PATH=./data/ewm_db
BA_CHROMA_DB_PATH=./data/ba_db
SUPPORT_CHROMA_DB_PATH=./data/support_db
DEV_CHROMA_DB_PATH=./data/dev_db

# Application Settings (Optional)
MAX_CLARIFICATIONS=5
DOC_RETRIEVAL_COUNT=3
LLM_TEMPERATURE=0.1
```

## 🔧 Advanced Features

### Index Building
The system includes specialized index builders for different data types:

- **Document Index Builder** - Processes PDFs and Word documents
- **Source Code Index Builder** - Analyzes code structure and patterns
- **Database Schema Index Builder** - Understands database schemas

### Mix & Match Capabilities
Create custom coach combinations:

```python
# Example: Create a coach that combines EWM and Dev Guru capabilities
mixed_coach = MixedCoach([CoachType.EWM_COACH, CoachType.DEV_GURU], config)
```

### Custom Coach Creation
Extend the system with your own specialized coaches:

```python
class CustomCoach(BaseCoach):
    def _define_capabilities(self) -> CoachCapabilities:
        return CoachCapabilities(
            data_sources=[DataSourceType.PDF],
            can_read_documents=True,
            # ... other capabilities
        )
```

## 🛠️ Troubleshooting

### Common Issues

**1. Coach Not Available**
- Check if the corresponding database exists
- Run setup script to rebuild indexes
- Verify data files are in the correct directories

**2. No Relevant Responses**
- Ensure your data files are properly indexed
- Check if the coach has access to relevant documentation
- Try rephrasing your question

**3. Performance Issues**
- Reduce the number of documents being indexed
- Check system resources
- Consider using smaller chunk sizes

### Logs
Check the logs for detailed information:
- `./logs/skippy_multi_coach.log` - Main application logs
- `./logs/setup.log` - Setup process logs

## 📈 Performance Optimization

### For Large Codebases (7000+ classes)
- Use the source code index builder
- Consider splitting large files
- Use appropriate chunk sizes
- Monitor memory usage

### For Large Document Collections
- Organize documents by topic
- Use the document index builder
- Consider document preprocessing
- Monitor indexing time

## 🔒 Security Considerations

- Store API keys securely in environment variables
- Don't commit sensitive data to version control
- Use appropriate access controls for data directories
- Consider data encryption for sensitive information

## 🚀 Future Enhancements

- **Real-time Collaboration** - Multiple users working together
- **Custom Coach Training** - Train coaches on specific datasets
- **API Integration** - REST API for programmatic access
- **Advanced Analytics** - Usage analytics and insights
- **Plugin System** - Extensible architecture for custom capabilities

## 📞 Support

For issues or questions:
1. Check the logs in the `./logs/` directory
2. Verify your configuration settings
3. Ensure all dependencies are properly installed
4. Review the troubleshooting section above

## 📄 License

MIT License - See LICENSE file for details.

---

**Version**: 1.0.0 (Multi-Coach Edition)  
**Last Updated**: September 2024

**Ready to experience the power of specialized AI coaching!** 🎉
