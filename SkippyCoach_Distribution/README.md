# 🚀 Skippy Coach - SAP EWM Assistant

**Enterprise-grade SAP EWM coaching assistant with advanced context management and chatbot interface.**

## 📋 Features

- **🧠 Smart Context Management**: Maintains conversation history and builds upon previous exchanges
- **💬 Modern Chat Interface**: Scrollable chat container with message bubbles and timestamps
- **🎯 Intent Recognition**: Automatically detects learning, navigation, and error resolution needs
- **👥 Role Detection**: Adapts responses based on user role (execution, supervisor, configuration)
- **📚 Document Retrieval**: Uses ChromaDB for intelligent document search and retrieval
- **🔒 Enterprise Security**: Secure configuration management with environment variables
- **⚡ Production Ready**: Comprehensive error handling, logging, and input validation

## 🚀 Quick Start

### Option 1: Using the Launcher Script (Recommended)

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python launch_skippy.py
   ```

3. **Open in Browser**: The application will automatically open at `http://localhost:8501`

### Option 2: Direct Streamlit Command

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with Streamlit**:
   ```bash
   streamlit run skippy_coach_production.py
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the distribution directory with your Azure OpenAI credentials:

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

# Database Configuration
CHROMA_DB_PATH=./chroma

# Application Settings (Optional)
MAX_CLARIFICATIONS=5
DOC_RETRIEVAL_COUNT=3
LLM_TEMPERATURE=0.1
```

### Default Configuration

If no environment variables are set, the application will use demo/development credentials (not recommended for production).

## 📁 File Structure

```
SkippyCoach_Distribution/
├── skippy_coach_production.py    # Main application
├── launch_skippy.py              # Launcher script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── chroma/                       # ChromaDB vector database
│   ├── chroma.sqlite3
│   └── [vector data files]
├── data/                         # SAP EWM documentation
│   ├── pdfs/
│   └── [PDF files]
└── logs/                         # Application logs
    └── skippy_coach.log
```

## 🎯 Usage

1. **Start the Application**: Use the launcher script or streamlit command
2. **Ask Questions**: Type your SAP EWM questions in the chat interface
3. **Get Contextual Help**: The coach will provide step-by-step guidance based on your role and intent
4. **Maintain Context**: The conversation history is preserved throughout your session

### Example Questions

- **Learning**: "What is putaway in SAP EWM?"
- **Navigation**: "I'm stuck in the putaway confirmation screen, what's the next step?"
- **Error Resolution**: "I'm getting an error when trying to confirm putaway"

## 🔧 Troubleshooting

### Common Issues

1. **Module Not Found Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **ChromaDB Not Found**:
   - Verify the `chroma/` directory exists and contains the database files

3. **Azure OpenAI Connection Issues**:
   - Check your API keys and endpoints in the `.env` file
   - Ensure your Azure OpenAI service is active

4. **Port Already in Use**:
   - The application uses port 8501 by default
   - Kill any existing Streamlit processes or change the port

### Logs

Check the `logs/skippy_coach.log` file for detailed error information.

## 🛡️ Security Notes

- **API Keys**: Never commit API keys to version control
- **Environment Variables**: Use `.env` files for local development
- **Production**: Use secure secret management for production deployments

## 📞 Support

For issues or questions:
1. Check the logs in the `logs/` directory
2. Verify your configuration settings
3. Ensure all dependencies are properly installed

## 🏗️ Architecture

- **Frontend**: Streamlit with custom CSS for chat interface
- **Backend**: Python with LangChain for AI orchestration
- **Vector Database**: ChromaDB for document retrieval
- **AI Models**: Azure OpenAI (GPT-4 and text-embedding-ada-002)
- **Context Management**: Custom conversation state tracking

## 📄 License

MIT License - See LICENSE file for details.

---

**Version**: 4.0.0 (Production Edition)  
**Last Updated**: September 2024
