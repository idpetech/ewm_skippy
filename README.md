# 🤖 Skippy - SAP EWM AI Assistant

**Your friendly AI coach for SAP Extended Warehouse Management!**

Skippy is a production-ready AI assistant that helps warehouse operators and SAP EWM consultants with configuration, troubleshooting, and process guidance. Built with Streamlit, OpenAI GPT, and ChromaDB for intelligent document retrieval.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/framework-Streamlit-red)

## ✨ Features

- 🎯 **Expert SAP EWM Knowledge**: Covers warehouse operations, configuration, troubleshooting
- 💬 **Conversational Interface**: Natural chat with memory and context awareness
- 📚 **Document-Aware**: Upload PDF training materials to expand knowledge base
- 🔍 **Smart Retrieval**: ChromaDB vector database for relevant context finding
- 🤖 **Multiple AI Models**: Support for GPT-3.5-turbo, GPT-4, and GPT-4-turbo
- 📱 **Professional UI**: Clean interface with avatar support and status indicators
- 🔧 **Production Ready**: Comprehensive error handling, logging, and deployment support

## 🎯 Use Cases

### 🏭 **Warehouse Operations**
- Receiving and putaway processes
- Picking and packing workflows
- Shipping and outbound logistics
- Inventory management and cycle counting

### 🔧 **System Configuration & Troubleshooting**
- EWM setup and customization
- Error resolution and debugging  
- Transaction guidance and best practices
- Integration with other SAP modules (MM, SD, PP)

### 📱 **Mobile Warehouse Solutions**
- RF device configuration
- Mobile warehouse processes
- Handheld scanner operations

### 📚 **Training & Process Improvement**
- Step-by-step process guidance
- Best practice recommendations
- User training support
- Workflow optimization

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Git (for cloning the repository)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/ewm_skippy.git
cd ewm_skippy
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 4. Prepare Training Data

```bash
# Create data directory if it doesn't exist
mkdir -p data

# Add your SAP EWM training PDFs to the data/ folder
# Example: data/sap_ewm_training.pdf, data/warehouse_processes.pdf
```

### 5. Build Knowledge Base

```bash
# Process PDFs and build ChromaDB index
python build_index.py
```

### 6. Run Application

```bash
# Start Skippy
streamlit run skippy_app.py
```

Open your browser to `http://localhost:8501` and start chatting with Skippy!

## 📁 Project Structure

```
ewm_skippy/
├── skippy_app.py              # Main Streamlit application
├── build_index.py             # PDF processing and ChromaDB indexing
├── requirements.txt           # Python dependencies
├── .env.example              # Environment configuration template
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── data/                     # Training PDF documents
│   └── .gitkeep
├── assets/                   # Application assets (avatars, etc.)
│   └── skippy.png           # Skippy's avatar (optional)
├── logs/                     # Application logs
│   └── .gitkeep
└── chroma/                   # ChromaDB vector database
    └── (generated automatically)
```

## 🔧 Configuration

### Environment Variables

All configuration is managed through environment variables. Copy `.env.example` to `.env` and customize:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `OPENAI_MODEL_DEFAULT` | Default AI model | gpt-3.5-turbo |
| `OPENAI_TEMPERATURE_DEFAULT` | Response creativity (0.0-1.0) | 0.7 |
| `CHROMA_DB_PATH` | Vector database path | ./chroma |
| `DATA_DIRECTORY` | PDF documents directory | ./data |
| `DEFAULT_CONTEXT_DOCS` | Documents per query | 3 |
| `EMBEDDING_MODEL` | Text embedding model | all-MiniLM-L6-v2 |

### Supported AI Models

- **GPT-3.5-turbo**: Fast, cost-effective, good quality
- **GPT-4**: Higher quality, more expensive
- **GPT-4-turbo**: Latest model with improved performance
- **GPT-4o**: Optimized model variant

## 📚 Usage Examples

### Basic Questions
```
"How do I reverse a goods receipt in MIGO?"
"What should I do when I get error WM_TASK_001?"
"Walk me through the wave planning process"
```

### Configuration Help
```
"How do I configure storage types in EWM?"
"What are the steps to set up RF menus?"
"How do I integrate EWM with SAP TM?"
```

### Troubleshooting
```
"I'm getting a posting error during goods receipt"
"The warehouse task is stuck in processing status"
"How do I resolve bin blocking issues?"
```

## 🚀 Deployment

### Streamlit Cloud

1. Push your repository to GitHub
2. Connect to [Streamlit Cloud](https://share.streamlit.io)
3. Add your `OPENAI_API_KEY` in the app settings
4. Deploy!

### Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway add --service postgresql  # Optional: for persistent storage
railway deploy
```

Add environment variables in the Railway dashboard.

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "skippy_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🔒 Security & Privacy

- **API Keys**: Never commit API keys to version control
- **PDF Content**: Be mindful of proprietary SAP documentation
- **Logging**: Logs may contain query content - review before sharing
- **Access Control**: Implement authentication for production deployments
- **Data Retention**: Configure appropriate data retention policies

## 🛠️ Development

### Adding New Features

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes with proper documentation
4. **Test** thoroughly with various SAP EWM scenarios
5. **Commit** your changes: `git commit -m 'Add amazing feature'`
6. **Push** to the branch: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

### Code Quality

```bash
# Format code
black *.py

# Lint code
flake8 *.py

# Type checking
mypy *.py
```

## 📊 Monitoring & Analytics

### Logs

Check application logs in the `logs/` directory:
- `skippy_app.log` - Main application logs
- `build_index.log` - Indexing process logs

### Metrics

Monitor these key metrics:
- Response time per query
- ChromaDB query performance
- OpenAI API usage and costs
- Error rates and types
- User engagement patterns

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines:

1. **Bug Reports**: Use GitHub issues with detailed reproduction steps
2. **Feature Requests**: Describe use case and expected behavior
3. **Code Contributions**: Follow coding standards and include tests
4. **Documentation**: Help improve documentation and examples

## 📋 Changelog

### v1.0.0 (2024-08-29)
- ✨ Initial release with full SAP EWM knowledge base
- 🤖 OpenAI GPT integration with multiple model support
- 📚 ChromaDB vector database for document retrieval
- 🎨 Professional Streamlit UI with conversational interface
- 📄 PDF upload and indexing capabilities
- 🔧 Production-ready error handling and logging
- 🚀 Multi-platform deployment support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing the GPT models
- **Streamlit** for the amazing web framework
- **ChromaDB** for the vector database capabilities
- **Langchain** for document processing utilities
- **SAP Community** for EWM knowledge and best practices

---

**Made with ❤️ for the SAP EWM community**

*Skippy helps warehouse operators succeed with SAP Extended Warehouse Management through intelligent, conversational AI assistance.*
