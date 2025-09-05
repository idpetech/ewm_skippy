#!/bin/bash
# Skippy SAP EWM Chatbot Launcher
# Usage: ./run_skippy.sh

echo "🤖 Starting Skippy - SAP EWM Expert Chatbot"
echo "============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "📋 Installing/updating requirements..."
pip install -r requirements.txt --quiet

# Check if ChromaDB index exists
if [ ! -d "data/eWMDB" ]; then
    echo "⚠️  Warning: ChromaDB index not found at data/eWMDB"
    echo "   Please run the index builder first to create the knowledge base"
    echo ""
fi

# Launch Skippy
echo "🚀 Launching Skippy chatbot..."
echo "   Open your browser to: http://localhost:8501"
echo "   Press Ctrl+C to stop Skippy"
echo ""

streamlit run skippy_chatbot.py --server.port 8501 --server.address localhost

echo ""
echo "👋 Thanks for using Skippy!"
