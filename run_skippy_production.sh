#!/bin/bash
# Skippy SAP EWM Production Coach Launcher
# Usage: ./run_skippy_production.sh

echo "🚀 Starting Skippy - Production SAP EWM Coach"
echo "=============================================="

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

# Launch Production Skippy Coach
echo "🚀 Launching Production Skippy Coach..."
echo ""
echo "   🔒 SECURITY FEATURES:"
echo "      • Secure configuration management (no hardcoded credentials)"
echo "      • Comprehensive input validation & sanitization"  
echo "      • Structured logging with security monitoring"
echo ""
echo "   🏗️  ARCHITECTURE IMPROVEMENTS:"
echo "      • Clean separation of concerns"
echo "      • Type safety with enums & dataclasses"
echo "      • Modular, testable components"
echo "      • Performance optimizations (caching)"
echo ""
echo "   🧠 INTELLIGENT FEATURES:"
echo "      • Context-aware conversation memory"
echo "      • Smart clarification system" 
echo "      • Role & intent classification"
echo "      • Progressive coaching methodology"
echo ""
echo "   📊 MONITORING & RELIABILITY:"
echo "      • Comprehensive error handling"
echo "      • Structured logging to ./logs/"
echo "      • Performance metrics & caching"
echo "      • Production-ready exception management"
echo ""
echo "   🌐 ACCESS:"
echo "      • URL: http://localhost:8505"
echo "      • Press Ctrl+C to stop"
echo ""

streamlit run skippy_coach_production.py --server.port 8505 --server.address localhost

echo ""
echo "👋 Thanks for using Production Skippy Coach!"
