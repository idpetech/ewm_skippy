#!/bin/bash
# Skippy SAP EWM Smart Coach Launcher (Memory-Fixed)
# Usage: ./run_skippy_smart.sh

echo "🧠 Starting Skippy - Smart SAP EWM Coach (Memory-Fixed Edition)"
echo "=============================================================="

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

# Launch Skippy Smart Coach
echo "🧠 Launching Skippy Smart Coach..."
echo "   🧠 Memory-Aware: Remembers original questions"
echo "   🎯 Smart Clarifications: No repetitive questions"
echo "   📚 Context Intelligence: Progressive learning"  
echo "   🧭 Conversation Tracking: Maintains thread continuity"
echo "   Open your browser to: http://localhost:8503"
echo "   Press Ctrl+C to stop Skippy Smart Coach"
echo ""

streamlit run skippy_coach_fixed.py --server.port 8503 --server.address localhost

echo ""
echo "👋 Thanks for using Skippy Smart Coach!"
