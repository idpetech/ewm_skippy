#!/bin/bash
# Skippy SAP EWM Clean Coach Launcher
# Usage: ./run_skippy_clean.sh

echo "🧠 Starting Skippy - Clean SAP EWM Coach (Refactored Edition)"
echo "============================================================"

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

# Launch Clean Skippy Coach
echo "🧠 Launching Clean Skippy Coach..."
echo "   🚀 Clean Architecture: Simplified and maintainable"
echo "   📦 Reduced Complexity: Single coach class"  
echo "   🎯 Unified Prompts: No template duplication"
echo "   ⚡ Better Error Handling: Proper exception management"
echo "   💡 Minimal UI: Clean, focused interface"
echo "   Open your browser to: http://localhost:8504"
echo "   Press Ctrl+C to stop Clean Skippy Coach"
echo ""

streamlit run skippy_coach_clean.py --server.port 8504 --server.address localhost

echo ""
echo "👋 Thanks for using Clean Skippy Coach!"
