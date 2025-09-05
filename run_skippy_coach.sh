#!/bin/bash
# Skippy SAP EWM Coach Launcher
# Usage: ./run_skippy_coach.sh

echo "🎯 Starting Skippy - SAP EWM Coach (Coaching Edition)"
echo "===================================================="

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

# Launch Skippy Coach
echo "🎯 Launching Skippy Coach..."
echo "   🎯 Coaching Mode: Step-by-step guidance"
echo "   📚 Intent Classification: Learning, Navigation, Error Resolution"  
echo "   🧭 Role-Aware: Execution, Supervisor, Configuration"
echo "   Open your browser to: http://localhost:8502"
echo "   Press Ctrl+C to stop Skippy Coach"
echo ""

streamlit run skippy_coach.py --server.port 8502 --server.address localhost

echo ""
echo "👋 Thanks for using Skippy Coach!"
