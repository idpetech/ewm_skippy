#!/bin/bash

# Skippy Coach Launcher Script for macOS/Linux

echo "🚀 Starting Skippy Coach - SAP EWM Assistant"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3 and try again"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed or not in PATH"
    echo "Please install pip3 and try again"
    exit 1
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "⚠️  Warning: requirements.txt not found, skipping dependency installation"
fi

# Check if main application file exists
if [ ! -f "skippy_coach_production.py" ]; then
    echo "❌ Error: skippy_coach_production.py not found"
    echo "Please ensure you're running this script from the correct directory"
    exit 1
fi

echo "🌐 Starting Streamlit application..."
echo "📱 The application will open in your web browser"
echo "🌐 URL: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the application"
echo "----------------------------------------"

# Launch Streamlit
python3 -m streamlit run skippy_coach_production.py --server.port=8501 --server.headless=false

echo "👋 Skippy Coach stopped"
