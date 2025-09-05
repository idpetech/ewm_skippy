@echo off
REM Skippy Coach Launcher Script for Windows

echo 🚀 Starting Skippy Coach - SAP EWM Assistant
echo ==============================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: pip is not installed or not in PATH
    echo Please install pip and try again
    pause
    exit /b 1
)

REM Install dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Error: Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed successfully
) else (
    echo ⚠️  Warning: requirements.txt not found, skipping dependency installation
)

REM Check if main application file exists
if not exist "skippy_coach_production.py" (
    echo ❌ Error: skippy_coach_production.py not found
    echo Please ensure you're running this script from the correct directory
    pause
    exit /b 1
)

echo 🌐 Starting Streamlit application...
echo 📱 The application will open in your web browser
echo 🌐 URL: http://localhost:8501
echo ⏹️  Press Ctrl+C to stop the application
echo ----------------------------------------

REM Launch Streamlit
python -m streamlit run skippy_coach_production.py --server.port=8501 --server.headless=false

echo 👋 Skippy Coach stopped
pause
