#!/usr/bin/env python3
"""
Skippy Coach Launcher
Simple launcher script for the Skippy Coach application
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Skippy Coach application"""
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.absolute()
        
        # Path to the main application
        app_path = script_dir / "skippy_coach_production.py"
        
        if not app_path.exists():
            print("❌ Error: skippy_coach_production.py not found!")
            print(f"Expected location: {app_path}")
            return 1
        
        print("🚀 Starting Skippy Coach...")
        print("📱 The application will open in your web browser")
        print("🌐 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 50)
        
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port=8501", "--server.headless=false"]
        
        subprocess.run(cmd, cwd=script_dir)
        
    except KeyboardInterrupt:
        print("\n👋 Skippy Coach stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching Skippy Coach: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
