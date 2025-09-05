#!/usr/bin/env python3
"""
Skippy Multi-Coach System Launcher

Launcher script for the comprehensive multi-coach system with:
- EWM Coach
- Business Analyst Coach  
- Support Coach
- Dev Guru Coach
- Mixed Coach (combines all capabilities)
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Skippy Multi-Coach System"""
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.absolute()
        
        # Path to the main application
        app_path = script_dir / "skippy_multi_coach.py"
        
        if not app_path.exists():
            print("❌ Error: skippy_multi_coach.py not found!")
            print(f"Expected location: {app_path}")
            return 1
        
        print("🚀 Starting Skippy Multi-Coach System...")
        print("📱 The application will open in your web browser")
        print("🌐 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 60)
        print("Available Coaches:")
        print("🏭 EWM Coach - SAP warehouse operations")
        print("📋 Business Analyst Coach - Requirements & processes")
        print("🔧 Support Coach - Technical troubleshooting")
        print("💻 Dev Guru Coach - Code analysis & development")
        print("🌟 Mixed Coach - All capabilities combined")
        print("-" * 60)
        
        # Launch Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port=8501", "--server.headless=false"]
        
        subprocess.run(cmd, cwd=script_dir)
        
    except KeyboardInterrupt:
        print("\n👋 Skippy Multi-Coach System stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching Skippy Multi-Coach System: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
