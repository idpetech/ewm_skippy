#!/usr/bin/env python3
"""
Skippy All Coaches Launcher

This script provides options to launch different versions of Skippy:
1. Original EWM Coach (Production)
2. Multi-Coach System (All Specialized Coaches)
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch Skippy coaching system"""
    try:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.absolute()
        
        print("🚀 Skippy Coaching System Launcher")
        print("=" * 50)
        print("Choose which version to launch:")
        print()
        print("1. 🏭 EWM Coach (Production) - Original SAP EWM specialist")
        print("2. 🌟 Multi-Coach System - All specialized coaches")
        print("3. 🔧 Setup Multi-Coach System - Build indexes and configure")
        print("4. ❌ Exit")
        print()
        
        while True:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                # Launch original EWM coach
                app_path = script_dir / "skippy_coach_production.py"
                if app_path.exists():
                    print("\n🏭 Launching EWM Coach...")
                    print("📱 The application will open in your web browser")
                    print("🌐 URL: http://localhost:8501")
                    print("⏹️  Press Ctrl+C to stop the application")
                    print("-" * 50)
                    
                    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port=8501", "--server.headless=false"]
                    subprocess.run(cmd, cwd=script_dir)
                else:
                    print("❌ EWM Coach not found!")
                break
                
            elif choice == "2":
                # Launch multi-coach system
                app_path = script_dir / "skippy_multi_coach.py"
                if app_path.exists():
                    print("\n🌟 Launching Multi-Coach System...")
                    print("📱 The application will open in your web browser")
                    print("🌐 URL: http://localhost:8501")
                    print("⏹️  Press Ctrl+C to stop the application")
                    print("-" * 50)
                    print("Available Coaches:")
                    print("🏭 EWM Coach - SAP warehouse operations")
                    print("📋 Business Analyst Coach - Requirements & processes")
                    print("🔧 Support Coach - Technical troubleshooting")
                    print("💻 Dev Guru Coach - Code analysis & development")
                    print("🌟 Mixed Coach - All capabilities combined")
                    print("-" * 50)
                    
                    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port=8501", "--server.headless=false"]
                    subprocess.run(cmd, cwd=script_dir)
                else:
                    print("❌ Multi-Coach System not found!")
                break
                
            elif choice == "3":
                # Run setup
                setup_path = script_dir / "setup_multi_coach.py"
                if setup_path.exists():
                    print("\n🔧 Running Multi-Coach Setup...")
                    print("This will build indexes and configure the system.")
                    print("-" * 50)
                    
                    cmd = [sys.executable, str(setup_path)]
                    subprocess.run(cmd, cwd=script_dir)
                else:
                    print("❌ Setup script not found!")
                break
                
            elif choice == "4":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        
    except KeyboardInterrupt:
        print("\n👋 Skippy stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
