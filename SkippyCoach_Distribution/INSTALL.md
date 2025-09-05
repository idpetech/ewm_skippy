# 🚀 Skippy Coach Installation Guide

## Quick Installation

### For macOS/Linux:
```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Run the application
./run_skippy.sh
```

### For Windows:
```cmd
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
run_skippy.bat
```

### Alternative (All Platforms):
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with Python launcher
python launch_skippy.py
```

## Prerequisites

- **Python 3.8+** installed on your system
- **pip** package manager
- **Internet connection** for downloading dependencies

## Configuration (Optional)

1. Copy `env.example` to `.env`
2. Edit `.env` with your Azure OpenAI credentials
3. If no `.env` file is provided, demo credentials will be used

## Troubleshooting

### Python Not Found
- **macOS**: Install Python from [python.org](https://python.org) or use Homebrew: `brew install python3`
- **Windows**: Install Python from [python.org](https://python.org) or Microsoft Store
- **Linux**: Use your package manager: `sudo apt install python3 python3-pip` (Ubuntu/Debian)

### Permission Denied (macOS/Linux)
```bash
chmod +x run_skippy.sh
```

### Port Already in Use
The application uses port 8501. If it's busy:
- Kill existing processes: `lsof -ti:8501 | xargs kill -9`
- Or change the port in the launcher scripts

## What's Included

- ✅ Main application (`skippy_coach_production.py`)
- ✅ ChromaDB vector database (`chroma/`)
- ✅ SAP EWM documentation (`data/`)
- ✅ Launcher scripts for all platforms
- ✅ Complete dependency list (`requirements.txt`)
- ✅ Configuration template (`env.example`)
- ✅ Comprehensive documentation (`README.md`)

## Support

If you encounter issues:
1. Check the logs in the `logs/` directory
2. Verify Python and pip are properly installed
3. Ensure all dependencies installed successfully
4. Check your internet connection

---

**Ready to use!** 🎉
