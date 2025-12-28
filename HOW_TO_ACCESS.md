# How to Access the RAG Observatory Project

## 📍 Project Location

The project is located at:
```
/Users/namrathatm/rag_observatory
```

## 🚀 Quick Access Methods

### Method 1: Using Terminal (macOS/Linux)

```bash
# Navigate to project
cd /Users/namrathatm/rag_observatory

# Open in Finder (macOS)
open .

# Or open specific files
open RAG_OBSERVATORY_REPORT.html
open README.md
```

### Method 2: Using Finder (macOS)

1. Open Finder
2. Press `Cmd + Shift + G` (Go to Folder)
3. Type: `/Users/namrathatm/rag_observatory`
4. Press Enter

### Method 3: Using VS Code / Cursor

```bash
# Open entire project in editor
code /Users/namrathatm/rag_observatory

# Or in Cursor
cursor /Users/namrathatm/rag_observatory
```

## 📊 Accessing the HTML Report

### Option 1: Double-click (Easiest)
1. Navigate to the project folder
2. Find `RAG_OBSERVATORY_REPORT.html`
3. Double-click to open in your default browser

### Option 2: From Terminal
```bash
cd /Users/namrathatm/rag_observatory
open RAG_OBSERVATORY_REPORT.html
```

### Option 3: Direct Browser Access
1. Open your web browser (Chrome, Firefox, Safari, etc.)
2. Press `Cmd + O` (Open File)
3. Navigate to: `/Users/namrathatm/rag_observatory/RAG_OBSERVATORY_REPORT.html`
4. Click Open

### Option 4: Drag and Drop
1. Open your web browser
2. Drag `RAG_OBSERVATORY_REPORT.html` from Finder into the browser window

## 🎯 Accessing the Dashboard

The Streamlit dashboard is already running! Access it at:

**URL:** http://localhost:8501

### To open in browser:
```bash
# macOS
open http://localhost:8501

# Or manually navigate to:
# http://localhost:8501
```

### If dashboard is not running:
```bash
cd /Users/namrathatm/rag_observatory
source venv/bin/activate
cd dashboards
streamlit run streamlit_app.py
```

## 📁 Project Structure Overview

```
rag_observatory/
├── RAG_OBSERVATORY_REPORT.html  ← HTML Report (with charts!)
├── README.md                    ← Full documentation
├── START_HERE.md                 ← Quick start guide
├── QUICK_REFERENCE.md            ← Command reference
├── HOW_TO_ACCESS.md             ← This file
│
├── src/                         ← Source code
│   ├── metrics_collector.py
│   ├── failure_detector.py
│   ├── observable_rag.py
│   └── config.py
│
├── dashboards/                   ← Streamlit dashboard
│   └── streamlit_app.py
│
├── tests/                        ← Test suite
│   ├── test_all.py
│   ├── test_integration.py
│   └── ...
│
└── requirements.txt              ← Dependencies
```

## 🔍 Key Files to Access

### Documentation
- **RAG_OBSERVATORY_REPORT.html** - Interactive HTML report with charts
- **README.md** - Complete project documentation
- **START_HERE.md** - Quick start guide
- **QUICK_REFERENCE.md** - Command cheat sheet

### Code
- **src/observable_rag.py** - Main RAG system
- **src/metrics_collector.py** - Metrics computation
- **src/failure_detector.py** - Failure detection
- **dashboards/streamlit_app.py** - Dashboard application

### Tests
- **tests/test_all.py** - Comprehensive test suite
- **tests/test_integration.py** - End-to-end tests

## 💻 Quick Commands

```bash
# Navigate to project
cd /Users/namrathatm/rag_observatory

# View report
open RAG_OBSERVATORY_REPORT.html

# Read documentation
open README.md
cat START_HERE.md

# Run tests
source venv/bin/activate
python tests/test_all.py

# Launch dashboard
cd dashboards
streamlit run streamlit_app.py

# Verify setup
python setup_verify.py
```

## 🌐 Web Access

### HTML Report
- **File:** `RAG_OBSERVATORY_REPORT.html`
- **Location:** `/Users/namrathatm/rag_observatory/`
- **Open with:** Any web browser

### Dashboard
- **URL:** http://localhost:8501
- **Status:** Running in background
- **Access:** Open in browser

## 📱 Mobile Access (Optional)

If you want to access the dashboard from your phone/tablet:

1. Find your computer's IP address:
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```

2. On your mobile device, navigate to:
   ```
   http://YOUR_IP_ADDRESS:8501
   ```

3. Make sure your computer and mobile device are on the same network

## ✅ Verification

To verify everything is accessible:

```bash
cd /Users/namrathatm/rag_observatory

# Check report exists
ls -lh RAG_OBSERVATORY_REPORT.html

# Check dashboard is running
lsof -ti:8501 && echo "Dashboard running!" || echo "Dashboard not running"

# Check project structure
ls -la
```

## 🆘 Troubleshooting

### Report won't open?
- Make sure you're using a modern browser (Chrome, Firefox, Safari, Edge)
- Try right-click → Open With → Browser

### Dashboard not accessible?
- Check if it's running: `lsof -ti:8501`
- Restart it: `cd dashboards && streamlit run streamlit_app.py`
- Check if port 8501 is available

### Can't find project?
- Use: `cd /Users/namrathatm/rag_observatory`
- Or search in Finder: `rag_observatory`

## 📞 Quick Reference

| What | Where | How |
|------|-------|-----|
| HTML Report | `/Users/namrathatm/rag_observatory/RAG_OBSERVATORY_REPORT.html` | Double-click or `open` command |
| Dashboard | http://localhost:8501 | Open in browser |
| Documentation | `/Users/namrathatm/rag_observatory/README.md` | Text editor or `cat` |
| Source Code | `/Users/namrathatm/rag_observatory/src/` | Code editor |
| Tests | `/Users/namrathatm/rag_observatory/tests/` | Run with `python` |

---

**Need help?** Check `README.md` or `START_HERE.md` for more details!

