# 🚀 Start Here - RAG Observatory

Welcome to RAG Observatory! This guide will get you up and running in 5 minutes.

## Step 1: Setup (2 minutes)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python setup_verify.py
```

You should see: `✅ SUCCESS! RAG Observatory is ready to use`

## Step 2: Run Tests (1 minute)

```bash
python tests/test_all.py
```

All tests should pass. If any fail, check the error messages.

## Step 3: Try a Quick Example (2 minutes)

Create a file `quick_demo.py`:

```python
from src.observable_rag import ObservableRAG
from llama_index.core import Document

# Create some documents
documents = [
    Document(text="Python is a programming language."),
    Document(text="Machine learning uses algorithms to learn from data."),
    Document(text="RAG combines retrieval and generation for better answers."),
]

# Initialize RAG system
rag = ObservableRAG(
    documents=documents,
    top_k=2
)

# Ask a question
result = rag.query("What is Python?")
print(f"Answer: {result['answer']}")
print(f"Relevance: {result['metrics']['avg_relevance']:.3f}")
print(f"Failures: {result['num_failures']}")
```

Run it:
```bash
python quick_demo.py
```

## Step 4: Launch Dashboard

```bash
cd dashboards
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501` and explore the dashboard!

## Next Steps

- Read `README.md` for detailed documentation
- Check `QUICK_REFERENCE.md` for common commands
- Run `python tests/test_integration.py` for end-to-end testing
- Customize thresholds in `src/config.py`

## Need Help?

- Check `README.md` for detailed docs
- Review test files in `tests/` for usage examples
- Check error messages - they're usually descriptive

Happy observing! 🔍

