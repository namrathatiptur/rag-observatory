# How to Test RAG Observatory

## Understanding the Workflow

RAG Observatory doesn't work by uploading a CSV and running queries on it. Instead:

1. **You provide documents** (your knowledge base)
2. **You run queries** against those documents
3. **RAG Observatory tracks metrics** for each query
4. **You view results** in the dashboard

## Quick Test Example

### Step 1: Create a Test Script

```python
from src.observable_rag import ObservableRAG
from llama_index.core import Document

# Your documents (knowledge base)
documents = [
    Document(text="Python is a programming language."),
    Document(text="Machine learning uses algorithms."),
    Document(text="RAG combines retrieval and generation."),
]

# Initialize RAG with observability
rag = ObservableRAG(documents=documents, top_k=3)

# Run a query
result = rag.query("What is Python?")

# See metrics
print(f"Relevance: {result['metrics']['avg_relevance']:.3f}")
print(f"Failures: {result['num_failures']}")
```

### Step 2: Run the Test

```bash
cd /Users/namrathatm/rag_observatory
source venv/bin/activate
python test_rag_example.py
```

## Testing with Your Own Data

### Option 1: From CSV (if CSV contains text documents)

```python
import pandas as pd
from src.observable_rag import ObservableRAG
from llama_index.core import Document

# Load your CSV
df = pd.read_csv("your_documents.csv")

# Convert to documents (assuming 'text' column)
documents = [Document(text=row['text']) for _, row in df.iterrows()]

# Create RAG system
rag = ObservableRAG(documents=documents)

# Test queries
result = rag.query("Your question here?")
print(result['metrics'])
```

### Option 2: From Text Files

```python
from llama_index.core import SimpleDirectoryReader
from src.observable_rag import ObservableRAG

# Load documents from a directory
documents = SimpleDirectoryReader("your_documents/").load_data()

# Create RAG system
rag = ObservableRAG(documents=documents)

# Test
result = rag.query("Your question?")
```

### Option 3: From Database/API

```python
# Fetch your data from anywhere
your_data = fetch_from_database()

# Convert to documents
documents = [Document(text=item['content']) for item in your_data]

# Create RAG system
rag = ObservableRAG(documents=documents)

# Test
result = rag.query("Your question?")
```

## What Gets Tested?

For each query, RAG Observatory checks:

1. **Relevance**: Are the right documents retrieved?
2. **Coverage**: Do retrieved docs cover the query?
3. **Grounding**: Is the answer based on retrieved context?
4. **Usage**: Is the context actually used?
5. **Failures**: Any problems detected?

## Viewing Results

### Method 1: In Code

```python
result = rag.query("What is Python?")
print(result['metrics'])  # See all metrics
print(result['failures'])  # See any failures
```

### Method 2: Health Report

```python
report = rag.get_health_report()
print(f"Failure rate: {report['failure_rate']:.1%}")
```

### Method 3: Dashboard

```python
# Export metrics
rag.export_data("metrics.csv")

# Then upload metrics.csv to dashboard
```

## Example Test Scenarios

### Test 1: Good Query (Should Succeed)

```python
result = rag.query("What is Python?")
# Expected: High relevance, no failures
assert result['metrics']['avg_relevance'] > 0.7
assert result['num_failures'] == 0
```

### Test 2: Out-of-Domain Query (Should Fail)

```python
result = rag.query("What is quantum physics?")
# Expected: Low relevance, failures detected
assert result['num_failures'] > 0
```

### Test 3: Ambiguous Query (May Partially Succeed)

```python
result = rag.query("Tell me about AI")
# Expected: Medium relevance, some failures possible
```

## Running the Full Test Suite

```bash
# Run all tests
python tests/test_all.py

# Run integration test
python tests/test_integration.py

# Run example
python test_rag_example.py
```

## Common Questions

### Q: Can I upload a CSV and query it?

**A:** Not directly. The workflow is:
1. Load CSV → Convert to Documents
2. Create RAG system with documents
3. Run queries → Get metrics
4. Export metrics → View in dashboard

### Q: What if my CSV has documents?

**A:** Perfect! Convert CSV rows to Documents:

```python
df = pd.read_csv("documents.csv")
documents = [Document(text=row['content']) for _, row in df.iterrows()]
rag = ObservableRAG(documents=documents)
```

### Q: How do I know if retrieval is working?

**A:** Check the metrics:
- `avg_relevance` > 0.7 = Good retrieval
- `query_coverage` > 0.6 = Good coverage
- `num_failures` = 0 = No problems

### Q: Can I test without Ollama/LLM?

**A:** Yes! The system uses a mock LLM if Ollama isn't available. Metrics still work correctly.

## Next Steps

1. Run `python test_rag_example.py` to see it in action
2. Modify it with your own documents
3. Export metrics and view in dashboard
4. Iterate and improve!

