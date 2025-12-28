# RAG Observatory - Quick Reference

## Quick Start Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify
python setup_verify.py

# Test
python tests/test_all.py

# Dashboard
cd dashboards && streamlit run streamlit_app.py
```

## Basic Usage

```python
from src.observable_rag import ObservableRAG
from llama_index.core import Document

# Initialize
rag = ObservableRAG(
    documents=[Document(text="Your document...")],
    top_k=5
)

# Query
result = rag.query("Your question?")
print(result['answer'])
print(f"Relevance: {result['metrics']['avg_relevance']:.3f}")

# Health report
report = rag.get_health_report()
print(f"Failure rate: {report['failure_rate']:.1%}")

# Export
rag.export_data("metrics.csv")
```

## Metrics Reference

| Metric | Range | Description |
|--------|-------|-------------|
| `avg_relevance` | 0-1 | Average similarity of retrieved chunks |
| `query_coverage` | 0-1 | How well chunks cover the query |
| `answer_grounding_rate` | 0-1 | How well answer is grounded in context |
| `context_usage_rate` | 0-1 | How much context is used |
| `query_answer_similarity` | 0-1 | Similarity between query and answer |
| `retrieval_diversity` | 0-1 | Diversity of retrieved chunks |
| `answer_specificity` | 0-1 | Level of detail in answer |
| `num_chunks_retrieved` | ≥0 | Number of chunks retrieved |

## Failure Types

- **bad_retrieval**: Low relevance or coverage
- **potential_hallucination**: Answer not grounded
- **low_coverage**: Query not well covered
- **poor_grounding**: Context not used effectively

## Configuration

```python
from src.config import RAGConfig

config = RAGConfig(
    embedding_model="BAAI/bge-small-en-v1.5",
    llm_model="llama2",  # or "gpt-3.5-turbo"
    top_k=5,
    strict_mode=False
)
```

## Troubleshooting

**Import errors**: Make sure virtual environment is activated
**Ollama not found**: System will use mock LLM for testing
**ChromaDB errors**: Check permissions on `./chroma_db` directory
**Dashboard won't start**: Check port 8501 is available

