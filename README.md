# RAG Observatory

A comprehensive observability framework for Retrieval-Augmented Generation (RAG) systems. Monitor, analyze, and debug your RAG pipeline with real-time metrics, failure detection, and interactive dashboards.

## Overview

RAG Observatory provides production-ready observability for RAG systems built with LlamaIndex. It tracks 8 key performance metrics, automatically detects failures, and provides an interactive dashboard for analysis.

## Features

- **8 Key Metrics**: Track relevance, coverage, grounding, diversity, and more
- **Failure Detection**: Automatically detect 4 types of failures (bad retrieval, hallucinations, low coverage, poor grounding)
- **Interactive Dashboard**: Streamlit-based dashboard with 4 analysis tabs
- **LlamaIndex Integration**: Seamlessly wraps LlamaIndex RAG systems
- **Export & Analysis**: Export metrics to CSV for further analysis

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python setup_verify.py
```

### 3. Run Tests

```bash
python tests/test_all.py
```

### 4. Launch Dashboard

```bash
cd dashboards
streamlit run streamlit_app.py
```

## Usage

### Basic Example

```python
from src.observable_rag import ObservableRAG
from llama_index.core import Document

# Create documents
documents = [
    Document(text="Machine learning is a subset of AI."),
    Document(text="Deep learning uses neural networks."),
]

# Initialize RAG system
rag = ObservableRAG(
    documents=documents,
    llm_model="llama2",  # Or "gpt-3.5-turbo" for OpenAI
    top_k=5
)

# Query with metrics
result = rag.query("What is machine learning?")
print(f"Answer: {result['answer']}")
print(f"Avg Relevance: {result['metrics']['avg_relevance']:.3f}")
print(f"Failures: {result['num_failures']}")

# Get health report
report = rag.get_health_report()
print(f"Failure Rate: {report['failure_rate']:.1%}")

# Export metrics
rag.export_data("metrics.csv")
```

## Metrics

The system tracks 8 key metrics:

1. **Average Relevance**: Mean similarity score of retrieved chunks
2. **Query Coverage**: How well retrieved chunks cover the query
3. **Answer Grounding Rate**: How well answer is grounded in context
4. **Context Usage Rate**: How much of retrieved context is used
5. **Query-Answer Similarity**: Semantic similarity between query and answer
6. **Retrieval Diversity**: Diversity of retrieved chunks
7. **Answer Specificity**: Level of detail in the answer
8. **Number of Chunks Retrieved**: Count of retrieved documents

## Failure Detection

The system automatically detects 4 types of failures:

- **Bad Retrieval**: Low relevance or coverage scores
- **Potential Hallucination**: Answer not grounded in context
- **Low Coverage**: Query not well covered by retrieved chunks
- **Poor Grounding**: Answer doesn't use context effectively

Each failure is assigned a severity level: `low`, `medium`, `high`, or `critical`.

## Dashboard

The Streamlit dashboard provides 4 tabs:

1. **Real-time Monitor**: Live metrics and failure tracking
2. **Metrics Analysis**: Historical metrics visualization
3. **Failure Analysis**: Failure patterns and trends
4. **Query Inspector**: Detailed query-by-query analysis

## Project Structure

```
rag_observatory/
├── src/
│   ├── metrics_collector.py    # Metrics computation
│   ├── failure_detector.py     # Failure detection
│   ├── observable_rag.py       # Main RAG wrapper
│   └── config.py               # Configuration
├── dashboards/
│   └── streamlit_app.py        # Streamlit dashboard
├── tests/
│   ├── test_metrics_collector.py
│   ├── test_failure_detector.py
│   └── test_all.py             # Comprehensive test suite
├── data/                       # Data directory
├── notebooks/                  # Jupyter notebooks
├── requirements.txt
├── setup_verify.py
└── README.md
```

## Configuration

Configuration can be set via:

1. **Code**: Pass parameters to `ObservableRAG()`
2. **Config Object**: Use `RAGConfig` class
3. **Environment Variables**: Use `RAGConfig.from_env()`

```python
from src.config import RAGConfig

config = RAGConfig(
    embedding_model="BAAI/bge-small-en-v1.5",
    llm_model="llama2",
    top_k=5,
    strict_mode=False
)

rag = ObservableRAG(documents=docs, config=config)
```

## Requirements

- Python >= 3.10
- See `requirements.txt` for full dependency list

## LLM Support

The system supports multiple LLM providers:

- **Groq**: Fast inference with open-source models (recommended)
- **OpenAI**: GPT-3.5, GPT-4, and other OpenAI models
- **Ollama**: Local models (llama2, mistral, etc.)
- **Mock**: Testing mode (no actual LLM calls)

Configure the LLM in the dashboard or via code:

```python
rag = ObservableRAG(
    documents=documents,
    llm_model="llama3-8b-8192",  # Groq model
    llm_api_key="your-api-key",
    top_k=5
)
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

