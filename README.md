# RAG Observatory

Production-ready observability framework for Retrieval-Augmented Generation (RAG) systems. Monitor, analyze, and debug your RAG pipeline with real-time metrics, automatic failure detection, and interactive dashboards.

## Features

- **8 Key Metrics**: Track relevance, coverage, grounding, diversity, and more
- **Automatic Failure Detection**: Identifies 4 failure types with severity levels
- **Interactive Dashboard**: Streamlit-based UI with live querying and visualization
- **Multi-LLM Support**: Groq, OpenAI, and Ollama integration
- **Multi-Format Upload**: CSV, PDF, Excel, Parquet, and text files
- **Export & Analysis**: Export metrics to CSV for further analysis

## Quick Start

```bash
# Clone repository
git clone https://github.com/namrathatiptur/rag-observatory.git
cd rag-observatory

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify setup
python setup_verify.py

# Launch dashboard
cd dashboards
streamlit run streamlit_app.py
```

## Usage

```python
from src.observable_rag import ObservableRAG
from llama_index.core import Document

# Create documents
documents = [Document(text="Your document content here")]

# Initialize RAG system
rag = ObservableRAG(
    documents=documents,
    llm_model="llama3-8b-8192",  # Groq model
    llm_api_key="your-api-key",
    top_k=5
)

# Query with metrics
result = rag.query("Your question here")
print(f"Answer: {result['answer']}")
print(f"Relevance: {result['metrics']['avg_relevance']:.3f}")
print(f"Failures: {result['num_failures']}")
```

## Metrics

1. **Average Relevance**: Mean similarity of retrieved chunks
2. **Query Coverage**: How well chunks cover the query
3. **Answer Grounding Rate**: Answer grounded in context
4. **Context Usage Rate**: Utilization of retrieved context
5. **Query-Answer Similarity**: Semantic similarity between query and answer
6. **Retrieval Diversity**: Diversity of retrieved chunks
7. **Answer Specificity**: Level of detail in answer
8. **Chunks Retrieved**: Count of retrieved documents

## Failure Detection

Automatically detects:
- **Bad Retrieval**: Low relevance or coverage
- **Potential Hallucination**: Answer not grounded in context
- **Low Coverage**: Query not well covered
- **Poor Grounding**: Answer doesn't use context effectively

## Dashboard

Access the dashboard at `http://localhost:8501` after launching Streamlit. Features:
- Live document querying
- Real-time metrics visualization
- Failure analysis and trends
- Query history inspection
- Multi-format file upload

## LLM Support

- **Groq**: Fast inference with open-source models (recommended)
- **OpenAI**: GPT-3.5, GPT-4, and other models
- **Ollama**: Local models (llama2, mistral, etc.)

## Requirements

- Python >= 3.10
- See `requirements.txt` for dependencies

## Project Structure

```
rag_observatory/
├── src/              # Core modules
├── dashboards/       # Streamlit dashboard
├── tests/            # Test suite
├── screenshots/      # Dashboard screenshots
└── docs/            # Documentation
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Links

- **Repository**: https://github.com/namrathatiptur/rag-observatory
- **Issues**: https://github.com/namrathatiptur/rag-observatory/issues
