# How Vector Embeddings Work in RAG Observatory

## Overview

Yes! When you upload documents, they are automatically converted into **vector embeddings** (numerical representations) that enable semantic search.

## The Process

### Step 1: Document Upload
```
Your Documents (Text)
    ↓
Upload to Dashboard
```

### Step 2: Document Processing
```
Documents → ObservableRAG
    ↓
Convert to LlamaIndex Document objects
```

### Step 3: Embedding Generation
```
Documents → Embedding Model → Vector Embeddings
```

This happens when you click **"Initialize RAG System"**

### Step 4: Vector Storage
```
Vector Embeddings → ChromaDB (Vector Database)
    ↓
Stored for fast similarity search
```

### Step 5: Query Processing
```
User Query → Embedding Model → Query Embedding
    ↓
Similarity Search in ChromaDB
    ↓
Retrieve Most Similar Documents
```

## Technical Details

### What Are Embeddings?

Embeddings are **numerical vectors** (arrays of numbers) that represent the semantic meaning of text.

**Example:**
- Document: "Python is a programming language"
- Embedding: `[0.23, -0.45, 0.67, ..., 0.12]` (typically 384 or 768 dimensions)

### How Similarity Works

1. **Documents** → Converted to embeddings → Stored in vector database
2. **Query** → Converted to embedding → Compared with document embeddings
3. **Similarity** = Cosine similarity between vectors
4. **Top K** = Most similar documents retrieved

### The Code Flow

```python
# 1. Initialize embedding model
embedding_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"  # 384-dimensional embeddings
)

# 2. Build index (converts documents to embeddings)
index = VectorStoreIndex.from_documents(
    documents,  # Your uploaded documents
    vector_store=chroma_vector_store  # Stores embeddings
)

# 3. When querying:
# - Query is embedded
# - Similarity search finds closest document embeddings
# - Top K documents retrieved
```

## Embedding Model Used

**Default Model:** `BAAI/bge-small-en-v1.5`
- **Dimensions:** 384
- **Type:** Sentence transformer
- **Purpose:** Semantic similarity

You can change this in the code:
```python
rag = ObservableRAG(
    documents=documents,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # Alternative
)
```

## Vector Database: ChromaDB

- **Purpose:** Stores document embeddings
- **Location:** `./chroma_db/` directory
- **Persistence:** Embeddings are saved to disk
- **Speed:** Fast similarity search (milliseconds)

## What Happens When You Query?

1. **Query Text** → "What is Python?"
2. **Query Embedding** → `[0.15, -0.32, 0.58, ...]`
3. **Similarity Search** → Compare with all document embeddings
4. **Top K Results** → Documents with highest similarity scores
5. **Retrieval Scores** → Used for metrics (avg_relevance)

## Metrics That Use Embeddings

Several metrics rely on embeddings:

1. **Avg Relevance** - Similarity scores from embedding search
2. **Query Coverage** - Embedding similarity between query and chunks
3. **Answer Grounding Rate** - Embedding similarity between context and answer
4. **Context Usage Rate** - Embedding similarity between context and chunks
5. **Query-Answer Similarity** - Embedding similarity between query and answer
6. **Retrieval Diversity** - Embedding similarity between retrieved chunks

## Performance Considerations

### Embedding Generation Time
- **Small dataset** (< 100 docs): ~1-5 seconds
- **Medium dataset** (100-1000 docs): ~5-30 seconds
- **Large dataset** (> 1000 docs): ~30+ seconds

### Storage
- Each embedding: ~1.5 KB (384 dimensions × 4 bytes)
- 1000 documents: ~1.5 MB
- 10,000 documents: ~15 MB

### Query Speed
- **Embedding query**: ~10-50ms
- **Similarity search**: ~5-20ms
- **Total query time**: ~100-500ms (including LLM)

## Customization

### Change Embedding Model

```python
# In dashboard or code:
rag = ObservableRAG(
    documents=documents,
    embedding_model="sentence-transformers/all-mpnet-base-v2"  # Larger, more accurate
)
```

### Popular Models

- **BAAI/bge-small-en-v1.5** (default) - Fast, 384 dims
- **sentence-transformers/all-MiniLM-L6-v2** - Fast, 384 dims
- **sentence-transformers/all-mpnet-base-v2** - Accurate, 768 dims
- **BAAI/bge-large-en-v1.5** - Very accurate, 1024 dims (slower)

## Summary

✅ **Yes, documents are converted to embeddings**
- Happens automatically when you initialize RAG system
- Stored in ChromaDB vector database
- Used for semantic similarity search
- Powers all the metrics and retrieval

The entire process is **automatic** - you just upload documents and query!

