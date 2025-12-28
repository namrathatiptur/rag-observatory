"""Configuration settings for RAG Observatory."""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    
    # Embedding model
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    
    # LLM settings
    llm_model: str = "llama2"  # For Ollama
    llm_api_key: Optional[str] = None  # For OpenAI
    llm_base_url: Optional[str] = None  # For custom endpoints
    llm_temperature: float = 0.1
    
    # Retrieval settings
    top_k: int = 5
    similarity_top_k: int = 5
    
    # ChromaDB settings
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "rag_observatory"
    
    # Metrics thresholds
    min_avg_relevance: float = 0.5
    min_query_coverage: float = 0.4
    min_answer_grounding_rate: float = 0.5
    min_context_usage_rate: float = 0.3
    
    # Failure detection
    strict_mode: bool = False
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Load configuration from environment variables."""
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            llm_model=os.getenv("LLM_MODEL", "llama2"),
            llm_api_key=os.getenv("OPENAI_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            top_k=int(os.getenv("TOP_K", "5")),
            strict_mode=os.getenv("STRICT_MODE", "false").lower() == "true",
        )

