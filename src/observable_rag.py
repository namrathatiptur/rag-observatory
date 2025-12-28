"""Observable RAG System.

This module provides the ObservableRAG class that wraps LlamaIndex with
metrics collection and failure detection capabilities.
"""

from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.metrics_collector import RAGMetricsCollector
from src.failure_detector import FailureDetector, Failure
from src.config import RAGConfig

logger = logging.getLogger(__name__)


class ObservableRAG:
    """RAG system with built-in observability and failure detection."""
    
    def __init__(
        self,
        documents: List[Document],
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        llm_model: str = "llama2",
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        top_k: int = 5,
        strict_mode: bool = False,
        config: Optional[RAGConfig] = None
    ):
        """Initialize the Observable RAG system.
        
        Args:
            documents: List of Document objects to index
            embedding_model: HuggingFace model for embeddings
            llm_model: LLM model name (for Ollama) or "gpt-3.5-turbo" for OpenAI
            llm_api_key: API key for OpenAI (if using OpenAI)
            llm_base_url: Base URL for LLM API (for custom endpoints)
            top_k: Number of chunks to retrieve
            strict_mode: Use strict failure detection thresholds
            config: Optional RAGConfig object (overrides other params)
        """
        # Use config if provided, otherwise use parameters
        if config:
            self.config = config
            embedding_model = config.embedding_model
            llm_model = config.llm_model
            llm_api_key = config.llm_api_key
            llm_base_url = config.llm_base_url
            top_k = config.top_k
            strict_mode = config.strict_mode
        else:
            self.config = RAGConfig(
                embedding_model=embedding_model,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
                top_k=top_k,
                strict_mode=strict_mode
            )
        
        self.top_k = top_k
        self.strict_mode = strict_mode
        
        # Initialize components
        logger.info("Initializing embedding model...")
        self.embedding_model = HuggingFaceEmbedding(model_name=embedding_model)
        
        # Initialize LLM
        logger.info(f"Initializing LLM: {llm_model}")
        self._setup_llm(llm_model, llm_api_key, llm_base_url)
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        self._setup_vector_store()
        
        # Build index
        logger.info(f"Building index from {len(documents)} documents...")
        Settings.embed_model = self.embedding_model
        Settings.llm = self.llm
        
        self.index = VectorStoreIndex.from_documents(
            documents,
            vector_store=self.vector_store
        )
        
        # Initialize observability components
        self.metrics_collector = RAGMetricsCollector(embedding_model=embedding_model)
        self.failure_detector = FailureDetector(strict_mode=strict_mode)
        
        logger.info("ObservableRAG initialized successfully")
    
    def _setup_llm(self, llm_model: str, api_key: Optional[str], base_url: Optional[str]):
        """Setup the LLM based on model type."""
        # Force mock LLM for testing
        if llm_model == "mock_testing_only":
            logger.warning("Using Mock LLM for testing (no answer generation)")
            from llama_index.core.llms.mock import MockLLM
            self.llm = MockLLM()
            return
        
        try:
            # Check if Groq is requested (by checking if model contains groq)
            if llm_model.lower().startswith("groq"):
                # Use Groq
                try:
                    from llama_index.llms.groq import Groq
                    if not api_key:
                        raise ValueError("Groq API key required")
                    # Extract model name (remove 'groq/' prefix if present)
                    model_name = llm_model.replace("groq/", "").replace("groq-", "").strip()
                    if not model_name:
                        model_name = "llama3-8b-8192"
                    self.llm = Groq(
                        model=model_name,
                        api_key=api_key
                    )
                    logger.info(f"✅ Using Groq LLM: {model_name}")
                    return
                except ImportError:
                    # Try OpenAI-compatible interface for Groq
                    logger.info("Groq package not found, using OpenAI-compatible interface")
                    from llama_index.llms.openai import OpenAI
                    if not api_key:
                        raise ValueError("Groq API key required")
                    model_name = llm_model.replace("groq/", "").replace("groq-", "").strip()
                    if not model_name:
                        model_name = "llama3-8b-8192"
                    self.llm = OpenAI(
                        model=model_name,
                        api_key=api_key,
                        base_url="https://api.groq.com/openai/v1"
                    )
                    logger.info(f"✅ Using Groq LLM (via OpenAI interface): {model_name}")
                    return
                except Exception as e:
                    logger.error(f"Failed to initialize Groq: {e}")
                    raise
            
            if llm_model.startswith("gpt") or (api_key and not llm_model.startswith("groq")):
                # Use OpenAI
                from llama_index.llms.openai import OpenAI
                if not api_key:
                    raise ValueError("OpenAI API key required for GPT models")
                self.llm = OpenAI(
                    model=llm_model if llm_model.startswith("gpt") else "gpt-3.5-turbo",
                    api_key=api_key,
                    base_url=base_url
                )
                logger.info(f"✅ Using OpenAI LLM: {llm_model}")
            else:
                # Use Ollama (local)
                from llama_index.llms.ollama import Ollama
                self.llm = Ollama(
                    model=llm_model,
                    request_timeout=120.0
                )
                logger.info(f"✅ Using Ollama LLM: {llm_model}")
        except ImportError as e:
            logger.error(f"Failed to import LLM: {e}")
            logger.warning("Falling back to mock LLM for testing")
            # Create a simple mock LLM for testing
            from llama_index.core.llms.mock import MockLLM
            self.llm = MockLLM()
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            logger.warning("Falling back to mock LLM for testing")
            from llama_index.core.llms.mock import MockLLM
            self.llm = MockLLM()
    
    def _setup_vector_store(self):
        """Setup ChromaDB vector store."""
        try:
            # Create ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path=self.config.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            chroma_collection = chroma_client.get_or_create_collection(
                name=self.config.chroma_collection_name
            )
            
            # Create vector store
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB: {e}")
            raise
    
    def query(
        self,
        query_str: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Query the RAG system and collect metrics.
        
        Args:
            query_str: The user's query
            verbose: Whether to print progress messages
            
        Returns:
            Dictionary with 'answer', 'metrics', 'failures', 'num_failures'
        """
        if verbose:
            logger.info(f"Processing query: {query_str[:50]}...")
        
        # Create query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=self.top_k,
            verbose=verbose
        )
        
        # Execute query
        response = query_engine.query(query_str)
        answer = str(response)
        
        # Clean up answer - remove boilerplate
        if "Context information is below" in answer:
            # Try to extract just the answer part
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            elif "---------------------\n" in answer:
                parts = answer.split("---------------------\n")
                if len(parts) > 1:
                    answer = parts[-1].strip()
        
        # For very long answers, try to extract key sentence
        if len(answer) > 500 and "?" in query_str.lower():
            # Try to find a shorter, more direct answer
            sentences = answer.split('.')
            # Look for sentences that might directly answer the question
            query_words = set(query_str.lower().split())
            best_sentence = None
            best_score = 0
            
            for sent in sentences[:10]:  # Check first 10 sentences
                sent_lower = sent.lower()
                # Score based on query word matches and length
                score = sum(1 for word in query_words if word in sent_lower)
                if len(sent) < 200 and score > 0:  # Prefer shorter, relevant sentences
                    if score > best_score:
                        best_score = score
                        best_sentence = sent.strip()
            
            if best_sentence and len(best_sentence) < len(answer) * 0.5:
                answer = best_sentence + "."
        
        # Get retrieved nodes and scores
        retrieved_nodes = []
        retrieval_scores = []
        
        # Extract source nodes from response
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                retrieved_nodes.append(node.text)
                # Get similarity score if available
                score = getattr(node, 'score', 0.0)
                if score is None:
                    score = 0.0
                retrieval_scores.append(float(score))
        
        # If no source nodes, try to get from query engine
        if not retrieved_nodes:
            # Re-query to get retrieval results
            retriever = self.index.as_retriever(similarity_top_k=self.top_k)
            nodes = retriever.retrieve(query_str)
            for node in nodes:
                retrieved_nodes.append(node.text)
                score = getattr(node, 'score', 0.0)
                if score is None:
                    score = 0.0
                retrieval_scores.append(float(score))
        
        # Combine retrieved chunks into context
        context = " ".join(retrieved_nodes)
        
        # Compute metrics
        metrics = self.metrics_collector.compute_all_metrics(
            query=query_str,
            retrieved_chunks=retrieved_nodes,
            retrieval_scores=retrieval_scores,
            context=context,
            answer=answer
        )
        
        # Detect failures
        failures = self.failure_detector.detect_failures(metrics)
        
        if verbose:
            logger.info(f"Query completed. Failures detected: {len(failures)}")
        
        return {
            'answer': answer,
            'metrics': metrics,
            'failures': [self._failure_to_dict(f) for f in failures],
            'num_failures': len(failures),
            'retrieved_chunks': retrieved_nodes,
            'retrieval_scores': retrieval_scores
        }
    
    def batch_query(
        self,
        queries: List[str],
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple queries in batch.
        
        Args:
            queries: List of query strings
            verbose: Whether to print progress messages
            
        Returns:
            List of result dictionaries (same format as query())
        """
        results = []
        total = len(queries)
        
        for i, query_str in enumerate(queries, 1):
            if verbose:
                logger.info(f"Processing query {i}/{total}: {query_str[:50]}...")
            
            try:
                result = self.query(query_str, verbose=False)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query {i}: {e}")
                # Add error result
                results.append({
                    'answer': f"Error: {str(e)}",
                    'metrics': {},
                    'failures': [],
                    'num_failures': 0,
                    'error': str(e)
                })
        
        return results
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate a health report based on all collected metrics.
        
        Returns:
            Dictionary with health statistics
        """
        if not self.metrics_collector.metrics_history:
            return {
                'total_queries': 0,
                'message': 'No queries processed yet'
            }
        
        # Get all failures across history
        all_failures = []
        for metrics in self.metrics_collector.metrics_history:
            failures = self.failure_detector.detect_failures(metrics)
            all_failures.extend(failures)
        
        # Compute statistics
        total_queries = len(self.metrics_collector.metrics_history)
        total_failures = len(all_failures)
        failure_rate = total_failures / total_queries if total_queries > 0 else 0.0
        
        # Count by severity
        critical_failures = sum(1 for f in all_failures if f.severity.value == 'critical')
        critical_failure_rate = critical_failures / total_queries if total_queries > 0 else 0.0
        
        # Get summary stats
        summary_stats = self.metrics_collector.get_summary_stats()
        
        # Failure summary
        failure_summary = self.failure_detector.get_failure_summary(all_failures)
        
        return {
            'total_queries': total_queries,
            'total_failures': total_failures,
            'failure_rate': failure_rate,
            'critical_failures': critical_failures,
            'critical_failure_rate': critical_failure_rate,
            'failure_summary': failure_summary,
            'metrics_summary': summary_stats,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_data(self, filepath: str) -> None:
        """Export all collected metrics to CSV.
        
        Args:
            filepath: Path to output CSV file
        """
        self.metrics_collector.export_metrics(filepath)
        logger.info(f"Data exported to {filepath}")
    
    def _failure_to_dict(self, failure: Failure) -> Dict[str, Any]:
        """Convert Failure object to dictionary."""
        return {
            'type': failure.type.value,
            'severity': failure.severity.value,
            'message': failure.message,
            'metric_name': failure.metric_name,
            'metric_value': failure.metric_value,
            'threshold': failure.threshold
        }

