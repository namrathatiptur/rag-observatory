"""Metrics Collector for RAG Observatory.

This module provides the RAGMetricsCollector class that computes 8 key metrics
for evaluating RAG system performance:
1. Average Relevance
2. Query Coverage
3. Answer Grounding Rate
4. Context Usage Rate
5. Query-Answer Similarity
6. Retrieval Diversity
7. Answer Specificity
8. Number of Chunks Retrieved
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RAGMetricsCollector:
    """Collects and computes metrics for RAG system evaluation."""
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        """Initialize the metrics collector.
        
        Args:
            embedding_model: HuggingFace model name for semantic similarity.
                            Defaults to a lightweight model for fast computation.
        """
        self.embedding_model_name = embedding_model
        self._model: Optional[SentenceTransformer] = None
        self.metrics_history: List[Dict[str, Any]] = []
        
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            embeddings = self.model.encode([text1, text2], normalize_embeddings=True)
            similarity = np.dot(embeddings[0], embeddings[1])
            # Clamp to [0, 1] range (cosine similarity can be slightly outside)
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            return 0.0
    
    def compute_all_metrics(
        self,
        query: str,
        retrieved_chunks: List[str],
        retrieval_scores: List[float],
        context: str,
        answer: str,
        query_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute all 8 metrics for a RAG query.
        
        Args:
            query: The user's query string
            retrieved_chunks: List of retrieved document chunks
            retrieval_scores: List of relevance scores for each chunk
            context: The combined context used for generation
            answer: The generated answer
            query_id: Optional identifier for this query
            
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {
            'query_id': query_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'num_chunks_retrieved': len(retrieved_chunks),
        }
        
        # Metric 1: Average Relevance
        if retrieval_scores and len(retrieval_scores) > 0:
            metrics['avg_relevance'] = float(np.mean(retrieval_scores))
        else:
            metrics['avg_relevance'] = 0.0
        
        # Metric 2: Query Coverage (how well chunks cover the query)
        if retrieved_chunks:
            chunk_similarities = [
                self._compute_similarity(query, chunk) 
                for chunk in retrieved_chunks
            ]
            # Use max similarity as coverage (best matching chunk)
            metrics['query_coverage'] = float(np.max(chunk_similarities)) if chunk_similarities else 0.0
        else:
            metrics['query_coverage'] = 0.0
        
        # Metric 3: Answer Grounding Rate (how well answer is grounded in context)
        if context and answer:
            metrics['answer_grounding_rate'] = self._compute_similarity(context, answer)
        else:
            metrics['answer_grounding_rate'] = 0.0
        
        # Metric 4: Context Usage Rate (how much of retrieved context is used)
        if retrieved_chunks and context:
            # Compute how similar context is to retrieved chunks
            context_chunk_similarities = [
                self._compute_similarity(context, chunk)
                for chunk in retrieved_chunks
            ]
            # Average similarity indicates usage rate
            metrics['context_usage_rate'] = float(np.mean(context_chunk_similarities)) if context_chunk_similarities else 0.0
        else:
            metrics['context_usage_rate'] = 0.0
        
        # Metric 5: Query-Answer Similarity
        if query and answer:
            metrics['query_answer_similarity'] = self._compute_similarity(query, answer)
        else:
            metrics['query_answer_similarity'] = 0.0
        
        # Metric 6: Retrieval Diversity (how diverse are the retrieved chunks)
        if len(retrieved_chunks) > 1:
            # Compute pairwise similarities between chunks
            chunk_embeddings = self.model.encode(retrieved_chunks, normalize_embeddings=True)
            pairwise_similarities = []
            for i in range(len(chunk_embeddings)):
                for j in range(i + 1, len(chunk_embeddings)):
                    sim = np.dot(chunk_embeddings[i], chunk_embeddings[j])
                    pairwise_similarities.append(sim)
            # Diversity = 1 - average similarity (higher diversity = lower similarity)
            avg_pairwise_sim = np.mean(pairwise_similarities) if pairwise_similarities else 0.0
            metrics['retrieval_diversity'] = float(1.0 - avg_pairwise_sim)
        else:
            metrics['retrieval_diversity'] = 0.0  # Single chunk = no diversity
        
        # Metric 7: Answer Specificity (how specific/detailed is the answer)
        # Measured as answer length relative to query (longer = more specific)
        if query and answer:
            query_length = len(query.split())
            answer_length = len(answer.split())
            if query_length > 0:
                # Normalize: specificity = min(1.0, answer_length / (query_length * 2))
                # This gives a score where 2x query length = 1.0
                metrics['answer_specificity'] = float(min(1.0, answer_length / (query_length * 2.0)))
            else:
                metrics['answer_specificity'] = 0.0
        else:
            metrics['answer_specificity'] = 0.0
        
        # Store in history
        self.metrics_history.append(metrics.copy())
        
        return metrics
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics history to CSV file.
        
        Args:
            filepath: Path to output CSV file
        """
        if not self.metrics_history:
            logger.warning("No metrics to export")
            return
        
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(df)} metrics to {filepath}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all collected metrics.
        
        Returns:
            Dictionary with mean, std, min, max for each numeric metric
        """
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        numeric_cols = [
            'avg_relevance', 'query_coverage', 'answer_grounding_rate',
            'context_usage_rate', 'query_answer_similarity', 'retrieval_diversity',
            'answer_specificity', 'num_chunks_retrieved'
        ]
        
        summary = {}
        for col in numeric_cols:
            if col in df.columns:
                summary[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return summary

