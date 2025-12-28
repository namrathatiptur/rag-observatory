"""Test suite for Metrics Collector."""

import pytest
import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics_collector import RAGMetricsCollector


class TestMetricsCollector:
    """Test cases for RAGMetricsCollector."""
    
    def test_basic_functionality(self):
        """Test 1: Basic functionality"""
        collector = RAGMetricsCollector()
        metrics = collector.compute_all_metrics(
            query="What is AI?",
            retrieved_chunks=["AI is artificial intelligence.", "AI can learn from data."],
            retrieval_scores=[0.95, 0.78],
            context="AI is artificial intelligence. AI can learn from data.",
            answer="AI stands for artificial intelligence and involves machine learning."
        )
        
        assert 'avg_relevance' in metrics
        assert 0 <= metrics['avg_relevance'] <= 1
        assert metrics['num_chunks_retrieved'] == 2
        print("✓ Test 1 passed: Basic functionality")
    
    def test_empty_chunks(self):
        """Test 2: Edge case - empty chunks"""
        collector = RAGMetricsCollector()
        metrics = collector.compute_all_metrics(
            query="Test",
            retrieved_chunks=[],
            retrieval_scores=[],
            context="",
            answer="No answer"
        )
        
        assert metrics['avg_relevance'] == 0.0
        assert metrics['num_chunks_retrieved'] == 0
        assert metrics['query_coverage'] == 0.0
        print("✓ Test 2 passed: Empty chunks handled")
    
    def test_metric_ranges(self):
        """Test 3: Metric ranges"""
        collector = RAGMetricsCollector()
        metrics = collector.compute_all_metrics(
            query="What is AI?",
            retrieved_chunks=["AI is artificial intelligence."],
            retrieval_scores=[0.8],
            context="AI is artificial intelligence.",
            answer="AI is artificial intelligence."
        )
        
        assert 0 <= metrics['query_coverage'] <= 1
        assert 0 <= metrics['answer_grounding_rate'] <= 1
        assert 0 <= metrics['retrieval_diversity'] <= 1
        assert 0 <= metrics['answer_specificity'] <= 1
        print("✓ Test 3 passed: All metrics in valid range")
    
    def test_export_functionality(self):
        """Test 4: Export functionality"""
        collector = RAGMetricsCollector()
        collector.compute_all_metrics(
            query="Test query",
            retrieved_chunks=["Test chunk"],
            retrieval_scores=[0.9],
            context="Test context",
            answer="Test answer"
        )
        
        test_file = "test_metrics_export.csv"
        collector.export_metrics(test_file)
        
        assert os.path.exists(test_file)
        df = pd.read_csv(test_file)
        assert len(df) > 0
        assert 'avg_relevance' in df.columns
        
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        
        print("✓ Test 4 passed: Export functionality works")


if __name__ == "__main__":
    print("Running Metrics Collector Tests...")
    test = TestMetricsCollector()
    test.test_basic_functionality()
    test.test_empty_chunks()
    test.test_metric_ranges()
    test.test_export_functionality()
    print("\n✅ All Metrics Collector tests passed!")

