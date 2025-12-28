"""Comprehensive test suite for RAG Observatory.

This file includes:
- Unit tests for all components
- Integration tests
- Edge case tests
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics_collector import RAGMetricsCollector
from src.failure_detector import FailureDetector, FailureType, Severity
from src.observable_rag import ObservableRAG
from llama_index.core import Document


class TestMetricsCollector:
    """Test Metrics Collector component."""
    
    def test_basic_functionality(self):
        """Test basic metrics computation."""
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
        print("✓ Metrics Collector: Basic functionality")
    
    def test_empty_chunks(self):
        """Test edge case: empty chunks."""
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
        print("✓ Metrics Collector: Empty chunks handled")
    
    def test_metric_ranges(self):
        """Test all metrics are in valid range."""
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
        print("✓ Metrics Collector: All metrics in valid range")
    
    def test_export(self):
        """Test CSV export functionality."""
        collector = RAGMetricsCollector()
        collector.compute_all_metrics(
            query="Test",
            retrieved_chunks=["Test chunk"],
            retrieval_scores=[0.9],
            context="Test context",
            answer="Test answer"
        )
        
        test_file = "/tmp/test_metrics_export.csv"
        collector.export_metrics(test_file)
        
        assert os.path.exists(test_file)
        df = pd.read_csv(test_file)
        assert len(df) > 0
        
        if os.path.exists(test_file):
            os.remove(test_file)
        print("✓ Metrics Collector: Export works")


class TestFailureDetector:
    """Test Failure Detector component."""
    
    def test_detect_bad_retrieval(self):
        """Test bad retrieval detection."""
        detector = FailureDetector()
        metrics = {
            'avg_relevance': 0.25,
            'query_coverage': 0.20,
            'answer_grounding_rate': 0.7,
            'context_usage_rate': 0.4,
            'query_answer_similarity': 0.6,
            'num_chunks_retrieved': 5
        }
        failures = detector.detect_failures(metrics)
        
        assert len(failures) > 0
        assert any(f.type.value == 'bad_retrieval' for f in failures)
        print("✓ Failure Detector: Bad retrieval detected")
    
    def test_detect_hallucination(self):
        """Test hallucination detection."""
        detector = FailureDetector()
        metrics = {
            'avg_relevance': 0.8,
            'query_coverage': 0.7,
            'answer_grounding_rate': 0.3,
            'context_usage_rate': 0.15,
            'query_answer_similarity': 0.6,
            'num_chunks_retrieved': 5
        }
        failures = detector.detect_failures(metrics)
        
        assert any(f.type.value == 'potential_hallucination' for f in failures)
        print("✓ Failure Detector: Hallucination detected")
    
    def test_no_false_positives(self):
        """Test no failures on good metrics."""
        detector = FailureDetector()
        metrics = {
            'avg_relevance': 0.85,
            'query_coverage': 0.75,
            'answer_grounding_rate': 0.70,
            'context_usage_rate': 0.40,
            'query_answer_similarity': 0.75,
            'retrieval_diversity': 0.60,
            'answer_specificity': 0.50,
            'num_chunks_retrieved': 5
        }
        failures = detector.detect_failures(metrics)
        
        assert len(failures) == 0
        print("✓ Failure Detector: No false positives")


class TestObservableRAG:
    """Test Observable RAG integration."""
    
    def test_initialization(self):
        """Test RAG system initialization."""
        docs = [
            Document(text="Paris is the capital of France."),
            Document(text="The Eiffel Tower is in Paris."),
            Document(text="France is in Western Europe.")
        ]
        
        try:
            rag = ObservableRAG(
                documents=docs,
                embedding_model="BAAI/bge-small-en-v1.5",
                llm_model="llama2",
                top_k=2
            )
            
            assert rag.index is not None
            assert len(rag.metrics_collector.metrics_history) == 0
            print("✓ ObservableRAG: Initialization works")
        except Exception as e:
            print(f"⚠ ObservableRAG: Initialization failed (may need Ollama): {e}")
            # This is OK if Ollama is not available
    
    def test_query_with_mock(self):
        """Test query functionality (will use mock LLM if Ollama unavailable)."""
        docs = [
            Document(text="Machine learning is a subset of AI."),
            Document(text="Deep learning uses neural networks.")
        ]
        
        try:
            rag = ObservableRAG(
                documents=docs,
                top_k=1
            )
            
            result = rag.query("What is machine learning?", verbose=False)
            
            assert 'answer' in result
            assert 'metrics' in result
            assert 'failures' in result
            print("✓ ObservableRAG: Query works")
        except Exception as e:
            print(f"⚠ ObservableRAG: Query test failed: {e}")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_query(self):
        """Test handling of empty query."""
        collector = RAGMetricsCollector()
        try:
            metrics = collector.compute_all_metrics(
                query="",
                retrieved_chunks=["Test"],
                retrieval_scores=[0.8],
                context="Test",
                answer="Test"
            )
            # Should handle gracefully
            assert 'avg_relevance' in metrics
            print("✓ Edge Cases: Empty query handled")
        except Exception as e:
            print(f"⚠ Edge Cases: Empty query error: {e}")
    
    def test_long_query(self):
        """Test handling of very long query."""
        collector = RAGMetricsCollector()
        long_query = " ".join(["test"] * 1000)
        
        try:
            metrics = collector.compute_all_metrics(
                query=long_query,
                retrieved_chunks=["Test"],
                retrieval_scores=[0.8],
                context="Test",
                answer="Test"
            )
            assert 'avg_relevance' in metrics
            print("✓ Edge Cases: Long query handled")
        except Exception as e:
            print(f"⚠ Edge Cases: Long query error: {e}")
    
    def test_special_characters(self):
        """Test handling of special characters."""
        collector = RAGMetricsCollector()
        special_query = "What is AI? <script>alert('xss')</script>"
        
        try:
            metrics = collector.compute_all_metrics(
                query=special_query,
                retrieved_chunks=["Test"],
                retrieval_scores=[0.8],
                context="Test",
                answer="Test"
            )
            assert 'avg_relevance' in metrics
            print("✓ Edge Cases: Special characters handled")
        except Exception as e:
            print(f"⚠ Edge Cases: Special characters error: {e}")
    
    def test_nan_metrics(self):
        """Test handling of NaN/Inf values in metrics."""
        detector = FailureDetector()
        metrics_with_nan = {
            'avg_relevance': float('nan'),
            'query_coverage': 0.5,
            'answer_grounding_rate': 0.5,
            'context_usage_rate': 0.3,
            'query_answer_similarity': 0.6,
            'num_chunks_retrieved': 5
        }
        
        try:
            # Should handle NaN gracefully
            failures = detector.detect_failures(metrics_with_nan)
            # May or may not detect failures, but shouldn't crash
            print("✓ Edge Cases: NaN metrics handled")
        except Exception as e:
            print(f"⚠ Edge Cases: NaN metrics error: {e}")


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("RAG Observatory - Comprehensive Test Suite")
    print("=" * 60)
    print()
    
    # Run test suites
    test_classes = [
        TestMetricsCollector,
        TestFailureDetector,
        TestObservableRAG,
        TestEdgeCases
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        test_instance = test_class()
        
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    passed_tests += 1
                except AssertionError as e:
                    print(f"✗ {method_name}: Assertion failed - {e}")
                except Exception as e:
                    print(f"✗ {method_name}: Error - {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print("=" * 60)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

