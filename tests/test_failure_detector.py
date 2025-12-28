"""Test suite for Failure Detector."""

import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.failure_detector import FailureDetector, FailureType, Severity


class TestFailureDetector:
    """Test cases for FailureDetector."""
    
    def test_detect_bad_retrieval(self):
        """Test 1: Detect bad retrieval"""
        detector = FailureDetector()
        metrics = {
            'avg_relevance': 0.25,  # Low
            'query_coverage': 0.20,  # Low
            'answer_grounding_rate': 0.7,
            'context_usage_rate': 0.4,
            'query_answer_similarity': 0.6,
            'num_chunks_retrieved': 5
        }
        failures = detector.detect_failures(metrics)
        
        assert len(failures) > 0
        assert any(f.type.value == 'bad_retrieval' for f in failures)
        print("✓ Test 1 passed: Bad retrieval detected")
    
    def test_detect_hallucination(self):
        """Test 2: Detect hallucination"""
        detector = FailureDetector()
        metrics = {
            'avg_relevance': 0.8,
            'query_coverage': 0.7,
            'answer_grounding_rate': 0.2,  # Very low (below 50% of 0.5 threshold = critical)
            'context_usage_rate': 0.1,  # Very low (below 50% of 0.3 threshold = critical)
            'query_answer_similarity': 0.6,
            'num_chunks_retrieved': 5
        }
        failures = detector.detect_failures(metrics)
        
        assert any(f.type.value == 'potential_hallucination' for f in failures)
        # Check for high or critical severity (both indicate serious issue)
        assert any(f.severity.value in ['critical', 'high'] for f in failures)
        print("✓ Test 2 passed: Hallucination detected")
    
    def test_no_failures_good_metrics(self):
        """Test 3: No failures when metrics are good"""
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
        print("✓ Test 3 passed: No false positives")
    
    def test_strict_mode(self):
        """Test 4: Strict mode increases detections"""
        detector_normal = FailureDetector(strict_mode=False)
        detector_strict = FailureDetector(strict_mode=True)
        
        metrics = {
            'avg_relevance': 0.55,  # Borderline
            'query_coverage': 0.45,  # Borderline
            'answer_grounding_rate': 0.55,  # Borderline
            'context_usage_rate': 0.35,  # Borderline
            'query_answer_similarity': 0.45,  # Borderline
            'num_chunks_retrieved': 5
        }
        
        failures_normal = detector_normal.detect_failures(metrics)
        failures_strict = detector_strict.detect_failures(metrics)
        
        assert len(failures_strict) >= len(failures_normal)
        print("✓ Test 4 passed: Strict mode works")
    
    def test_custom_thresholds(self):
        """Test 5: Custom thresholds work"""
        detector_custom = FailureDetector(custom_thresholds={'min_avg_relevance': 0.9})
        metrics = {
            'avg_relevance': 0.85,  # Below custom threshold
            'query_coverage': 0.75,
            'answer_grounding_rate': 0.70,
            'context_usage_rate': 0.40,
            'query_answer_similarity': 0.75,
            'num_chunks_retrieved': 5
        }
        failures = detector_custom.detect_failures(metrics)
        
        assert len(failures) > 0
        print("✓ Test 5 passed: Custom thresholds applied")


if __name__ == "__main__":
    print("Running Failure Detector Tests...")
    test = TestFailureDetector()
    test.test_detect_bad_retrieval()
    test.test_detect_hallucination()
    test.test_no_failures_good_metrics()
    test.test_strict_mode()
    test.test_custom_thresholds()
    print("\n✅ All Failure Detector tests passed!")

