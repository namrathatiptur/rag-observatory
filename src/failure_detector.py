"""Failure Detector for RAG Observatory.

This module provides the FailureDetector class that identifies 4 types of failures:
1. Bad Retrieval - Low relevance or coverage
2. Potential Hallucination - Answer not grounded in context
3. Low Coverage - Query not well covered by retrieved chunks
4. Poor Grounding - Answer doesn't use the context effectively
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can be detected."""
    BAD_RETRIEVAL = "bad_retrieval"
    POTENTIAL_HALLUCINATION = "potential_hallucination"
    LOW_COVERAGE = "low_coverage"
    POOR_GROUNDING = "poor_grounding"


class Severity(Enum):
    """Severity levels for failures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Failure:
    """Represents a detected failure."""
    type: FailureType
    severity: Severity
    message: str
    metric_name: str
    metric_value: float
    threshold: float


class FailureDetector:
    """Detects failures in RAG system performance."""
    
    def __init__(
        self,
        strict_mode: bool = False,
        custom_thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize the failure detector.
        
        Args:
            strict_mode: If True, use stricter thresholds for detection
            custom_thresholds: Optional dict to override default thresholds
        """
        self.strict_mode = strict_mode
        
        # Default thresholds (normal mode)
        self.thresholds = {
            'min_avg_relevance': 0.5,
            'min_query_coverage': 0.4,
            'min_answer_grounding_rate': 0.5,
            'min_context_usage_rate': 0.3,
            'min_query_answer_similarity': 0.4,
            'min_retrieval_diversity': 0.2,
            'min_answer_specificity': 0.2,
        }
        
        # Stricter thresholds
        if strict_mode:
            self.thresholds = {
                'min_avg_relevance': 0.6,
                'min_query_coverage': 0.5,
                'min_answer_grounding_rate': 0.6,
                'min_context_usage_rate': 0.4,
                'min_query_answer_similarity': 0.5,
                'min_retrieval_diversity': 0.3,
                'min_answer_specificity': 0.3,
            }
        
        # Override with custom thresholds if provided
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)
    
    def _determine_severity(self, metric_value: float, threshold: float, metric_name: str) -> Severity:
        """Determine severity based on how far below threshold the metric is.
        
        Args:
            metric_value: The actual metric value
            threshold: The minimum acceptable threshold
            metric_name: Name of the metric (for context)
            
        Returns:
            Severity level
        """
        if metric_value >= threshold:
            return Severity.LOW  # Not actually a failure, but included for completeness
        
        gap = threshold - metric_value
        
        # Critical: metric is less than 50% of threshold
        if metric_value < threshold * 0.5:
            return Severity.CRITICAL
        
        # High: metric is 50-75% of threshold
        if metric_value < threshold * 0.75:
            return Severity.HIGH
        
        # Medium: metric is 75-90% of threshold
        if metric_value < threshold * 0.9:
            return Severity.MEDIUM
        
        # Low: metric is 90-100% of threshold (just barely below)
        return Severity.LOW
    
    def detect_failures(self, metrics: Dict[str, Any]) -> List[Failure]:
        """Detect failures based on metrics.
        
        Args:
            metrics: Dictionary of computed metrics from RAGMetricsCollector
            
        Returns:
            List of detected failures
        """
        failures = []
        
        # Validate metrics contain required fields
        required_fields = [
            'avg_relevance', 'query_coverage', 'answer_grounding_rate',
            'context_usage_rate', 'query_answer_similarity'
        ]
        
        for field in required_fields:
            if field not in metrics:
                logger.warning(f"Missing metric: {field}")
                continue
        
        # Failure 1: Bad Retrieval
        # Detected when relevance or coverage is low
        avg_relevance = metrics.get('avg_relevance', 0.0)
        query_coverage = metrics.get('query_coverage', 0.0)
        
        if avg_relevance < self.thresholds['min_avg_relevance']:
            severity = self._determine_severity(
                avg_relevance,
                self.thresholds['min_avg_relevance'],
                'avg_relevance'
            )
            failures.append(Failure(
                type=FailureType.BAD_RETRIEVAL,
                severity=severity,
                message=f"Low average relevance ({avg_relevance:.3f} < {self.thresholds['min_avg_relevance']:.3f})",
                metric_name='avg_relevance',
                metric_value=avg_relevance,
                threshold=self.thresholds['min_avg_relevance']
            ))
        
        if query_coverage < self.thresholds['min_query_coverage']:
            severity = self._determine_severity(
                query_coverage,
                self.thresholds['min_query_coverage'],
                'query_coverage'
            )
            failures.append(Failure(
                type=FailureType.LOW_COVERAGE,
                severity=severity,
                message=f"Low query coverage ({query_coverage:.3f} < {self.thresholds['min_query_coverage']:.3f})",
                metric_name='query_coverage',
                metric_value=query_coverage,
                threshold=self.thresholds['min_query_coverage']
            ))
        
        # Failure 2: Potential Hallucination
        # Detected when answer grounding rate is very low
        answer_grounding_rate = metrics.get('answer_grounding_rate', 0.0)
        context_usage_rate = metrics.get('context_usage_rate', 0.0)
        
        if answer_grounding_rate < self.thresholds['min_answer_grounding_rate']:
            severity = self._determine_severity(
                answer_grounding_rate,
                self.thresholds['min_answer_grounding_rate'],
                'answer_grounding_rate'
            )
            failures.append(Failure(
                type=FailureType.POTENTIAL_HALLUCINATION,
                severity=severity,
                message=f"Low answer grounding rate ({answer_grounding_rate:.3f} < {self.thresholds['min_answer_grounding_rate']:.3f}) - possible hallucination",
                metric_name='answer_grounding_rate',
                metric_value=answer_grounding_rate,
                threshold=self.thresholds['min_answer_grounding_rate']
            ))
        
        # Failure 3: Poor Grounding
        # Detected when context usage rate is low
        if context_usage_rate < self.thresholds['min_context_usage_rate']:
            severity = self._determine_severity(
                context_usage_rate,
                self.thresholds['min_context_usage_rate'],
                'context_usage_rate'
            )
            failures.append(Failure(
                type=FailureType.POOR_GROUNDING,
                severity=severity,
                message=f"Low context usage rate ({context_usage_rate:.3f} < {self.thresholds['min_context_usage_rate']:.3f}) - answer may not be using retrieved context",
                metric_name='context_usage_rate',
                metric_value=context_usage_rate,
                threshold=self.thresholds['min_context_usage_rate']
            ))
        
        # Additional check: Very low query-answer similarity might indicate off-topic answer
        query_answer_similarity = metrics.get('query_answer_similarity', 0.0)
        if query_answer_similarity < self.thresholds['min_query_answer_similarity']:
            # This could be either bad retrieval or hallucination
            # We'll classify it as poor grounding if context usage is also low
            if context_usage_rate < self.thresholds['min_context_usage_rate']:
                # Already added as poor grounding, skip duplicate
                pass
            else:
                # Answer doesn't match query well
                severity = self._determine_severity(
                    query_answer_similarity,
                    self.thresholds['min_query_answer_similarity'],
                    'query_answer_similarity'
                )
                failures.append(Failure(
                    type=FailureType.POOR_GROUNDING,
                    severity=severity,
                    message=f"Low query-answer similarity ({query_answer_similarity:.3f} < {self.thresholds['min_query_answer_similarity']:.3f}) - answer may not address the query",
                    metric_name='query_answer_similarity',
                    metric_value=query_answer_similarity,
                    threshold=self.thresholds['min_query_answer_similarity']
                ))
        
        return failures
    
    def get_failure_summary(self, failures: List[Failure]) -> Dict[str, Any]:
        """Get summary statistics for failures.
        
        Args:
            failures: List of detected failures
            
        Returns:
            Dictionary with failure counts by type and severity
        """
        summary = {
            'total_failures': len(failures),
            'by_type': {},
            'by_severity': {},
            'critical_count': 0,
            'high_count': 0,
            'medium_count': 0,
            'low_count': 0,
        }
        
        for failure in failures:
            # Count by type
            failure_type = failure.type.value
            summary['by_type'][failure_type] = summary['by_type'].get(failure_type, 0) + 1
            
            # Count by severity
            severity = failure.severity.value
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # Count by severity level
            if failure.severity == Severity.CRITICAL:
                summary['critical_count'] += 1
            elif failure.severity == Severity.HIGH:
                summary['high_count'] += 1
            elif failure.severity == Severity.MEDIUM:
                summary['medium_count'] += 1
            else:
                summary['low_count'] += 1
        
        return summary

