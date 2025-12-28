"""Quick demo script to show RAG Observatory results."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.observable_rag import ObservableRAG
from src.metrics_collector import RAGMetricsCollector
from src.failure_detector import FailureDetector
from llama_index.core import Document

print("=" * 70)
print("RAG Observatory - Demo Results")
print("=" * 70)

# Demo 1: Metrics Collector
print("\n" + "=" * 70)
print("DEMO 1: Metrics Collector")
print("=" * 70)

collector = RAGMetricsCollector()
metrics = collector.compute_all_metrics(
    query="What is artificial intelligence?",
    retrieved_chunks=[
        "Artificial intelligence (AI) is the simulation of human intelligence by machines.",
        "AI systems can learn, reason, and make decisions.",
        "Machine learning is a subset of AI that enables systems to learn from data."
    ],
    retrieval_scores=[0.92, 0.85, 0.78],
    context="Artificial intelligence (AI) is the simulation of human intelligence by machines. AI systems can learn, reason, and make decisions.",
    answer="Artificial intelligence, or AI, refers to the simulation of human intelligence in machines. These systems can learn from data, reason through problems, and make autonomous decisions."
)

print("\nComputed Metrics:")
print(f"  Average Relevance:        {metrics['avg_relevance']:.3f}")
print(f"  Query Coverage:           {metrics['query_coverage']:.3f}")
print(f"  Answer Grounding Rate:    {metrics['answer_grounding_rate']:.3f}")
print(f"  Context Usage Rate:       {metrics['context_usage_rate']:.3f}")
print(f"  Query-Answer Similarity:   {metrics['query_answer_similarity']:.3f}")
print(f"  Retrieval Diversity:      {metrics['retrieval_diversity']:.3f}")
print(f"  Answer Specificity:       {metrics['answer_specificity']:.3f}")
print(f"  Chunks Retrieved:         {metrics['num_chunks_retrieved']}")

# Demo 2: Failure Detector
print("\n" + "=" * 70)
print("DEMO 2: Failure Detector")
print("=" * 70)

detector = FailureDetector()

# Good metrics
print("\nTest Case 1: Good Metrics (should have no failures)")
good_metrics = {
    'avg_relevance': 0.85,
    'query_coverage': 0.75,
    'answer_grounding_rate': 0.70,
    'context_usage_rate': 0.40,
    'query_answer_similarity': 0.75,
    'num_chunks_retrieved': 5
}
failures = detector.detect_failures(good_metrics)
print(f"  Failures detected: {len(failures)}")
if failures:
    for f in failures:
        print(f"    - {f.type.value} ({f.severity.value}): {f.message}")
else:
    print("  ✅ No failures - system performing well!")

# Bad retrieval
print("\nTest Case 2: Bad Retrieval (should detect failures)")
bad_metrics = {
    'avg_relevance': 0.25,
    'query_coverage': 0.20,
    'answer_grounding_rate': 0.7,
    'context_usage_rate': 0.4,
    'query_answer_similarity': 0.6,
    'num_chunks_retrieved': 5
}
failures = detector.detect_failures(bad_metrics)
print(f"  Failures detected: {len(failures)}")
for f in failures:
    print(f"    - {f.type.value.upper()} ({f.severity.value.upper()}): {f.message}")

# Potential hallucination
print("\nTest Case 3: Potential Hallucination (should detect critical failure)")
hallucination_metrics = {
    'avg_relevance': 0.8,
    'query_coverage': 0.7,
    'answer_grounding_rate': 0.2,  # Very low
    'context_usage_rate': 0.1,  # Very low
    'query_answer_similarity': 0.6,
    'num_chunks_retrieved': 5
}
failures = detector.detect_failures(hallucination_metrics)
print(f"  Failures detected: {len(failures)}")
for f in failures:
    print(f"    - {f.type.value.upper()} ({f.severity.value.upper()}): {f.message}")

# Demo 3: Observable RAG
print("\n" + "=" * 70)
print("DEMO 3: Observable RAG System")
print("=" * 70)

print("\nInitializing RAG system with sample documents...")
documents = [
    Document(text="Python is a high-level programming language known for its simplicity."),
    Document(text="Machine learning is a method of data analysis that automates analytical model building."),
    Document(text="RAG (Retrieval-Augmented Generation) combines retrieval and generation for better answers."),
    Document(text="Natural language processing enables computers to understand human language."),
]

try:
    rag = ObservableRAG(
        documents=documents,
        top_k=2,
        strict_mode=False
    )
    print("✅ RAG system initialized")
    
    # Query 1: Good query
    print("\nQuery 1: 'What is Python?'")
    result1 = rag.query("What is Python?", verbose=False)
    print(f"  Answer: {result1['answer'][:100]}...")
    print(f"  Avg Relevance: {result1['metrics']['avg_relevance']:.3f}")
    print(f"  Query Coverage: {result1['metrics']['query_coverage']:.3f}")
    print(f"  Failures: {result1['num_failures']}")
    
    # Query 2: Out of domain
    print("\nQuery 2: 'What is quantum physics?' (out of domain)")
    result2 = rag.query("What is quantum physics?", verbose=False)
    print(f"  Answer: {result2['answer'][:100]}...")
    print(f"  Avg Relevance: {result2['metrics']['avg_relevance']:.3f}")
    print(f"  Query Coverage: {result2['metrics']['query_coverage']:.3f}")
    print(f"  Failures: {result2['num_failures']}")
    if result2['failures']:
        for f in result2['failures'][:2]:  # Show first 2
            print(f"    - {f['type']} ({f['severity']})")
    
    # Health report
    print("\n" + "-" * 70)
    print("Health Report:")
    report = rag.get_health_report()
    print(f"  Total Queries: {report['total_queries']}")
    print(f"  Total Failures: {report['total_failures']}")
    print(f"  Failure Rate: {report['failure_rate']:.1%}")
    print(f"  Critical Failures: {report['critical_failures']}")
    print(f"  Critical Failure Rate: {report['critical_failure_rate']:.1%}")
    
    # Metrics summary
    if 'metrics_summary' in report:
        print("\n  Metrics Summary (averages):")
        for metric, stats in list(report['metrics_summary'].items())[:4]:
            print(f"    {metric}: {stats['mean']:.3f} (std: {stats['std']:.3f})")
    
except Exception as e:
    print(f"⚠️  RAG system demo failed: {e}")
    print("   (This is OK if Ollama is not available - system uses mock LLM)")

print("\n" + "=" * 70)
print("✅ Demo Complete!")
print("=" * 70)
print("\nNext steps:")
print("  1. Launch dashboard: cd dashboards && streamlit run streamlit_app.py")
print("  2. Run full tests: python tests/test_all.py")
print("  3. Check README.md for more examples")

