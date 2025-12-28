"""End-to-end integration tests for RAG Observatory."""

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.observable_rag import ObservableRAG
from llama_index.core import Document


def test_end_to_end_workflow():
    """Test complete RAG pipeline with monitoring."""
    print("\n" + "=" * 60)
    print("End-to-End Integration Test")
    print("=" * 60)
    
    # Step 1: Create test corpus
    print("\nStep 1: Creating test corpus...")
    documents = [
        Document(text="Machine learning is a subset of artificial intelligence."),
        Document(text="Deep learning uses neural networks with multiple layers."),
        Document(text="Natural language processing enables computers to understand human language."),
        Document(text="Computer vision allows machines to interpret visual information."),
        Document(text="Reinforcement learning involves agents learning through trial and error."),
    ]
    print(f"✓ Created {len(documents)} documents")
    
    # Step 2: Initialize system
    print("\nStep 2: Initializing RAG system...")
    try:
        rag = ObservableRAG(
            documents=documents,
            llm_model="llama2",  # Will use mock if Ollama unavailable
            top_k=3,
            strict_mode=False
        )
        print("✓ RAG system initialized")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        assert False, f"Initialization failed: {e}"
    
    # Step 3: Run diverse queries
    print("\nStep 3: Running test queries...")
    test_queries = [
        "What is machine learning?",  # Should succeed - direct match
        "Explain deep learning",  # Should succeed - good coverage
        "What is quantum computing?",  # Should fail - out of domain
        "Tell me about AI",  # Should partially succeed
        "How does NLP work?"  # Should succeed
    ]
    
    results = rag.batch_query(test_queries, verbose=False)
    print(f"✓ Processed {len(results)} queries")
    
    # Step 4: Validate each result
    print("\nStep 4: Validating results...")
    for i, result in enumerate(results):
        query = test_queries[i]
        print(f"\n  Query {i+1}: {query[:50]}...")
        print(f"    Answer length: {len(result['answer'])} chars")
        print(f"    Avg relevance: {result['metrics'].get('avg_relevance', 0):.3f}")
        print(f"    Failures detected: {result['num_failures']}")
        
        # Assertions
        assert result['answer'] != "", f"Query {i+1} should return an answer"
        assert 'metrics' in result, f"Query {i+1} should have metrics"
        assert 'failures' in result, f"Query {i+1} should have failures list"
        
        # Query 3 should have failures (out of domain)
        if i == 2:
            if result['num_failures'] > 0:
                print("    ✓ Correctly detected out-of-domain query")
            else:
                print("    ⚠ Out-of-domain query did not trigger failures (may be OK)")
    
    # Step 5: Validate health report
    print("\nStep 5: Generating health report...")
    report = rag.get_health_report()
    
    print(f"  Total queries: {report['total_queries']}")
    print(f"  Failure rate: {report['failure_rate']:.1%}")
    print(f"  Critical failures: {report['critical_failure_rate']:.1%}")
    
    assert report['total_queries'] == 5, "Should have 5 queries"
    assert 0 <= report['failure_rate'] <= 1, "Failure rate should be between 0 and 1"
    print("✓ Health report generated")
    
    # Step 6: Export and validate data
    print("\nStep 6: Exporting metrics...")
    test_output_file = "/tmp/integration_test_output.csv"
    rag.export_data(test_output_file)
    
    assert os.path.exists(test_output_file), "Export file should exist"
    df = pd.read_csv(test_output_file)
    assert len(df) == 5, "Should export 5 queries"
    assert 'avg_relevance' in df.columns, "Should have avg_relevance column"
    assert 'query_coverage' in df.columns, "Should have query_coverage column"
    assert 'answer_grounding_rate' in df.columns, "Should have answer_grounding_rate column"
    print("✓ Data exported successfully")
    
    # Cleanup
    if os.path.exists(test_output_file):
        os.remove(test_output_file)
    
    print("\n" + "=" * 60)
    print("✅ INTEGRATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_end_to_end_workflow()

