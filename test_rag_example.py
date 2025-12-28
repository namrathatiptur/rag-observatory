"""Example: How to test RAG Observatory with your own documents and queries.

This script shows you how to:
1. Create documents from your data (CSV, text files, etc.)
2. Build a RAG system with observability
3. Run queries and see metrics
4. Export results for dashboard visualization
"""

from src.observable_rag import ObservableRAG
from llama_index.core import Document
import pandas as pd

print("=" * 70)
print("RAG Observatory - Testing Example")
print("=" * 70)

# ============================================================================
# STEP 1: Prepare Your Documents
# ============================================================================
print("\n📄 Step 1: Preparing Documents...")

# Option A: Create documents from text
documents = [
    Document(text="Python is a high-level programming language known for its simplicity and readability."),
    Document(text="Machine learning is a method of data analysis that automates analytical model building."),
    Document(text="RAG (Retrieval-Augmented Generation) combines retrieval and generation for better answers."),
    Document(text="Natural language processing enables computers to understand human language."),
    Document(text="Deep learning uses neural networks with multiple layers to learn complex patterns."),
    Document(text="Computer vision allows machines to interpret and understand visual information."),
    Document(text="Reinforcement learning involves agents learning through trial and error."),
    Document(text="Data science combines statistics, programming, and domain expertise."),
]

# Option B: Load from CSV (if you have text data in CSV)
# df = pd.read_csv("your_data.csv")
# documents = [Document(text=row['text_column']) for _, row in df.iterrows()]

print(f"✅ Created {len(documents)} documents")

# ============================================================================
# STEP 2: Initialize RAG System with Observability
# ============================================================================
print("\n🔧 Step 2: Initializing RAG System...")

try:
    rag = ObservableRAG(
        documents=documents,
        embedding_model="BAAI/bge-small-en-v1.5",  # Lightweight model
        llm_model="llama2",  # Will use mock if Ollama not available
        top_k=3,  # Retrieve top 3 most relevant documents
        strict_mode=False
    )
    print("✅ RAG system initialized with observability")
except Exception as e:
    print(f"⚠️  Error: {e}")
    print("   (This is OK - system will use mock LLM for testing)")

# ============================================================================
# STEP 3: Run Queries and See Metrics
# ============================================================================
print("\n🔍 Step 3: Running Test Queries...")
print("-" * 70)

test_queries = [
    "What is Python?",  # Should work well - direct match
    "Explain machine learning",  # Should work well
    "What is quantum computing?",  # Should fail - not in documents
    "Tell me about AI",  # Should partially work
    "How does RAG work?",  # Should work well
]

results = []

for i, query in enumerate(test_queries, 1):
    print(f"\nQuery {i}: {query}")
    print("-" * 70)
    
    try:
        result = rag.query(query, verbose=False)
        
        # Display results
        print(f"Answer: {result['answer'][:100]}...")
        print(f"\n📊 Metrics:")
        print(f"  • Avg Relevance:     {result['metrics']['avg_relevance']:.3f}")
        print(f"  • Query Coverage:    {result['metrics']['query_coverage']:.3f}")
        print(f"  • Answer Grounding:   {result['metrics']['answer_grounding_rate']:.3f}")
        print(f"  • Context Usage:      {result['metrics']['context_usage_rate']:.3f}")
        print(f"  • Chunks Retrieved:   {result['metrics']['num_chunks_retrieved']}")
        
        # Check for failures
        if result['num_failures'] > 0:
            print(f"\n⚠️  Failures Detected: {result['num_failures']}")
            for failure in result['failures'][:2]:  # Show first 2
                print(f"  - {failure['type'].upper()} ({failure['severity']}): {failure['message']}")
        else:
            print(f"\n✅ No failures detected!")
        
        results.append({
            'query': query,
            **result['metrics'],
            'num_failures': result['num_failures'],
            'answer_length': len(result['answer'])
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================================================
# STEP 4: View Health Report
# ============================================================================
print("\n" + "=" * 70)
print("📈 Step 4: System Health Report")
print("=" * 70)

health_report = rag.get_health_report()

print(f"\nTotal Queries Processed: {health_report['total_queries']}")
print(f"Total Failures:          {health_report['total_failures']}")
print(f"Failure Rate:            {health_report['failure_rate']:.1%}")
print(f"Critical Failures:       {health_report['critical_failures']}")
print(f"Critical Failure Rate:   {health_report['critical_failure_rate']:.1%}")

if 'metrics_summary' in health_report:
    print(f"\n📊 Average Metrics:")
    for metric, stats in list(health_report['metrics_summary'].items())[:5]:
        print(f"  • {metric}: {stats['mean']:.3f} (std: {stats['std']:.3f})")

# ============================================================================
# STEP 5: Export Metrics for Dashboard
# ============================================================================
print("\n" + "=" * 70)
print("💾 Step 5: Exporting Metrics")
print("=" * 70)

output_file = "rag_metrics_export.csv"
rag.export_data(output_file)
print(f"✅ Metrics exported to: {output_file}")
print(f"   You can now upload this file to the dashboard!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("✅ Testing Complete!")
print("=" * 70)
print("\nNext Steps:")
print("1. View the exported CSV: cat rag_metrics_export.csv")
print("2. Launch dashboard: cd dashboards && streamlit run streamlit_app.py")
print("3. Upload rag_metrics_export.csv to see visualizations")
print("4. Try with your own documents and queries!")

