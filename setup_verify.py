"""Setup verification script for RAG Observatory.

This script verifies that all dependencies are installed and components work correctly.
"""

import sys
import os

def check_import(module_name, package_name=None):
    """Check if a module can be imported.
    
    Args:
        module_name: Name of the module to import
        package_name: Display name (defaults to module_name)
    
    Returns:
        True if import successful, False otherwise
    """
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name}: {e}")
        return False


def check_component(component_name, test_func):
    """Check if a component works.
    
    Args:
        component_name: Name of the component
        test_func: Function that tests the component
    
    Returns:
        True if test passes, False otherwise
    """
    try:
        test_func()
        print(f"✓ {component_name}")
        return True
    except Exception as e:
        print(f"✗ {component_name}: {e}")
        return False


def main():
    """Run setup verification."""
    print("=" * 60)
    print("RAG Observatory - Setup Verification")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Check Python version
    print("Python Version:")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (>= 3.10 required)")
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (>= 3.10 required)")
        all_checks_passed = False
    print()
    
    # Check core dependencies
    print("Core Dependencies:")
    dependencies = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sentence_transformers", "sentence-transformers"),
        ("chromadb", "chromadb"),
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
    ]
    
    for module, package in dependencies:
        if not check_import(module, package):
            all_checks_passed = False
    print()
    
    # Check LlamaIndex dependencies
    print("LlamaIndex Dependencies:")
    llama_deps = [
        ("llama_index.core", "llama-index"),
        ("llama_index.vector_stores.chroma", "llama-index-vector-stores-chroma"),
        ("llama_index.embeddings.huggingface", "llama-index-embeddings-huggingface"),
    ]
    
    for module, package in llama_deps:
        if not check_import(module, package):
            all_checks_passed = False
    print()
    
    # Check project components
    print("Project Components:")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    def test_metrics_collector():
        from src.metrics_collector import RAGMetricsCollector
        collector = RAGMetricsCollector()
        assert collector is not None
    
    def test_failure_detector():
        from src.failure_detector import FailureDetector
        detector = FailureDetector()
        assert detector is not None
    
    def test_observable_rag():
        from src.observable_rag import ObservableRAG
        # Just check import, don't initialize (requires documents)
        assert ObservableRAG is not None
    
    def test_config():
        from src.config import RAGConfig
        config = RAGConfig()
        assert config is not None
    
    components = [
        ("Metrics Collector", test_metrics_collector),
        ("Failure Detector", test_failure_detector),
        ("Observable RAG", test_observable_rag),
        ("Config", test_config),
    ]
    
    for name, test_func in components:
        if not check_component(name, test_func):
            all_checks_passed = False
    print()
    
    # Check file structure
    print("File Structure:")
    required_files = [
        "src/metrics_collector.py",
        "src/failure_detector.py",
        "src/observable_rag.py",
        "src/config.py",
        "dashboards/streamlit_app.py",
        "requirements.txt",
    ]
    
    for filepath in required_files:
        full_path = os.path.join(os.path.dirname(__file__), filepath)
        if os.path.exists(full_path):
            print(f"✓ {filepath}")
        else:
            print(f"✗ {filepath} (missing)")
            all_checks_passed = False
    print()
    
    # Final result
    print("=" * 60)
    if all_checks_passed:
        print("✅ SUCCESS! RAG Observatory is ready to use")
        print()
        print("Next steps:")
        print("  1. Run tests: python tests/test_all.py")
        print("  2. Start dashboard: cd dashboards && streamlit run streamlit_app.py")
        print("  3. Try a query: python -c \"from src.observable_rag import ObservableRAG; ...\"")
        return 0
    else:
        print("❌ Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

