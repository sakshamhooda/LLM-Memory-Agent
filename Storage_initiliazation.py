#!/usr/bin/env python3
"""
Storage initialization script for the Memory Agent system.
This script sets up the databases and verifies the system is ready.
"""

import os
import sys
import logging
from config import Config
from chroma_store import ChromaVectorStore
from sqlite_store import SQLiteMetadataStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_storage():
    """Initialize all storage components."""
    try:
        print("🔧 Initializing Memory Agent Storage")
        print("=" * 50)
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(Config.DB_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(Config.SQL_PATH), exist_ok=True)
        
        print(f"📁 Database paths:")
        print(f"   ChromaDB: {Config.DB_PATH}")
        print(f"   SQLite: {Config.SQL_PATH}")
        
        # Initialize ChromaDB
        print("\n🔍 Initializing ChromaDB...")
        vector_store = ChromaVectorStore()
        collection_info = vector_store.get_collection_info()
        print(f"✅ ChromaDB initialized successfully")
        print(f"   Collection: {collection_info['name']}")
        print(f"   Memory count: {collection_info['count']}")
        
        # Initialize SQLite
        print("\n🗄️  Initializing SQLite...")
        metadata_store = SQLiteMetadataStore()
        print(f"✅ SQLite initialized successfully")
        
        # Test basic operations
        print("\n🧪 Testing basic operations...")
        
        # Test adding a sample memory
        test_memory_id = "test_init_memory"
        test_user_id = "test_user"
        test_content = "This is a test memory for initialization"
        
        metadata_store.add_memory(test_memory_id, test_user_id, test_content)
        print("✅ SQLite write test passed")
        
        # Test reading the memory
        memory = metadata_store.get_memory_by_id(test_memory_id)
        if memory and memory['content'] == test_content:
            print("✅ SQLite read test passed")
        else:
            print("❌ SQLite read test failed")
            return False
        
        # Test deleting the test memory
        metadata_store.delete_memory(test_memory_id)
        print("✅ SQLite delete test passed")
        
        # Test ChromaDB operations
        test_embedding = [0.1] * 1536  # 1536 dimensions for text-embedding-3-small
        vector_store.add_memory(test_memory_id, test_embedding, test_content)
        print("✅ ChromaDB write test passed")
        
        # Test ChromaDB query
        results = vector_store.query(test_embedding, n=1)
        if results and results.get('ids') and results['ids'][0]:
            print("✅ ChromaDB query test passed")
        else:
            print("❌ ChromaDB query test failed")
            return False
        
        # Clean up test data
        vector_store.delete_memory(test_memory_id)
        print("✅ ChromaDB delete test passed")
        
        print("\n🎉 Storage initialization completed successfully!")
        print("\nSystem Status:")
        print("✅ ChromaDB: Ready")
        print("✅ SQLite: Ready")
        print("✅ Basic operations: Working")
        
        return True
        
    except Exception as e:
        logger.error(f"Storage initialization failed: {e}")
        print(f"❌ Storage initialization failed: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    
    try:
        import openai
        print("✅ OpenAI: Available")
    except ImportError:
        missing_deps.append("openai")
        print("❌ OpenAI: Missing")
    
    try:
        import chromadb
        print("✅ ChromaDB: Available")
    except ImportError:
        missing_deps.append("chromadb")
        print("❌ ChromaDB: Missing")
    
    try:
        import sqlite3
        print("✅ SQLite3: Available")
    except ImportError:
        missing_deps.append("sqlite3")
        print("❌ SQLite3: Missing")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing dependencies:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    print("✅ All dependencies available")
    return True

def check_environment():
    """Check environment configuration."""
    print("\n🔧 Checking environment...")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OPENAI_API_KEY: Set")
    else:
        print("❌ OPENAI_API_KEY: Not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    # Check database paths
    db_dir = os.path.dirname(Config.DB_PATH)
    sql_dir = os.path.dirname(Config.SQL_PATH)
    
    if os.access(db_dir, os.W_OK):
        print(f"✅ ChromaDB directory writable: {db_dir}")
    else:
        print(f"❌ ChromaDB directory not writable: {db_dir}")
        return False
    
    if os.access(sql_dir, os.W_OK):
        print(f"✅ SQLite directory writable: {sql_dir}")
    else:
        print(f"❌ SQLite directory not writable: {sql_dir}")
        return False
    
    print("✅ Environment configuration valid")
    return True

def main():
    """Main initialization function."""
    print("🚀 Memory Agent Storage Initialization")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing dependencies.")
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please configure your environment.")
        sys.exit(1)
    
    # Initialize storage
    if initialize_storage():
        print("\n🎉 All systems ready!")
        print("\nNext steps:")
        print("1. Run the demo: python main.py")
        print("2. Try interactive mode: python main.py --interactive")
        print("3. Run tests: python test_memory_agent.py")
    else:
        print("\n❌ Storage initialization failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()