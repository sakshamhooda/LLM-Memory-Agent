#!/usr/bin/env python3
"""
Test script for the Memory Agent system.
This verifies that all components work together correctly.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch
import tempfile
import shutil

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import MemoryAgent
from config import Config
from fact_extractor import FactExtractor
from embedding_service import EmbeddingService
from chroma_store import ChromaVectorStore
from sqlite_store import SQLiteMetadataStore
from memory_storage import MemoryStorage
from memory_manager import MemoryManager

class TestMemoryAgent(unittest.TestCase):
    """Test cases for the Memory Agent system."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.original_db_path = Config.DB_PATH
        self.original_sql_path = Config.SQL_PATH
        
        # Update config for testing
        Config.DB_PATH = os.path.join(self.test_dir, "test_memory_db")
        Config.SQL_PATH = os.path.join(self.test_dir, "test_memories.db")
        
        # Mock OpenAI client
        self.mock_client = Mock()
        self.mock_response = Mock()
        self.mock_response.choices = [Mock()]
        self.mock_response.choices[0].message.content = '["User uses Shram", "User uses Magnet"]'
        self.mock_client.chat.completions.create.return_value = self.mock_response
        
        # Mock embedding response
        self.mock_embedding_response = Mock()
        self.mock_embedding_response.data = [Mock()]
        self.mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        self.mock_client.embeddings.create.return_value = self.mock_embedding_response
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original config
        Config.DB_PATH = self.original_db_path
        Config.SQL_PATH = self.original_sql_path
        
        # Remove test directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('app.OpenAI')
    def test_memory_agent_initialization(self, mock_openai):
        """Test that MemoryAgent initializes correctly."""
        mock_openai.return_value = self.mock_client
        
        # Mock environment variable
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = MemoryAgent()
            
            self.assertIsNotNone(agent)
            self.assertIsNotNone(agent.fact_extractor)
            self.assertIsNotNone(agent.embedding_service)
            self.assertIsNotNone(agent.memory_manager)
    
    @patch('app.OpenAI')
    def test_fact_extraction(self, mock_openai):
        """Test fact extraction functionality."""
        mock_openai.return_value = self.mock_client
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = MemoryAgent()
            
            # Test fact extraction
            facts = agent.fact_extractor.extract_facts("I use Shram and Magnet as productivity tools")
            
            self.assertIsInstance(facts, list)
            self.assertGreater(len(facts), 0)
    
    @patch('app.OpenAI')
    def test_memory_addition(self, mock_openai):
        """Test adding memories."""
        mock_openai.return_value = self.mock_client
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = MemoryAgent()
            
            # Test adding memory
            result = agent.add_memory("test_user", "I use Shram and Magnet as productivity tools")
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertIn('message', result)
    
    @patch('app.OpenAI')
    def test_memory_deletion(self, mock_openai):
        """Test deleting memories."""
        mock_openai.return_value = self.mock_client
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = MemoryAgent()
            
            # Test deleting memory
            result = agent.delete_memory("test_user", "I don't use Magnet anymore")
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertIn('message', result)
    
    @patch('app.OpenAI')
    def test_memory_query(self, mock_openai):
        """Test querying with memory context."""
        mock_openai.return_value = self.mock_client
        
        # Mock GPT response for query
        mock_query_response = Mock()
        mock_query_response.choices = [Mock()]
        mock_query_response.choices[0].message.content = "You use Shram as a productivity tool."
        self.mock_client.chat.completions.create.return_value = mock_query_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = MemoryAgent()
            
            # Test querying with memory
            response = agent.query_with_memory("test_user", "What productivity tools do I use?")
            
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
    
    @patch('app.OpenAI')
    def test_memory_summary(self, mock_openai):
        """Test getting memory summary."""
        mock_openai.return_value = self.mock_client
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = MemoryAgent()
            
            # Test getting memory summary
            summary = agent.get_memory_summary("test_user")
            
            self.assertIsInstance(summary, dict)
            self.assertIn('stats', summary)
            self.assertIn('recent_memories', summary)
    
    @patch('app.OpenAI')
    def test_memory_search(self, mock_openai):
        """Test searching memories."""
        mock_openai.return_value = self.mock_client
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            agent = MemoryAgent()
            
            # Test searching memories
            results = agent.search_memories("test_user", "productivity", n=5)
            
            self.assertIsInstance(results, list)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with missing API key
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                Config.validate()
        
        # Test with valid API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            self.assertTrue(Config.validate())

def run_basic_tests():
    """Run basic functionality tests without OpenAI API."""
    print("🧪 Running Basic Tests")
    print("=" * 40)
    
    # Test configuration
    print("✅ Testing configuration...")
    assert Config.EMBEDDING_MODEL == "text-embedding-3-small"
    assert Config.GPT_MODEL == "gpt-4-turbo"
    assert Config.COLLECTION_NAME == "user_memories"
    print("✅ Configuration tests passed")
    
    # Test imports
    print("✅ Testing imports...")
    try:
        from fact_extractor import FactExtractor
        from embedding_service import EmbeddingService
        from chroma_store import ChromaVectorStore
        from sqlite_store import SQLiteMetadataStore
        from memory_storage import MemoryStorage
        from memory_manager import MemoryManager
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test storage initialization
    print("✅ Testing storage initialization...")
    try:
        vector_store = ChromaVectorStore()
        metadata_store = SQLiteMetadataStore()
        print("✅ Storage initialization successful")
    except Exception as e:
        print(f"❌ Storage initialization error: {e}")
        return False
    
    print("✅ All basic tests passed!")
    return True

if __name__ == "__main__":
    print("🧪 Memory Agent Test Suite")
    print("=" * 50)
    
    # Run basic tests first
    if run_basic_tests():
        print("\n🚀 Running full test suite...")
        unittest.main(verbosity=2)
    else:
        print("\n❌ Basic tests failed. Please check your setup.")
        sys.exit(1) 