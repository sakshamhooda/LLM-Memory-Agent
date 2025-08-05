import logging
from typing import Dict, Any, Optional
from openai import OpenAI
from config import Config
from fact_extractor import FactExtractor
from embedding_service import EmbeddingService
from chroma_store import ChromaVectorStore
from sqlite_store import SQLiteMetadataStore
from memory_storage import MemoryStorage
from memory_manager import MemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryAgent:
    """Main memory agent that provides a high-level interface for memory operations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the memory agent with all required components.
        
        Args:
            api_key: OpenAI API key (if not provided, will use environment variable)
        """
        try:
            # Validate configuration
            Config.validate()
            
            # Initialize OpenAI client
            self.client = OpenAI(api_key=api_key or Config.OPENAI_API_KEY)
            
            # Initialize services
            self.fact_extractor = FactExtractor(self.client)
            self.embedding_service = EmbeddingService(self.client)
            
            # Initialize storage
            self.vector_store = ChromaVectorStore()
            self.metadata_store = SQLiteMetadataStore()
            
            # Initialize memory storage
            self.memory_storage = MemoryStorage(
                self.vector_store,
                self.metadata_store,
                self.embedding_service
            )
            
            # Initialize memory manager
            self.memory_manager = MemoryManager(
                self.fact_extractor,
                self.memory_storage
            )
            
            logger.info("Memory Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory Agent: {e}")
            raise
    
    def add_memory(self, user_id: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add memories from a user message.
        
        Args:
            user_id: ID of the user
            message: The user's message
            metadata: Optional metadata for the memories
            
        Returns:
            Dictionary with operation results
        """
        try:
            return self.memory_manager.process_message(user_id, message, "add", metadata)
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise
    
    def delete_memory(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Delete memories based on a user message.
        
        Args:
            user_id: ID of the user
            message: The user's message
            
        Returns:
            Dictionary with operation results
        """
        try:
            return self.memory_manager.process_message(user_id, message, "delete")
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise
    
    def query_with_memory(self, user_id: str, question: str, n_memories: int = 5) -> str:
        """
        Query with memory context using GPT.
        
        Args:
            user_id: ID of the user
            question: The user's question
            n_memories: Number of memories to include in context
            
        Returns:
            GPT response with memory context
        """
        try:
            # Get relevant memory context
            context = self.memory_manager.get_context(user_id, question, n_memories)
            
            # Create system message with memory context
            system_message = f"""You are a helpful assistant with access to the user's memories.

Relevant memories:
{context}

Use these memories to provide accurate and personalized responses. If no relevant memories are found, respond based on your general knowledge.

Always be helpful, accurate, and concise."""
            
            # Get response from GPT
            response = self.client.chat.completions.create(
                model=Config.GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error querying with memory: {e}")
            return f"Error: {str(e)}"
    
    def get_memory_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of a user's memories.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with memory summary
        """
        try:
            return self.memory_manager.get_user_memory_summary(user_id)
        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            raise
    
    def search_memories(self, user_id: str, query: str, n: int = 5) -> list:
        """
        Search for memories based on a query.
        
        Args:
            user_id: ID of the user
            query: Search query
            n: Number of results to return
            
        Returns:
            List of relevant memories
        """
        try:
            return self.memory_manager.search_memories(user_id, query, n)
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise
    
    def add_memory_directly(self, user_id: str, fact: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a memory directly without fact extraction.
        
        Args:
            user_id: ID of the user
            fact: The fact to store
            metadata: Optional metadata for the memory
            
        Returns:
            Memory ID of the created memory
        """
        try:
            return self.memory_manager.add_memory_directly(user_id, fact, metadata)
        except Exception as e:
            logger.error(f"Error adding memory directly: {e}")
            raise
    
    def delete_memory_directly(self, user_id: str, fact: str) -> bool:
        """
        Delete a memory directly without fact extraction.
        
        Args:
            user_id: ID of the user
            fact: The fact to delete
            
        Returns:
            True if memory was found and deleted, False otherwise
        """
        try:
            return self.memory_manager.delete_memory_directly(user_id, fact)
        except Exception as e:
            logger.error(f"Error deleting memory directly: {e}")
            raise
    
    def reset_user_memories(self, user_id: str) -> bool:
        """
        Reset all memories for a user (for testing purposes).
        
        Args:
            user_id: ID of the user
            
        Returns:
            True if successful
        """
        try:
            # Get all user memories
            memories = self.memory_storage.get_user_memories(user_id)
            
            # Delete each memory
            for memory in memories:
                memory_id = memory.get("id")
                if memory_id:
                    self.metadata_store.delete_memory(memory_id)
                    self.vector_store.delete_memory(memory_id)
            
            logger.info(f"Reset all memories for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting user memories: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and statistics.
        
        Returns:
            Dictionary with system information
        """
        try:
            collection_info = self.vector_store.get_collection_info()
            
            return {
                "vector_store": collection_info,
                "embedding_model": Config.EMBEDDING_MODEL,
                "gpt_model": Config.GPT_MODEL,
                "database_path": Config.DB_PATH,
                "sqlite_path": Config.SQL_PATH
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            raise
