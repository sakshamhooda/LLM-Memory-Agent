import logging
import uuid
from typing import List, Dict, Any, Optional
from storage import IVectorStore, IMetadataStore
from embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class MemoryStorage:
    """Coordinates memory operations between vector store and metadata store."""
    
    def __init__(
        self, 
        vector_store: IVectorStore, 
        metadata_store: IMetadataStore,
        embedding_service: EmbeddingService
    ):
        """
        Initialize memory storage with vector store, metadata store, and embedding service.
        
        Args:
            vector_store: Vector store for embeddings
            metadata_store: Metadata store for memory information
            embedding_service: Service for generating embeddings
        """
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.embedding_service = embedding_service
        logger.info("Initialized MemoryStorage")
    
    def add_memory(self, user_id: str, fact: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a memory for a user.
        
        Args:
            user_id: ID of the user
            fact: The fact to store
            metadata: Optional metadata for the memory
            
        Returns:
            Memory ID of the created memory
        """
        try:
            memory_id = str(uuid.uuid4())
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(fact)
            
            # Store in metadata store
            self.metadata_store.add_memory(memory_id, user_id, fact, metadata)
            
            # Store in vector store
            self.vector_store.add_memory(memory_id, embedding, fact, metadata)
            
            logger.info(f"Added memory for user {user_id}: {fact[:50]}...")
            return memory_id
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise
    
    def add_memories_batch(self, user_id: str, facts: List[str], metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add multiple memories for a user in batch.
        
        Args:
            user_id: ID of the user
            facts: List of facts to store
            metadata: Optional metadata for the memories
            
        Returns:
            List of memory IDs
        """
        try:
            if not facts:
                return []
            
            # Generate embeddings in batch
            embeddings = self.embedding_service.generate_embeddings_batch(facts)
            
            memory_ids = []
            for i, (fact, embedding) in enumerate(zip(facts, embeddings)):
                memory_id = str(uuid.uuid4())
                memory_ids.append(memory_id)
                
                # Store in metadata store
                self.metadata_store.add_memory(memory_id, user_id, fact, metadata)
                
                # Store in vector store
                self.vector_store.add_memory(memory_id, embedding, fact, metadata)
            
            logger.info(f"Added {len(facts)} memories for user {user_id}")
            return memory_ids
            
        except Exception as e:
            logger.error(f"Error adding memories in batch: {e}")
            raise
    
    def delete_memory(self, user_id: str, fact: str) -> bool:
        """
        Delete a memory for a user by finding similar memories.
        
        Args:
            user_id: ID of the user
            fact: The fact to delete
            
        Returns:
            True if memory was found and deleted, False otherwise
        """
        try:
            # Generate embedding for the fact to delete
            embedding = self.embedding_service.generate_embedding(fact)
            
            # Get active memory IDs for the user
            active_ids = self.metadata_store.get_active_memories(user_id)
            
            if not active_ids:
                logger.debug(f"No active memories found for user {user_id}")
                return False
            
            # Search for similar memories
            results = self.vector_store.query(
                embedding=embedding,
                filter_dict={"id": {"$in": active_ids}},
                n=1
            )
            
            if results and results.get("ids") and results["ids"][0]:
                memory_id = results["ids"][0][0]
                
                # Soft delete in metadata store
                self.metadata_store.delete_memory(memory_id)
                
                # Hard delete from vector store
                self.vector_store.delete_memory(memory_id)
                
                logger.info(f"Deleted memory for user {user_id}: {fact[:50]}...")
                return True
            else:
                logger.debug(f"No similar memory found for deletion: {fact}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise
    
    def retrieve_memories(self, user_id: str, query: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for a user based on a query.
        
        Args:
            user_id: ID of the user
            query: Query to search for relevant memories
            n: Number of memories to retrieve
            
        Returns:
            List of memory dictionaries with content and metadata
        """
        try:
            # Get active memory IDs for the user
            active_ids = self.metadata_store.get_active_memories(user_id)
            
            if not active_ids:
                logger.debug(f"No active memories found for user {user_id}")
                return []
            
            # Generate embedding for the query
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Search for similar memories
            results = self.vector_store.query(
                embedding=query_embedding,
                filter_dict={"id": {"$in": active_ids}},
                n=n
            )
            
            if not results or not results.get("ids") or not results["ids"][0]:
                logger.debug(f"No relevant memories found for query: {query}")
                return []
            
            # Get full memory details from metadata store
            memories = []
            for memory_id in results["ids"][0]:
                memory_details = self.metadata_store.get_memory_by_id(memory_id)
                if memory_details and not memory_details.get("is_deleted"):
                    memories.append(memory_details)
            
            logger.debug(f"Retrieved {len(memories)} memories for user {user_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            raise
    
    def get_user_memories(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all memories for a user.
        
        Args:
            user_id: ID of the user
            limit: Optional limit on number of memories
            
        Returns:
            List of memory dictionaries
        """
        try:
            return self.metadata_store.get_user_memories(user_id, limit)
        except Exception as e:
            logger.error(f"Error getting user memories: {e}")
            raise
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about a user's memories.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with memory statistics
        """
        try:
            return self.metadata_store.get_memory_stats(user_id)
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            raise
    
    def search_memories(self, user_id: str, query: str, n: int = 5, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Search memories with similarity threshold.
        
        Args:
            user_id: ID of the user
            query: Search query
            n: Number of results to return
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of memory dictionaries with similarity scores
        """
        try:
            memories = self.retrieve_memories(user_id, query, n)
            
            # Filter by threshold if needed
            if threshold < 1.0:
                # Note: This would require distance calculation
                # For now, return all results
                pass
            
            return memories
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise
