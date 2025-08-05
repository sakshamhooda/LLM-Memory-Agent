import logging
from typing import List, Dict, Any, Optional
from fact_extractor import FactExtractor
from memory_storage import MemoryStorage

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages memory operations including fact extraction and storage."""
    
    def __init__(self, fact_extractor: FactExtractor, memory_storage: MemoryStorage):
        """
        Initialize memory manager with fact extractor and memory storage.
        
        Args:
            fact_extractor: Service for extracting facts from messages
            memory_storage: Service for storing and retrieving memories
        """
        self.fact_extractor = fact_extractor
        self.memory_storage = memory_storage
        logger.info("Initialized MemoryManager")
    
    def process_message(self, user_id: str, message: str, action: str = "add", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user message to extract facts and perform memory operations.
        
        Args:
            user_id: ID of the user
            message: The user's message
            action: Action to perform ("add" or "delete")
            metadata: Optional metadata for the memories
            
        Returns:
            Dictionary with operation results
        """
        try:
            if action == "add":
                return self._add_memories_from_message(user_id, message, metadata)
            elif action == "delete":
                return self._delete_memories_from_message(user_id, message)
            else:
                raise ValueError(f"Invalid action: {action}. Must be 'add' or 'delete'")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise
    
    def _add_memories_from_message(self, user_id: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract facts from message and add them as memories.
        
        Args:
            user_id: ID of the user
            message: The user's message
            metadata: Optional metadata for the memories
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Extract facts from the message
            facts = self.fact_extractor.extract_facts(message)
            
            if not facts:
                logger.warning(f"No facts extracted from message: {message}")
                return {
                    "success": False,
                    "message": "No facts could be extracted from the message",
                    "facts": [],
                    "memory_ids": []
                }
            
            # Add memories for each fact
            memory_ids = []
            for fact in facts:
                memory_id = self.memory_storage.add_memory(user_id, fact, metadata)
                memory_ids.append(memory_id)
            
            logger.info(f"Added {len(facts)} memories for user {user_id}")
            
            return {
                "success": True,
                "message": f"Successfully added {len(facts)} memories",
                "facts": facts,
                "memory_ids": memory_ids
            }
            
        except Exception as e:
            logger.error(f"Error adding memories from message: {e}")
            raise
    
    def _delete_memories_from_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Extract deletion facts from message and delete corresponding memories.
        
        Args:
            user_id: ID of the user
            message: The user's message
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Extract facts to delete from the message
            deletion_facts = self.fact_extractor.extract_deletion_facts(message)
            
            if not deletion_facts:
                logger.warning(f"No deletion facts extracted from message: {message}")
                return {
                    "success": False,
                    "message": "No deletion facts could be extracted from the message",
                    "deletion_facts": [],
                    "deleted_count": 0
                }
            
            # Delete memories for each fact
            deleted_count = 0
            for fact in deletion_facts:
                if self.memory_storage.delete_memory(user_id, fact):
                    deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} memories for user {user_id}")
            
            return {
                "success": True,
                "message": f"Successfully deleted {deleted_count} memories",
                "deletion_facts": deletion_facts,
                "deleted_count": deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error deleting memories from message: {e}")
            raise
    
    def get_context(self, user_id: str, query: str, n: int = 5) -> str:
        """
        Get relevant memory context for a query.
        
        Args:
            user_id: ID of the user
            query: The query to get context for
            n: Number of memories to retrieve
            
        Returns:
            Formatted context string with relevant memories
        """
        try:
            memories = self.memory_storage.retrieve_memories(user_id, query, n)
            
            if not memories:
                return "No relevant memories found."
            
            context_lines = []
            for memory in memories:
                content = memory.get("content", "")
                created_at = memory.get("created_at", "")
                context_lines.append(f"- {content} (added: {created_at})")
            
            return "\n".join(context_lines)
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return "Error retrieving memories."
    
    def get_user_memory_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of a user's memories.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with memory summary
        """
        try:
            stats = self.memory_storage.get_memory_stats(user_id)
            recent_memories = self.memory_storage.get_user_memories(user_id, limit=10)
            
            return {
                "stats": stats,
                "recent_memories": recent_memories,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            raise
    
    def search_memories(self, user_id: str, query: str, n: int = 5) -> List[Dict[str, Any]]:
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
            return self.memory_storage.search_memories(user_id, query, n)
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
            return self.memory_storage.add_memory(user_id, fact, metadata)
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
            return self.memory_storage.delete_memory(user_id, fact)
        except Exception as e:
            logger.error(f"Error deleting memory directly: {e}")
            raise
