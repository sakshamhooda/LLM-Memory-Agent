import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from storage import IVectorStore
from config import Config

logger = logging.getLogger(__name__)

class ChromaVectorStore(IVectorStore):
    """ChromaDB implementation of vector storage for memories."""
    
    def __init__(self):
        """Initialize ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(
                path=Config.DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            self.collection = self.client.get_or_create_collection(
                name=Config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Initialized ChromaDB collection: {Config.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_memory(self, memory_id: str, embedding: List[float], content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a memory with its embedding to the vector store.
        
        Args:
            memory_id: Unique identifier for the memory
            embedding: Vector embedding of the memory content
            content: The actual memory content
            metadata: Optional metadata for the memory
        """
        try:
            # Prepare metadata
            memory_metadata = metadata or {}
            memory_metadata["memory_id"] = memory_id
            
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[memory_metadata]
            )
            logger.debug(f"Added memory to ChromaDB: {memory_id}")
        except Exception as e:
            logger.error(f"Error adding memory to ChromaDB: {e}")
            raise
    
    def query(self, embedding: List[float], filter_dict: Optional[Dict[str, Any]] = None, n: int = 5) -> Dict[str, Any]:
        """
        Query the vector store for similar memories.
        
        Args:
            embedding: Query embedding to search for
            filter_dict: Optional filter conditions
            n: Number of results to return
            
        Returns:
            Dictionary containing query results with keys: ids, distances, documents, metadatas
        """
        try:
            # Convert filter_dict to ChromaDB format if needed
            where_clause = None
            if filter_dict:
                where_clause = self._convert_filter_to_chroma_format(filter_dict)
            
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            logger.debug(f"ChromaDB query returned {len(results.get('ids', [[]])[0])} results")
            return results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            raise
    
    def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory from the vector store.
        
        Args:
            memory_id: ID of the memory to delete
        """
        try:
            self.collection.delete(ids=[memory_id])
            logger.debug(f"Deleted memory from ChromaDB: {memory_id}")
        except Exception as e:
            logger.error(f"Error deleting memory from ChromaDB: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            count = self.collection.count()
            return {
                "name": self.collection.name,
                "count": count,
                "metadata": self.collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise
    
    def _convert_filter_to_chroma_format(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert our filter format to ChromaDB's where clause format.
        
        Args:
            filter_dict: Filter dictionary in our format
            
        Returns:
            ChromaDB where clause format
        """
        if not filter_dict:
            return None
        
        where_clause = {}
        
        for key, value in filter_dict.items():
            if isinstance(value, dict) and "$in" in value:
                # Handle $in operator
                where_clause[key] = {"$in": value["$in"]}
            elif isinstance(value, dict) and "$eq" in value:
                # Handle $eq operator
                where_clause[key] = {"$eq": value["$eq"]}
            else:
                # Default to equality
                where_clause[key] = {"$eq": value}
        
        return where_clause
    
    def reset_collection(self) -> None:
        """Reset the collection (useful for testing)."""
        try:
            self.client.delete_collection(Config.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=Config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Reset ChromaDB collection")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise
