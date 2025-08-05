import logging
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
from storage import IMetadataStore
from config import Config

logger = logging.getLogger(__name__)

class SQLiteMetadataStore(IMetadataStore):
    """SQLite implementation of metadata storage for memories."""
    
    def __init__(self):
        """Initialize SQLite connection and create tables."""
        try:
            self.conn = sqlite3.connect(Config.SQL_PATH, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access
            self._initialize_db()
            logger.info(f"Initialized SQLite database: {Config.SQL_PATH}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            raise
    
    def _initialize_db(self):
        """Create the memories table if it doesn't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                is_deleted BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # Create indexes for better performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON memories(user_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_is_deleted ON memories(is_deleted)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)")
        
        self.conn.commit()
    
    def add_memory(self, memory_id: str, user_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a memory with metadata.
        
        Args:
            memory_id: Unique identifier for the memory
            user_id: ID of the user who owns this memory
            content: The memory content
            metadata: Optional metadata as dictionary
        """
        try:
            metadata_json = self._dict_to_json(metadata or {})
            
            self.conn.execute("""
                INSERT INTO memories (id, user_id, content, metadata)
                VALUES (?, ?, ?, ?)
            """, (memory_id, user_id, content, metadata_json))
            
            self.conn.commit()
            logger.debug(f"Added memory to SQLite: {memory_id}")
        except Exception as e:
            logger.error(f"Error adding memory to SQLite: {e}")
            raise
    
    def delete_memory(self, memory_id: str) -> None:
        """
        Soft delete a memory by marking it as deleted.
        
        Args:
            memory_id: ID of the memory to delete
        """
        try:
            self.conn.execute("""
                UPDATE memories 
                SET is_deleted = 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (memory_id,))
            
            self.conn.commit()
            logger.debug(f"Soft deleted memory in SQLite: {memory_id}")
        except Exception as e:
            logger.error(f"Error deleting memory from SQLite: {e}")
            raise
    
    def get_active_memories(self, user_id: str) -> List[str]:
        """
        Get all active (non-deleted) memory IDs for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of active memory IDs
        """
        try:
            cursor = self.conn.execute("""
                SELECT id FROM memories 
                WHERE user_id = ? AND is_deleted = 0
                ORDER BY created_at DESC
            """, (user_id,))
            
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting active memories: {e}")
            raise
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get memory details by ID.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Memory details as dictionary or None if not found
        """
        try:
            cursor = self.conn.execute("""
                SELECT * FROM memories WHERE id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        except Exception as e:
            logger.error(f"Error getting memory by ID: {e}")
            raise
    
    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> None:
        """
        Update memory metadata.
        
        Args:
            memory_id: ID of the memory to update
            updates: Dictionary of fields to update
        """
        try:
            # Build dynamic update query
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key in ['content', 'metadata']:
                    set_clauses.append(f"{key} = ?")
                    if key == 'metadata':
                        values.append(self._dict_to_json(value))
                    else:
                        values.append(value)
            
            if set_clauses:
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                values.append(memory_id)
                
                query = f"UPDATE memories SET {', '.join(set_clauses)} WHERE id = ?"
                self.conn.execute(query, values)
                self.conn.commit()
                
                logger.debug(f"Updated memory in SQLite: {memory_id}")
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            raise
    
    def get_user_memories(self, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all memories for a user with optional limit.
        
        Args:
            user_id: ID of the user
            limit: Optional limit on number of memories to return
            
        Returns:
            List of memory dictionaries
        """
        try:
            query = """
                SELECT * FROM memories 
                WHERE user_id = ? AND is_deleted = 0
                ORDER BY created_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor = self.conn.execute(query, (user_id,))
            return [dict(row) for row in cursor.fetchall()]
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
            cursor = self.conn.execute("""
                SELECT 
                    COUNT(*) as total_memories,
                    COUNT(CASE WHEN is_deleted = 0 THEN 1 END) as active_memories,
                    COUNT(CASE WHEN is_deleted = 1 THEN 1 END) as deleted_memories,
                    MIN(created_at) as first_memory,
                    MAX(created_at) as last_memory
                FROM memories 
                WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else {}
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            raise
    
    def _dict_to_json(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to JSON string."""
        import json
        return json.dumps(data)
    
    def _json_to_dict(self, json_str: str) -> Dict[str, Any]:
        """Convert JSON string to dictionary."""
        import json
        try:
            return json.loads(json_str)
        except:
            return {}
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed SQLite connection")
    
    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()
