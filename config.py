import os
from typing import Optional

class Config:
    """Configuration class for the memory system."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    GPT_MODEL: str = "gpt-4-turbo"
    GPT_FACT_EXTRACTION_MODEL: str = "gpt-3.5-turbo"
    
    # Database Configuration
    DB_PATH: str = os.getenv("DB_PATH", "memory_db")
    SQL_PATH: str = os.getenv("SQL_PATH", "memories.db")
    
    # Memory Configuration
    DEFAULT_MEMORY_RETRIEVAL_COUNT: int = 5
    SIMILARITY_THRESHOLD: float = 0.8
    
    # Collection Configuration
    COLLECTION_NAME: str = "user_memories"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return True
