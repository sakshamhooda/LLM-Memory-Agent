## Task

We need to create a memory storage and retrieval agent which works with OpenAl APIs to help
give GPT long term memory. This system must record memory (efficiently) for messages in a
conversation (whichever ones applicable), for example, &quot;a memory&quot; must be created when a
user says, &quot;I use Shram and Magnet as productivity tools&quot;. Later, when the user asks, &quot;What are
the productivity tools that I use&quot; in a new conversation, GPT should be able to answer, &quot;You use
Shram and Magnet&quot;. Just like addition, this system should also be able to delete memories
when the user says things like &quot;I don&#39;t use Magnet anymore”.

## Tentative approach

To implement a long-term memory system for GPT using OpenAI APIs, follow this structured approach:

### Core Components
1. **Memory Storage**  
   Use **ChromaDB** (vector database) for efficient similarity searches and **SQLite** for transactional operations (deletion/updates).

2. **Memory Processing**  
   Leverage **OpenAI's Embedding API** for vectorization and **Chat Completion API** for fact extraction.

---

### Step-by-Step Implementation

#### 1. **Memory Storage Setup**
```python
import chromadb
import sqlite3
import uuid
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

# Initialize ChromaDB (vector store)
chroma_client = chromadb.PersistentClient(path="memory_db")
collection = chroma_client.get_or_create_collection("user_memories")

# Initialize SQLite (for metadata/operations)
sql_conn = sqlite3.connect("memories.db")
sql_conn.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        content TEXT,
        is_deleted BOOLEAN DEFAULT 0
    )
""")
```

#### 2. **Memory Creation**
- **Fact Extraction**: Use GPT to split user messages into atomic facts.
- **Vectorization**: Store embeddings in ChromaDB + metadata in SQLite.

```python
def extract_facts(message: str) -> list:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract atomic facts as a JSON array. Example: ['User uses X', ...]"},
            {"role": "user", "content": f"Extract facts: '{message}'"}
        ]
    )
    return json.loads(response.choices[0].message.content)

def add_memory(user_id: str, message: str):
    facts = extract_facts(message)
    for fact in facts:
        mem_id = str(uuid.uuid4())
        
        # Store in SQLite
        sql_conn.execute(
            "INSERT INTO memories (id, user_id, content) VALUES (?, ?, ?)",
            (mem_id, user_id, fact)
        )
        
        # Generate embedding and store in ChromaDB
        embedding = client.embeddings.create(input=fact, model="text-embedding-3-small").data[0].embedding
        collection.add(ids=[mem_id], embeddings=[embedding], documents=[fact])
```

#### 3. **Memory Deletion**
- **Fact Extraction**: Extract the fact to delete (e.g., "use Magnet").
- **Similarity Search**: Find matching memories in ChromaDB.
- **Soft Delete**: Mark as deleted in SQLite.

```python
def delete_memory(user_id: str, message: str):
    fact_to_delete = extract_facts(message)[0]  # e.g., "User uses Magnet"
    
    # Find similar memories
    query_embedding = client.embeddings.create(
        input=fact_to_delete, model="text-embedding-3-small"
    ).data[0].embedding
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        where={"user_id": user_id}
    )
    
    if results["ids"][0]:
        mem_id = results["ids"][0][0]
        # Soft-delete in SQLite
        sql_conn.execute(
            "UPDATE memories SET is_deleted=1 WHERE id=?", (mem_id,)
        )
```

#### 4. **Memory Retrieval**
- **Search Active Memories**: Fetch relevant facts from ChromaDB, filtered by SQLite's `is_deleted=0`.

```python
def retrieve_memories(user_id: str, query: str, n=3) -> list:
    query_embedding = client.embeddings.create(
        input=query, model="text-embedding-3-small"
    ).data[0].embedding
    
    # Get IDs of non-deleted memories
    active_ids = [row[0] for row in sql_conn.execute(
        "SELECT id FROM memories WHERE user_id=? AND is_deleted=0", (user_id,)
    )]
    
    # Query ChromaDB with filtered IDs
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where={"id": {"$in": active_ids}}  # Filter active memories
    )
    return results["documents"][0]
```

#### 5. **Integrate with GPT**
Inject retrieved memories into the prompt:

```python
def ask_gpt(user_id: str, question: str):
    memories = retrieve_memories(user_id, question)
    memory_context = "\n".join([f"- {mem}" for mem in memories])
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": f"Relevant memories:\n{memory_context}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content
```

---

### Key Optimizations
1. **Atomic Facts**:  
   Storing small facts (e.g., "User uses Shram") instead of sentences enables precise updates.

2. **Hybrid Storage**:  
   - **ChromaDB**: Handles high-speed similarity searches.  
   - **SQLite**: Manages deletions/updates transactionally.  

3. **Soft Deletes**:  
   Avoid re-indexing by flagging deleted memories instead of removing them physically.

4. **Embedding Efficiency**:  
   Use `text-embedding-3-small` (fast/cheap) for memory operations and GPT-4 only for final responses.

---

### Example Workflow
```python
# User 1: Adds memory
add_memory("user123", "I use Shram and Magnet as productivity tools")

# User 2: Deletes memory
delete_memory("user123", "I don't use Magnet anymore")

# Query
response = ask_gpt("user123", "What productivity tools do I use?")
print(response)  # "You use Shram."
```

---

### Scalability Considerations
- **Scale to Millions**: Replace SQLite with PostgreSQL and ChromaDB with Pinecone.
- **Cost Control**: Cache embeddings, batch operations, and use cheaper models for embeddings.
- **Concurrency**: Add connection pooling for database operations.

This architecture balances accuracy, efficiency, and simplicity while leveraging OpenAI's APIs for core memory operations.

Here's a modular implementation with a low-level design (LLD) for a long-term memory system with OpenAI integration:

### LLD Overview
```
┌──────────────────────┐       ┌───────────────────┐
│   MemoryManager      │───────│  FactExtractor    │
└──────────────────────┘       └───────────────────┘
         │                             │
         ├─────────────────────────────┤
         │                             │
┌──────────────────────┐       ┌───────────────────┐
│   MemoryStorage      │───────│  EmbeddingService │
└──────────────────────┘       └───────────────────┘
         │
         ├──────────────────────┐
         │                      │
┌──────────────────┐    ┌──────────────────┐
│  VectorStore     │    │  MetadataStore   │
│  (ChromaDB)      │    │  (SQLite)        │
└──────────────────┘    └──────────────────┘
```

### Modular Implementation

**1. Config Module (`config.py`)**
```python
import os

class Config:
    EMBEDDING_MODEL = "text-embedding-3-small"
    GPT_MODEL = "gpt-4-turbo"
    DB_PATH = os.getenv("DB_PATH", "memory_db")
    SQL_PATH = os.getenv("SQL_PATH", "memories.db")
```

**2. Fact Extraction Module (`fact_extractor.py`)**
```python
from openai import OpenAI
import json

class FactExtractor:
    def __init__(self, client: OpenAI):
        self.client = client
    
    def extract_facts(self, text: str) -> list[str]:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Extract atomic facts as JSON array. Example: ['User uses X']"
                },
                {"role": "user", "content": f"Extract facts: '{text}'"}
            ]
        )
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return [text]  # Fallback to original text
```

**3. Embedding Service (`embedding_service.py`)**
```python
from openai import OpenAI
from config import Config

class EmbeddingService:
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = Config.EMBEDDING_MODEL
    
    def generate_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding
```

**4. Storage Interfaces (`storage.py`)**
```python
from abc import ABC, abstractmethod

class IVectorStore(ABC):
    @abstractmethod
    def add_memory(self, memory_id: str, embedding: list[float], content: str):
        pass
    
    @abstractmethod
    def query(self, embedding: list[float], filter: dict, n: int) -> list:
        pass

class IMetadataStore(ABC):
    @abstractmethod
    def add_memory(self, memory_id: str, user_id: str, content: str):
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str):
        pass
    
    @abstractmethod
    def get_active_memories(self, user_id: str) -> list:
        pass
```

**5. ChromaDB Implementation (`chroma_store.py`)**
```python
import chromadb
from storage import IVectorStore
from config import Config

class ChromaVectorStore(IVectorStore):
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.DB_PATH)
        self.collection = self.client.get_or_create_collection("user_memories")
    
    def add_memory(self, memory_id: str, embedding: list[float], content: str):
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content]
        )
    
    def query(self, embedding: list[float], filter: dict, n: int) -> list:
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where=filter
        )
```

**6. SQLite Implementation (`sqlite_store.py`)**
```python
import sqlite3
import uuid
from storage import IMetadataStore
from config import Config

class SQLiteMetadataStore(IMetadataStore):
    def __init__(self):
        self.conn = sqlite3.connect(Config.SQL_PATH)
        self._initialize_db()
    
    def _initialize_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                content TEXT,
                is_deleted BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def add_memory(self, memory_id: str, user_id: str, content: str):
        self.conn.execute(
            "INSERT INTO memories (id, user_id, content) VALUES (?, ?, ?)",
            (memory_id, user_id, content)
        )
        self.conn.commit()
    
    def delete_memory(self, memory_id: str):
        self.conn.execute(
            "UPDATE memories SET is_deleted=1 WHERE id=?",
            (memory_id,)
        )
        self.conn.commit()
    
    def get_active_memories(self, user_id: str) -> list:
        cursor = self.conn.execute(
            "SELECT id FROM memories WHERE user_id=? AND is_deleted=0",
            (user_id,)
        )
        return [row[0] for row in cursor.fetchall()]
```

**7. Memory Storage Service (`memory_storage.py`)**
```python
import uuid
from storage import IVectorStore, IMetadataStore

class MemoryStorage:
    def __init__(
        self, 
        vector_store: IVectorStore, 
        metadata_store: IMetadataStore,
        embedding_service
    ):
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.embedding_service = embedding_service
    
    def add_memory(self, user_id: str, fact: str):
        memory_id = str(uuid.uuid4())
        
        # Store metadata
        self.metadata_store.add_memory(memory_id, user_id, fact)
        
        # Generate and store embedding
        embedding = self.embedding_service.generate_embedding(fact)
        self.vector_store.add_memory(memory_id, embedding, fact)
    
    def delete_memory(self, user_id: str, fact: str):
        # Find matching memory
        embedding = self.embedding_service.generate_embedding(fact)
        active_ids = self.metadata_store.get_active_memories(user_id)
        results = self.vector_store.query(
            embedding,
            filter={"id": {"$in": active_ids}},
            n=1
        )
        
        if results and results["ids"][0]:
            memory_id = results["ids"][0][0]
            self.metadata_store.delete_memory(memory_id)
    
    def retrieve_memories(self, user_id: str, query: str, n: int = 5) -> list:
        # Get active memory IDs
        active_ids = self.metadata_store.get_active_memories(user_id)
        
        # Search relevant memories
        embedding = self.embedding_service.generate_embedding(query)
        results = self.vector_store.query(
            embedding,
            filter={"id": {"$in": active_ids}},
            n=n
        )
        return results["documents"][0] if results else []
```

**8. Memory Manager (`memory_manager.py`)**
```python
class MemoryManager:
    def __init__(self, fact_extractor, memory_storage):
        self.fact_extractor = fact_extractor
        self.memory_storage = memory_storage
    
    def process_message(self, user_id: str, message: str, action: str = "add"):
        facts = self.fact_extractor.extract_facts(message)
        
        for fact in facts:
            if action == "add":
                self.memory_storage.add_memory(user_id, fact)
            elif action == "delete":
                self.memory_storage.delete_memory(user_id, fact)
    
    def get_context(self, user_id: str, query: str) -> str:
        memories = self.memory_storage.retrieve_memories(user_id, query)
        return "\n".join([f"- {mem}" for mem in memories])
```

**9. GPT Integration (`gpt_integration.py`)**
```python
from openai import OpenAI
from config import Config

class GPTIntegration:
    def __init__(self, client: OpenAI, memory_manager: MemoryManager):
        self.client = client
        self.memory_manager = memory_manager
        self.model = Config.GPT_MODEL
    
    def query(self, user_id: str, question: str) -> str:
        # Retrieve relevant memories
        context = self.memory_manager.get_context(user_id, question)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"Relevant memories:\n{context}"},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content
```

**10. Main Application (`app.py`)**
```python
from openai import OpenAI
from fact_extractor import FactExtractor
from embedding_service import EmbeddingService
from chroma_store import ChromaVectorStore
from sqlite_store import SQLiteMetadataStore
from memory_storage import MemoryStorage
from memory_manager import MemoryManager
from gpt_integration import GPTIntegration

# Initialize dependencies
client = OpenAI(api_key="YOUR_API_KEY")

# Create components
fact_extractor = FactExtractor(client)
embedding_service = EmbeddingService(client)
vector_store = ChromaVectorStore()
metadata_store = SQLiteMetadataStore()
memory_storage = MemoryStorage(vector_store, metadata_store, embedding_service)
memory_manager = MemoryManager(fact_extractor, memory_storage)
gpt = GPTIntegration(client, memory_manager)

# Example usage
user_id = "user_123"

# Add memories
memory_manager.process_message(user_id, "I use Shram and Magnet as productivity tools")

# Delete memory
memory_manager.process_message(user_id, "I don't use Magnet anymore", action="delete")

# Query with memory
response = gpt.query(user_id, "What productivity tools do I use?")
print(response)  # "You use Shram."
```

### Key Modularity Features

1. **Separation of Concerns**:
   - Each component handles a specific responsibility
   - Clear interfaces between modules (ABC interfaces)
   - Single Responsibility Principle enforced

2. **Replaceable Components**:
   - Vector store interchangeable (ChromaDB, Pinecone, FAISS)
   - Metadata store interchangeable (SQLite, PostgreSQL, MySQL)
   - Fact extraction strategies can be modified

3. **Dependency Injection**:
   - Components receive dependencies via constructor
   - Easy to mock for testing
   - Swappable implementations

4. **Configurable**:
   - Centralized configuration
   - Environment variable support
   - Model parameters in one place

5. **Extendable**:
   - New storage systems can implement interfaces
   - Additional processing steps can be added
   - Multiple memory types support (future)

### Scaling Considerations

1. For production use:
   - Replace SQLite with PostgreSQL/MySQL
   - Use ChromaDB in client-server mode
   - Add connection pooling
   - Implement caching layer

2. Performance:
   - Batch embedding generation
   - Asynchronous operations
   - Vector indexing optimization

3. Enhanced Features:
   - Memory expiration policies
   - Memory confidence scoring
   - User-specific embedding models
   - Conflict resolution for contradictory memories

This modular design allows independent development and testing of components while maintaining a clean architecture for long-term memory management with OpenAI integration.