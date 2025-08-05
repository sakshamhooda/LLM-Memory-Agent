# LLM Memory Agent

A long-term memory system for GPT that enables persistent, context-aware conversations using OpenAI APIs, ChromaDB for vector storage, and SQLite for metadata management.

## 🎯 Features

- **Fact Extraction**: Automatically extracts atomic facts from natural language using GPT
- **Memory Storage**: Hybrid storage with ChromaDB (vectors) and SQLite (metadata)
- **Memory Deletion**: Soft-delete memories with natural language commands
- **Context-Aware Responses**: GPT responses enhanced with relevant memories
- **Memory Search**: Semantic search through stored memories
- **Modular Architecture**: Clean separation of concerns with replaceable components

## 🏗️ Architecture

```
┌──────────────────────┐       ┌───────────────────┐
│   MemoryAgent        │───────│  FactExtractor    │
└──────────────────────┘       └───────────────────┘
         │                             │
         ├─────────────────────────────┤
         │                             │
┌──────────────────────┐       ┌───────────────────┐
│   MemoryManager      │───────│  EmbeddingService │
└──────────────────────┘       └───────────────────┘
         │
         ├──────────────────────┐
         │                      │
┌──────────────────┐    ┌──────────────────┐
│  ChromaVectorStore│    │  SQLiteMetadataStore│
│  (ChromaDB)      │    │  (SQLite)        │
└──────────────────┘    └──────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Run the Demo

```bash
python main.py
```

### 4. Interactive Mode

```bash
python main.py --interactive
```

## 📖 Usage Examples

### Basic Usage

```python
from app import MemoryAgent

# Initialize the agent
agent = MemoryAgent()

# Add memories
result = agent.add_memory("user123", "I use Shram and Magnet as productivity tools")
print(result['message'])  # "Successfully added 2 memories"

# Delete memories
result = agent.delete_memory("user123", "I don't use Magnet anymore")
print(result['message'])  # "Successfully deleted 1 memories"

# Query with memory context
response = agent.query_with_memory("user123", "What productivity tools do I use?")
print(response)  # "You use Shram as a productivity tool."
```

### Advanced Usage

```python
# Get memory summary
summary = agent.get_memory_summary("user123")
print(summary['stats'])

# Search memories
results = agent.search_memories("user123", "productivity", n=5)
for memory in results:
    print(memory['content'])

# Add memory directly
memory_id = agent.add_memory_directly("user123", "User likes coffee", {"category": "preferences"})
```

## 🔧 Configuration

Edit `config.py` to customize:

- **OpenAI Models**: Change embedding and GPT models
- **Database Paths**: Configure ChromaDB and SQLite paths
- **Memory Settings**: Adjust retrieval count and similarity thresholds

```python
class Config:
    EMBEDDING_MODEL = "text-embedding-3-small"
    GPT_MODEL = "gpt-4-turbo"
    DB_PATH = "memory_db"
    SQL_PATH = "memories.db"
    DEFAULT_MEMORY_RETRIEVAL_COUNT = 5
```

## 🧪 Testing

The demo script (`main.py`) demonstrates the core functionality:

1. **Memory Addition**: Extracts facts from natural language
2. **Memory Deletion**: Removes memories based on user statements
3. **Context-Aware Queries**: GPT responses with relevant memories
4. **Memory Search**: Semantic search through stored memories
5. **Memory Statistics**: Summary and analytics

## 📁 Project Structure

```
LLM-Memory-Agent/
├── app.py                 # Main MemoryAgent class
├── config.py              # Configuration settings
├── fact_extractor.py      # Fact extraction using GPT
├── embedding_service.py   # OpenAI embedding generation
├── storage.py             # Abstract storage interfaces
├── chroma_store.py        # ChromaDB vector store implementation
├── sqlite_store.py        # SQLite metadata store implementation
├── memory_storage.py      # Memory storage coordination
├── memory_manager.py      # Memory operations management
├── main.py               # Demo and interactive scripts
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔄 Workflow

1. **Memory Creation**:
   - User provides natural language input
   - FactExtractor uses GPT to extract atomic facts
   - EmbeddingService generates vector embeddings
   - MemoryStorage stores in both ChromaDB and SQLite

2. **Memory Retrieval**:
   - Query generates embedding
   - ChromaDB finds similar memories
   - SQLite filters active (non-deleted) memories
   - Relevant memories injected into GPT context

3. **Memory Deletion**:
   - Extract deletion facts from user input
   - Find similar memories in vector store
   - Soft-delete in SQLite, hard-delete from ChromaDB

## 🚀 Scaling Considerations

### Production Deployment

- **Database**: Replace SQLite with PostgreSQL/MySQL
- **Vector Store**: Use ChromaDB client-server mode or Pinecone
- **Caching**: Add Redis for embedding and query caching
- **Load Balancing**: Multiple agent instances with shared storage

### Performance Optimization

- **Batch Operations**: Process multiple memories simultaneously
- **Async Operations**: Non-blocking memory operations
- **Connection Pooling**: Efficient database connections
- **Vector Indexing**: Optimize similarity search performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT and embedding APIs
- ChromaDB for vector storage
- The modular architecture design pattern
