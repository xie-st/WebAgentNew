"""
Memory systems for effGen.

This package provides short-term, long-term, and vector-based memory systems
for managing conversation history, persistent data, and semantic search.
"""

from .long_term import (
    ImportanceLevel,
    JSONStorageBackend,
    LongTermMemory,
    MemoryEntry,
    MemoryType,
    Session,
    SQLiteStorageBackend,
    StorageBackend,
)
from .short_term import ConversationSummary, Message, MessageRole, ShortTermMemory
from .vector_store import (
    CHROMA_AVAILABLE,
    FAISS_AVAILABLE,
    ChromaBackend,
    EmbeddingModel,
    EmbeddingProvider,
    FAISSBackend,
    SearchResult,
    SentenceTransformerEmbedding,
    SimpleEmbedding,
    VectorMemoryEntry,
    VectorMemoryStore,
    VectorStoreBackend,
)

__all__ = [
    # Short-term memory
    "ShortTermMemory",
    "Message",
    "MessageRole",
    "ConversationSummary",

    # Long-term memory
    "LongTermMemory",
    "MemoryEntry",
    "Session",
    "MemoryType",
    "ImportanceLevel",
    "StorageBackend",
    "JSONStorageBackend",
    "SQLiteStorageBackend",

    # Vector store
    "VectorMemoryStore",
    "VectorMemoryEntry",
    "SearchResult",
    "EmbeddingProvider",
    "SentenceTransformerEmbedding",
    "SimpleEmbedding",
    "VectorStoreBackend",
    "FAISSBackend",
    "ChromaBackend",
    "EmbeddingModel",
    "FAISS_AVAILABLE",
    "CHROMA_AVAILABLE",
]
