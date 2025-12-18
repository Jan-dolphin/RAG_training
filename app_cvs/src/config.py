"""Centralized configuration for RAG CV Application"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AzureConfig:
    """Azure OpenAI configuration"""
    endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    embedding_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002-dolphin-1"))
    api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"))
    llm_deployment: str = "gpt-4o"
    temperature: float = 0.0

    # Rate limiting settings
    max_retries: int = 5
    retry_delay: float = 1.0  # Initial delay in seconds
    max_retry_delay: float = 60.0  # Max delay between retries
    batch_size: int = 5  # Number of documents to process at once
    batch_delay: float = 2.0  # Delay between batches in seconds

    def validate(self) -> None:
        """Validate required configuration fields"""
        if not self.endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set")


@dataclass
class RAGConfig:
    """RAG pipeline configuration"""

    # Parent Document Retriever settings
    parent_chunk_size: int = 2000  # Celý CV nebo velká část
    parent_chunk_overlap: int = 200

    child_chunk_size: int = 400  # Menší části se znalostmi
    child_chunk_overlap: int = 50

    # Retrieval settings
    top_k: int = 5  # Počet dokumentů k vrácení

    # Hybrid search settings
    use_hybrid_search: bool = True  # Kombinuje BM25 (keyword) + embeddings (semantic)
    bm25_k: int = 10  # Počet výsledků z BM25 keyword search
    embedding_k: int = 10  # Počet výsledků z embedding search
    bm25_weight: float = 0.5  # Váha BM25 při fúzi (0.0-1.0)
    embedding_weight: float = 0.5  # Váha embeddings při fúzi (0.0-1.0)

    # Similarity threshold (používá se pro non-hybrid fallback)
    similarity_threshold: float = 0.4  # Max cosine distance pro relevantní výsledky
    # Poznámka: ChromaDB používá cosine similarity
    # 0.0-0.3: velmi relevantní, 0.3-0.5: relevantní, >0.5: často irelevantní

    # Vector store settings
    collection_name: str = "cv_candidates"
    persist_directory: str = "./chroma_db"

    # Data paths
    data_directory: str = "./data/OneDrive_2025-12-16"
    data_directory_ntb: str = "../data/OneDrive_2025-12-16"

@dataclass
class AppConfig:
    """Main application configuration"""
    azure: AzureConfig = field(default_factory=AzureConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.azure.validate()


def get_config() -> AppConfig:
    """Get application configuration singleton"""
    return AppConfig()
