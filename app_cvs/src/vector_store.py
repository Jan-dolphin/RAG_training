"""Vector store management with ChromaDB"""

import logging
import shutil
import stat
from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

from .config import RAGConfig

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages ChromaDB vector store operations"""

    def __init__(self, config: RAGConfig, embeddings: AzureOpenAIEmbeddings):
        """
        Initialize vector store manager

        Args:
            config: RAG configuration
            embeddings: Azure OpenAI embeddings instance
        """
        self.config = config
        self.embeddings = embeddings
        self._vectorstore: Optional[Chroma] = None

    def _force_remove_readonly(self, func, path, exc_info):
        """Helper to remove read-only files"""
        Path(path).chmod(stat.S_IWRITE)
        func(path)

    def clear_vectorstore(self) -> None:
        """Delete existing vector store"""
        persist_dir = Path(self.config.persist_directory)

        if persist_dir.exists():
            logger.info(f"Clearing vector store at {persist_dir}")
            shutil.rmtree(persist_dir, onerror=self._force_remove_readonly)
            logger.info("Vector store cleared")

    def create_or_load_vectorstore(self) -> Chroma:
        """
        Create new empty vector store or load existing one

        Note: This creates an empty vectorstore. Documents are added via ParentDocumentRetriever.

        Returns:
            Chroma vector store instance
        """
        persist_dir = Path(self.config.persist_directory)

        # If vectorstore exists, load it
        if persist_dir.exists():
            logger.info(f"Loading existing vector store from {persist_dir}")
            return self.load_vectorstore()

        # Otherwise, create new empty vectorstore
        logger.info(f"Creating new empty vector store at {persist_dir}")

        persist_dir.mkdir(parents=True, exist_ok=True)

        self._vectorstore = Chroma(
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
            persist_directory=self.config.persist_directory,
            collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity for text-embedding-ada-002
        )

        logger.info("Empty vector store created with cosine similarity metric")
        return self._vectorstore

    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Load existing vector store from disk

        Returns:
            Chroma vector store instance or None if not found
        """
        persist_dir = Path(self.config.persist_directory)

        if not persist_dir.exists():
            logger.warning(f"Vector store not found at {persist_dir}")
            return None

        logger.info(f"Loading vector store from {persist_dir}")

        self._vectorstore = Chroma(
            persist_directory=self.config.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name
        )

        # Get collection count
        try:
            count = self._vectorstore._collection.count()
            logger.info(f"Vector store loaded with {count} documents")
        except Exception as e:
            logger.warning(f"Could not get document count: {e}")

        return self._vectorstore

    def get_vectorstore(self) -> Optional[Chroma]:
        """
        Get current vector store instance

        Returns:
            Chroma vector store instance or None
        """
        return self._vectorstore


    def get_stats(self) -> dict:
        """
        Get vector store statistics

        Returns:
            Dictionary with statistics
        """
        if self._vectorstore is None:
            return {"status": "not_initialized", "document_count": 0}

        try:
            count = self._vectorstore._collection.count()
            return {
                "status": "initialized",
                "document_count": count,
                "collection_name": self.config.collection_name,
                "persist_directory": self.config.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}
