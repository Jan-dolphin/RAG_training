"""Unit tests for vector_store module"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.vector_store import VectorStoreManager
from src.config import RAGConfig


class TestVectorStoreManager:
    """Test VectorStoreManager class"""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock RAG configuration"""
        config = RAGConfig()
        config.persist_directory = str(tmp_path / "test_chroma")
        config.collection_name = "test_collection"
        return config

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings"""
        return Mock()

    @pytest.fixture
    def vector_store_manager(self, mock_config, mock_embeddings):
        """Create VectorStoreManager instance"""
        return VectorStoreManager(mock_config, mock_embeddings)

    def test_init(self, vector_store_manager, mock_config, mock_embeddings):
        """Test initialization"""
        assert vector_store_manager.config == mock_config
        assert vector_store_manager.embeddings == mock_embeddings
        assert vector_store_manager._vectorstore is None

    def test_clear_vectorstore_existing(self, vector_store_manager, tmp_path):
        """Test clearing existing vector store"""
        # Create fake directory
        persist_dir = tmp_path / "test_chroma"
        persist_dir.mkdir()
        (persist_dir / "test_file.txt").touch()

        vector_store_manager.config.persist_directory = str(persist_dir)

        vector_store_manager.clear_vectorstore()

        assert not persist_dir.exists()

    def test_clear_vectorstore_nonexistent(self, vector_store_manager):
        """Test clearing non-existent vector store (should not raise error)"""
        vector_store_manager.clear_vectorstore()  # Should complete without error

    @patch('src.vector_store.Chroma.from_documents')
    def test_create_vectorstore(self, mock_from_docs, vector_store_manager, mock_config):
        """Test creating new vector store"""
        mock_vs = Mock()
        mock_from_docs.return_value = mock_vs

        documents = [
            Document(page_content="Test doc 1", metadata={"id": 1}),
            Document(page_content="Test doc 2", metadata={"id": 2})
        ]

        result = vector_store_manager.create_vectorstore(documents)

        mock_from_docs.assert_called_once_with(
            documents=documents,
            embedding=vector_store_manager.embeddings,
            collection_name=mock_config.collection_name,
            persist_directory=mock_config.persist_directory
        )
        assert result == mock_vs
        assert vector_store_manager._vectorstore == mock_vs

    @patch('src.vector_store.Chroma')
    def test_load_vectorstore_success(self, mock_chroma_class, vector_store_manager, tmp_path):
        """Test loading existing vector store"""
        # Create persist directory
        persist_dir = tmp_path / "test_chroma"
        persist_dir.mkdir()
        vector_store_manager.config.persist_directory = str(persist_dir)

        # Mock Chroma instance
        mock_vs = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 10
        mock_vs._collection = mock_collection
        mock_chroma_class.return_value = mock_vs

        result = vector_store_manager.load_vectorstore()

        mock_chroma_class.assert_called_once()
        assert result == mock_vs
        assert vector_store_manager._vectorstore == mock_vs

    def test_load_vectorstore_not_found(self, vector_store_manager):
        """Test loading non-existent vector store"""
        result = vector_store_manager.load_vectorstore()
        assert result is None

    def test_get_vectorstore(self, vector_store_manager):
        """Test getting current vector store"""
        mock_vs = Mock()
        vector_store_manager._vectorstore = mock_vs

        result = vector_store_manager.get_vectorstore()
        assert result == mock_vs

    def test_add_documents_success(self, vector_store_manager):
        """Test adding documents to vector store"""
        mock_vs = Mock()
        vector_store_manager._vectorstore = mock_vs

        documents = [Document(page_content="New doc", metadata={})]
        vector_store_manager.add_documents(documents)

        mock_vs.add_documents.assert_called_once_with(documents)

    def test_add_documents_not_initialized(self, vector_store_manager):
        """Test adding documents when vector store not initialized"""
        documents = [Document(page_content="New doc", metadata={})]

        with pytest.raises(ValueError, match="Vector store not initialized"):
            vector_store_manager.add_documents(documents)

    def test_similarity_search_success(self, vector_store_manager):
        """Test similarity search"""
        mock_vs = Mock()
        mock_results = [
            Document(page_content="Result 1", metadata={}),
            Document(page_content="Result 2", metadata={})
        ]
        mock_vs.similarity_search.return_value = mock_results
        vector_store_manager._vectorstore = mock_vs

        results = vector_store_manager.similarity_search("test query", k=2)

        mock_vs.similarity_search.assert_called_once_with("test query", k=2)
        assert results == mock_results

    def test_similarity_search_not_initialized(self, vector_store_manager):
        """Test similarity search when vector store not initialized"""
        with pytest.raises(ValueError, match="Vector store not initialized"):
            vector_store_manager.similarity_search("test query")

    def test_get_stats_initialized(self, vector_store_manager):
        """Test getting statistics when initialized"""
        mock_vs = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_vs._collection = mock_collection
        vector_store_manager._vectorstore = mock_vs

        stats = vector_store_manager.get_stats()

        assert stats["status"] == "initialized"
        assert stats["document_count"] == 42
        assert "collection_name" in stats
        assert "persist_directory" in stats

    def test_get_stats_not_initialized(self, vector_store_manager):
        """Test getting statistics when not initialized"""
        stats = vector_store_manager.get_stats()

        assert stats["status"] == "not_initialized"
        assert stats["document_count"] == 0

    def test_get_stats_error(self, vector_store_manager):
        """Test getting statistics with error"""
        mock_vs = Mock()
        mock_vs._collection.count.side_effect = Exception("Test error")
        vector_store_manager._vectorstore = mock_vs

        stats = vector_store_manager.get_stats()

        assert stats["status"] == "error"
        assert "error" in stats
