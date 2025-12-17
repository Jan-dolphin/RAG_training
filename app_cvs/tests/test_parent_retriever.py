"""Unit tests for parent_retriever module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.parent_retriever import CVParentRetriever
from src.config import RAGConfig
from src.models import RetrievalResult


class TestCVParentRetriever:
    """Test CVParentRetriever class"""

    @pytest.fixture
    def mock_config(self):
        """Create mock RAG configuration"""
        config = RAGConfig()
        config.parent_chunk_size = 2000
        config.parent_chunk_overlap = 200
        config.child_chunk_size = 400
        config.child_chunk_overlap = 50
        config.top_k = 5
        return config

    @pytest.fixture
    def mock_vectorstore(self):
        """Create mock vector store"""
        vs = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 10
        vs._collection = mock_collection
        return vs

    @pytest.fixture
    def parent_retriever(self, mock_config, mock_vectorstore):
        """Create CVParentRetriever instance"""
        return CVParentRetriever(mock_config, mock_vectorstore)

    def test_init(self, parent_retriever, mock_config, mock_vectorstore):
        """Test initialization"""
        assert parent_retriever.config == mock_config
        assert parent_retriever.vectorstore == mock_vectorstore
        assert parent_retriever._retriever is None

    def test_create_splitters(self, parent_retriever):
        """Test splitter creation"""
        parent_splitter, child_splitter = parent_retriever._create_splitters()

        assert parent_splitter._chunk_size == 2000
        assert parent_splitter._chunk_overlap == 200
        assert child_splitter._chunk_size == 400
        assert child_splitter._chunk_overlap == 50

    @patch('src.parent_retriever.ParentDocumentRetriever')
    def test_initialize_retriever(self, mock_pdr_class, parent_retriever):
        """Test retriever initialization"""
        mock_retriever = Mock()
        mock_pdr_class.return_value = mock_retriever

        documents = [
            Document(page_content="CV content 1", metadata={"candidate_name": "Alice"}),
            Document(page_content="CV content 2", metadata={"candidate_name": "Bob"})
        ]

        parent_retriever.initialize_retriever(documents)

        mock_pdr_class.assert_called_once()
        mock_retriever.add_documents.assert_called_once_with(documents)
        assert parent_retriever._retriever == mock_retriever

    def test_retrieve_not_initialized(self, parent_retriever):
        """Test retrieve when not initialized"""
        with pytest.raises(ValueError, match="Retriever not initialized"):
            parent_retriever.retrieve("test query")

    @patch('src.parent_retriever.ParentDocumentRetriever')
    def test_retrieve_success(self, mock_pdr_class, parent_retriever):
        """Test successful retrieval"""
        mock_retriever = Mock()
        mock_results = [
            Document(page_content="CV 1 full content", metadata={"candidate_name": "Alice"}),
            Document(page_content="CV 2 full content", metadata={"candidate_name": "Bob"})
        ]
        mock_retriever.invoke.return_value = mock_results
        mock_pdr_class.return_value = mock_retriever

        # Initialize
        parent_retriever._retriever = mock_retriever

        # Retrieve
        results = parent_retriever.retrieve("Python developer", top_k=2)

        mock_retriever.invoke.assert_called_once_with("Python developer")
        assert len(results) == 2
        assert results[0].metadata["candidate_name"] == "Alice"

    @patch('src.parent_retriever.ParentDocumentRetriever')
    def test_retrieve_uses_default_top_k(self, mock_pdr_class, parent_retriever):
        """Test that retrieve uses config default top_k when not specified"""
        mock_retriever = Mock()
        mock_results = [Document(page_content="CV", metadata={})] * 10
        mock_retriever.invoke.return_value = mock_results
        mock_pdr_class.return_value = mock_retriever

        parent_retriever._retriever = mock_retriever

        # Call without top_k
        results = parent_retriever.retrieve("test")

        # Should limit to config.top_k (5)
        assert len(results) == 5

    def test_retrieve_with_scores(self, parent_retriever, mock_vectorstore):
        """Test retrieval with similarity scores"""
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Document(page_content="CV 1", metadata={"candidate_name": "Alice"}), 0.95),
            (Document(page_content="CV 2", metadata={"candidate_name": "Bob"}), 0.87)
        ]

        results = parent_retriever.retrieve_with_scores("Python", top_k=2)

        mock_vectorstore.similarity_search_with_score.assert_called_once_with("Python", k=2)
        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)
        assert results[0].candidate_name == "Alice"
        assert results[0].score == 0.95
        assert results[1].candidate_name == "Bob"
        assert results[1].score == 0.87

    def test_get_stats_initialized(self, parent_retriever):
        """Test getting statistics when initialized"""
        mock_retriever = Mock()
        parent_retriever._retriever = mock_retriever
        parent_retriever.docstore.yield_keys = Mock(return_value=iter(["key1", "key2", "key3"]))

        stats = parent_retriever.get_stats()

        assert stats["status"] == "initialized"
        assert stats["parent_chunks"] == 3
        assert stats["child_chunks"] == 10  # From mock_vectorstore fixture
        assert stats["parent_chunk_size"] == 2000
        assert stats["child_chunk_size"] == 400

    def test_get_stats_not_initialized(self, parent_retriever):
        """Test getting statistics when not initialized"""
        parent_retriever.docstore.yield_keys = Mock(return_value=iter([]))

        stats = parent_retriever.get_stats()

        assert stats["status"] == "not_initialized"

    def test_get_stats_error(self, parent_retriever, mock_vectorstore):
        """Test getting statistics with error"""
        mock_vectorstore._collection.count.side_effect = Exception("Test error")

        stats = parent_retriever.get_stats()

        assert stats["status"] == "error"
        assert "error" in stats
