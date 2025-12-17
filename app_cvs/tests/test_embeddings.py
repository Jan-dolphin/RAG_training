"""Unit tests for embeddings module"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.embeddings import EmbeddingsManager
from src.config import AzureConfig


class TestEmbeddingsManager:
    """Test EmbeddingsManager class"""

    @pytest.fixture
    def mock_config(self):
        """Create mock Azure configuration"""
        config = AzureConfig()
        config.endpoint = "https://test.openai.azure.com"
        config.api_key = "test-key"
        config.embedding_deployment = "test-embedding"
        config.api_version = "2023-05-15"
        return config

    @pytest.fixture
    def embeddings_manager(self, mock_config):
        """Create EmbeddingsManager instance"""
        return EmbeddingsManager(mock_config)

    @patch('src.embeddings.AzureOpenAIEmbeddings')
    def test_get_embeddings_creates_instance(self, mock_embeddings_class, embeddings_manager, mock_config):
        """Test that get_embeddings creates embeddings instance"""
        mock_instance = Mock()
        mock_embeddings_class.return_value = mock_instance

        result = embeddings_manager.get_embeddings()

        mock_embeddings_class.assert_called_once_with(
            azure_endpoint=mock_config.endpoint,
            api_key=mock_config.api_key,
            azure_deployment=mock_config.embedding_deployment,
            openai_api_version=mock_config.api_version
        )
        assert result == mock_instance

    @patch('src.embeddings.AzureOpenAIEmbeddings')
    def test_get_embeddings_singleton_pattern(self, mock_embeddings_class, embeddings_manager):
        """Test that get_embeddings returns same instance (singleton)"""
        mock_instance = Mock()
        mock_embeddings_class.return_value = mock_instance

        result1 = embeddings_manager.get_embeddings()
        result2 = embeddings_manager.get_embeddings()

        # Should only be called once
        assert mock_embeddings_class.call_count == 1
        assert result1 == result2

    @patch('src.embeddings.AzureOpenAIEmbeddings')
    def test_embed_query(self, mock_embeddings_class, embeddings_manager):
        """Test embedding a single query"""
        mock_instance = Mock()
        mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings_class.return_value = mock_instance

        result = embeddings_manager.embed_query("test query")

        mock_instance.embed_query.assert_called_once_with("test query")
        assert result == [0.1, 0.2, 0.3]

    @patch('src.embeddings.AzureOpenAIEmbeddings')
    def test_embed_documents(self, mock_embeddings_class, embeddings_manager):
        """Test embedding multiple documents"""
        mock_instance = Mock()
        mock_instance.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        mock_embeddings_class.return_value = mock_instance

        texts = ["doc1", "doc2"]
        result = embeddings_manager.embed_documents(texts)

        mock_instance.embed_documents.assert_called_once_with(texts)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch('src.embeddings.AzureOpenAIEmbeddings')
    def test_test_connection_success(self, mock_embeddings_class, embeddings_manager):
        """Test successful connection test"""
        mock_instance = Mock()
        # Ada-002 has 1536 dimensions
        mock_instance.embed_query.return_value = [0.1] * 1536
        mock_embeddings_class.return_value = mock_instance

        result = embeddings_manager.test_connection()

        assert result is True
        mock_instance.embed_query.assert_called_once_with("test connection")

    @patch('src.embeddings.AzureOpenAIEmbeddings')
    def test_test_connection_wrong_dimension(self, mock_embeddings_class, embeddings_manager):
        """Test connection with wrong embedding dimension"""
        mock_instance = Mock()
        mock_instance.embed_query.return_value = [0.1] * 100  # Wrong dimension
        mock_embeddings_class.return_value = mock_instance

        result = embeddings_manager.test_connection()

        assert result is False

    @patch('src.embeddings.AzureOpenAIEmbeddings')
    def test_test_connection_exception(self, mock_embeddings_class, embeddings_manager):
        """Test connection test with exception"""
        mock_instance = Mock()
        mock_instance.embed_query.side_effect = Exception("Connection error")
        mock_embeddings_class.return_value = mock_instance

        result = embeddings_manager.test_connection()

        assert result is False
