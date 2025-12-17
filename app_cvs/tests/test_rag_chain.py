"""Unit tests for rag_chain module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.rag_chain import CVRAGChain
from src.config import AzureConfig
from src.models import RAGResponse


class TestCVRAGChain:
    """Test CVRAGChain class"""

    @pytest.fixture
    def mock_config(self):
        """Create mock Azure configuration"""
        config = AzureConfig()
        config.endpoint = "https://test.openai.azure.com"
        config.api_key = "test-key"
        config.llm_deployment = "gpt-4o"
        config.api_version = "2023-05-15"
        config.temperature = 0.0
        return config

    @pytest.fixture
    def mock_retriever(self):
        """Create mock retriever"""
        retriever = Mock()
        retriever.retrieve.return_value = [
            Document(
                page_content="Alice has Python and AWS skills",
                metadata={"candidate_name": "Alice Smith"}
            ),
            Document(
                page_content="Bob has Java and Docker skills",
                metadata={"candidate_name": "Bob Jones"}
            )
        ]
        return retriever

    @pytest.fixture
    @patch('src.rag_chain.AzureChatOpenAI')
    def rag_chain(self, mock_llm_class, mock_config, mock_retriever):
        """Create CVRAGChain instance"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        chain = CVRAGChain(mock_config, mock_retriever)
        return chain

    @patch('src.rag_chain.AzureChatOpenAI')
    def test_init(self, mock_llm_class, mock_config, mock_retriever):
        """Test initialization"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        chain = CVRAGChain(mock_config, mock_retriever)

        assert chain.config == mock_config
        assert chain.retriever == mock_retriever
        assert chain.prompt_template == CVRAGChain.DEFAULT_PROMPT_TEMPLATE

        mock_llm_class.assert_called_once_with(
            azure_deployment="gpt-4o",
            openai_api_version="2023-05-15",
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
            temperature=0.0
        )

    @patch('src.rag_chain.AzureChatOpenAI')
    def test_init_custom_prompt(self, mock_llm_class, mock_config, mock_retriever):
        """Test initialization with custom prompt"""
        custom_prompt = "Custom prompt: {context} {question}"

        chain = CVRAGChain(mock_config, mock_retriever, prompt_template=custom_prompt)

        assert chain.prompt_template == custom_prompt

    def test_format_docs(self, rag_chain):
        """Test document formatting"""
        docs = [
            Document(
                page_content="Content 1",
                metadata={"candidate_name": "Alice"}
            ),
            Document(
                page_content="Content 2",
                metadata={"candidate_name": "Bob"}
            )
        ]

        formatted = rag_chain._format_docs(docs)

        assert "Alice" in formatted
        assert "Bob" in formatted
        assert "Content 1" in formatted
        assert "Content 2" in formatted
        assert "CV #1" in formatted
        assert "CV #2" in formatted

    def test_format_docs_unknown_candidate(self, rag_chain):
        """Test formatting docs with missing candidate name"""
        docs = [Document(page_content="Content", metadata={})]

        formatted = rag_chain._format_docs(docs)

        assert "Unknown" in formatted

    @patch('src.rag_chain.AzureChatOpenAI')
    def test_invoke(self, mock_llm_class, mock_config, mock_retriever):
        """Test RAG chain invocation"""
        # Setup mock LLM
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        # Mock the chain invocation to return a simple string
        with patch.object(CVRAGChain, '_create_chain') as mock_create_chain:
            mock_chain = Mock()
            mock_chain.invoke.return_value = "Alice Smith and Bob Jones have relevant skills."
            mock_create_chain.return_value = mock_chain

            chain = CVRAGChain(mock_config, mock_retriever)
            response = chain.invoke("Who has Python skills?")

        # Assertions
        assert isinstance(response, RAGResponse)
        assert response.query == "Who has Python skills?"
        assert response.answer == "Alice Smith and Bob Jones have relevant skills."
        assert len(response.retrieved_contexts) == 2
        assert response.retrieved_contexts[0].candidate_name == "Alice Smith"
        assert response.retrieved_contexts[1].candidate_name == "Bob Jones"
        assert response.metadata["num_contexts"] == 2

    @patch('src.rag_chain.AzureChatOpenAI')
    def test_batch_invoke(self, mock_llm_class, mock_config, mock_retriever):
        """Test batch query processing"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        with patch.object(CVRAGChain, '_create_chain') as mock_create_chain:
            mock_chain = Mock()
            mock_chain.invoke.side_effect = ["Answer 1", "Answer 2"]
            mock_create_chain.return_value = mock_chain

            chain = CVRAGChain(mock_config, mock_retriever)
            responses = chain.batch_invoke(["Query 1", "Query 2"])

        assert len(responses) == 2
        assert responses[0].query == "Query 1"
        assert responses[1].query == "Query 2"

    @patch('src.rag_chain.AzureChatOpenAI')
    def test_batch_invoke_with_error(self, mock_llm_class, mock_config, mock_retriever):
        """Test batch processing with error"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        # Mock retriever to raise error on second call
        mock_retriever.retrieve.side_effect = [
            [Document(page_content="Content", metadata={"candidate_name": "Alice"})],
            Exception("Retrieval error")
        ]

        with patch.object(CVRAGChain, '_create_chain') as mock_create_chain:
            mock_chain = Mock()
            mock_chain.invoke.return_value = "Answer"
            mock_create_chain.return_value = mock_chain

            chain = CVRAGChain(mock_config, mock_retriever)
            responses = chain.batch_invoke(["Query 1", "Query 2"])

        assert len(responses) == 2
        assert "error" in responses[1].answer.lower() or "error" in responses[1].metadata

    @patch('src.rag_chain.AzureChatOpenAI')
    def test_update_prompt(self, mock_llm_class, mock_config, mock_retriever):
        """Test prompt template update"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        chain = CVRAGChain(mock_config, mock_retriever)

        new_prompt = "New prompt: {context} - {question}"
        chain.update_prompt(new_prompt)

        assert chain.prompt_template == new_prompt

    @patch('src.rag_chain.AzureChatOpenAI')
    def test_get_llm(self, mock_llm_class, mock_config, mock_retriever):
        """Test getting LLM instance"""
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        chain = CVRAGChain(mock_config, mock_retriever)
        llm = chain.get_llm()

        assert llm == mock_llm
