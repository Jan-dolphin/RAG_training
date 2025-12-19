import sys
from unittest.mock import MagicMock, patch

# Mock module availability before importing the factory
mock_retrievers = MagicMock()
mock_compression = MagicMock()
mock_ensemble = MagicMock()
mock_mq = MagicMock()
mock_self_query = MagicMock()

sys.modules["langchain.retrievers"] = mock_retrievers
sys.modules["langchain.retrievers.contextual_compression"] = mock_compression
sys.modules["langchain.retrievers.ensemble"] = mock_ensemble
sys.modules["langchain.retrievers.multi_query"] = mock_mq
sys.modules["langchain.retrievers.self_query.base"] = mock_self_query

# Also mock langchain_community
mock_community = MagicMock()
sys.modules["langchain_community.retrievers"] = mock_community

# Now we can import the factory
from src.retriever_factory import RetrieverFactory
from src.config import AppConfig, RAGConfig
import unittest

class TestRetrieverFactory(unittest.TestCase):

    def setUp(self):
        self.config = AppConfig()
        self.config.rag.top_k = 2
        
        self.mock_vectorstore = MagicMock()
        self.mock_base_retriever = MagicMock()
        self.mock_vectorstore.as_retriever.return_value = self.mock_base_retriever
        
        self.mock_llm = MagicMock()

    def test_create_vector_retriever(self):
        self.config.rag.retrieval_strategy = RAGConfig.RetrievalStrategy.VECTOR
        retriever = RetrieverFactory.create_retriever(self.config, self.mock_vectorstore)
        
        # Should return base retriever
        self.assertEqual(retriever, self.mock_base_retriever)

    def test_create_multi_query_retriever(self):
        self.config.rag.retrieval_strategy = RAGConfig.RetrievalStrategy.MULTI_QUERY
        
        # The factory imports MultiQueryRetriever. 
        # Since we mocked the module, the class is a MagicMock attribute of 'langchain.retrievers'
        # We need to ensure the factory sees it.
        
        # Re-importing inside test to ensure mocks are active if needed, but sys.modules handle it.
        # But we need to make sure 'MultiQueryRetriever.from_llm' works.
        
        # The factory does: from langchain.retrievers import MultiQueryRetriever
        # So MultiQueryRetriever in factory is mock_retrievers.MultiQueryRetriever via imports
        
        mock_mq_class = mock_retrievers.MultiQueryRetriever
        mock_mq_instance = MagicMock()
        mock_mq_class.from_llm.return_value = mock_mq_instance
        
        retriever = RetrieverFactory.create_retriever(self.config, self.mock_vectorstore, llm=self.mock_llm)
        
        self.assertEqual(retriever, mock_mq_instance)
        mock_mq_class.from_llm.assert_called_with(retriever=self.mock_base_retriever, llm=self.mock_llm)

    def test_create_compression_retriever(self):
        self.config.rag.retrieval_strategy = RAGConfig.RetrievalStrategy.COMPRESSION
        
        # Mock LLMChainExtractor which is imported in factory
        # Factory: from langchain.retrievers.document_compressors import LLMChainExtractor
        # We need to mock that module too.
        mock_doc_compressors = MagicMock()
        sys.modules["langchain.retrievers.document_compressors"] = mock_doc_compressors
        
        # Reload factory to pick up new mock? No, sys.modules hack works if done before import.
        # But we already imported factory.
        # We probably missed mocking 'document_compressors' before import.
        # So factory might have failed to import if real module missing.
        # Assuming the initial import succeeded (it might have created a mock if parent mocked?)
        
        # Let's patch the name in the factory module directly
        with patch("src.retriever_factory.LLMChainExtractor") as mock_extractor_class:
            mock_compressor = MagicMock()
            mock_extractor_class.from_llm.return_value = mock_compressor
            
            with patch("src.retriever_factory.ContextualCompressionRetriever") as mock_cc_class:
                mock_cc_instance = MagicMock()
                mock_cc_class.return_value = mock_cc_instance
                
                retriever = RetrieverFactory.create_retriever(self.config, self.mock_vectorstore, llm=self.mock_llm)
                
                self.assertEqual(retriever, mock_cc_instance)

if __name__ == '__main__':
    unittest.main()
