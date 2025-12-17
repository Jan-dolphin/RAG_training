"""Unit tests for training module"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from src.training import TrainingPipeline
from src.config import AppConfig
from src.models import Candidate


class TestTrainingPipeline:
    """Test TrainingPipeline class"""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock app configuration"""
        config = AppConfig()
        config.rag.data_directory = str(tmp_path / "data")
        config.rag.persist_directory = str(tmp_path / "chroma")
        return config

    @pytest.fixture
    def training_pipeline(self, mock_config):
        """Create TrainingPipeline instance"""
        return TrainingPipeline(mock_config)

    def test_init(self, training_pipeline, mock_config):
        """Test initialization"""
        assert training_pipeline.config == mock_config
        assert training_pipeline.metrics.total_documents == 0

    def test_init_with_log_file(self, mock_config, tmp_path):
        """Test initialization with log file"""
        log_file = tmp_path / "training.log"
        pipeline = TrainingPipeline(mock_config, log_file=str(log_file))

        assert pipeline.log_file == str(log_file)

    @patch('src.training.CVDocumentLoader')
    def test_load_documents_success(self, mock_loader_class, training_pipeline):
        """Test successful document loading"""
        mock_loader = Mock()
        mock_candidates = [
            Candidate(name="Alice", full_cv_text="CV1", file_path="a.docx"),
            Candidate(name="Bob", full_cv_text="CV2", file_path="b.docx")
        ]
        mock_loader.load_all_cvs.return_value = mock_candidates
        mock_loader_class.return_value = mock_loader

        loader = training_pipeline.load_documents()

        assert training_pipeline.metrics.total_documents == 2
        assert loader == mock_loader

    @patch('src.training.CVDocumentLoader')
    def test_load_documents_failure(self, mock_loader_class, training_pipeline):
        """Test document loading failure"""
        mock_loader_class.side_effect = Exception("Load error")

        with pytest.raises(Exception, match="Load error"):
            training_pipeline.load_documents()

        assert len(training_pipeline.metrics.errors) > 0

    @patch('src.training.EmbeddingsManager')
    def test_setup_embeddings_success(self, mock_emb_class, training_pipeline):
        """Test successful embeddings setup"""
        mock_emb = Mock()
        mock_emb.test_connection.return_value = True
        mock_emb_class.return_value = mock_emb

        embeddings_mgr = training_pipeline.setup_embeddings()

        assert embeddings_mgr == mock_emb
        mock_emb.test_connection.assert_called_once()

    @patch('src.training.EmbeddingsManager')
    def test_setup_embeddings_connection_fail(self, mock_emb_class, training_pipeline):
        """Test embeddings setup with connection failure"""
        mock_emb = Mock()
        mock_emb.test_connection.return_value = False
        mock_emb_class.return_value = mock_emb

        embeddings_mgr = training_pipeline.setup_embeddings()

        assert len(training_pipeline.metrics.errors) > 0

    @patch('src.training.VectorStoreManager')
    def test_create_vector_store(self, mock_vs_class, training_pipeline):
        """Test vector store creation"""
        # Mock loader
        mock_loader = Mock()
        mock_candidates = [Candidate(name="Alice", full_cv_text="CV", file_path="a.docx")]
        mock_loader.load_all_cvs.return_value = mock_candidates
        mock_loader.convert_to_langchain_documents.return_value = [Mock()]

        # Mock embeddings manager
        mock_emb_mgr = Mock()

        # Mock vector store manager
        mock_vs = Mock()
        mock_vs.get_stats.return_value = {"document_count": 1}
        mock_vs_class.return_value = mock_vs

        vs_manager = training_pipeline.create_vector_store(mock_loader, mock_emb_mgr)

        mock_vs.clear_vectorstore.assert_called_once()
        mock_vs.create_vectorstore.assert_called_once()
        assert vs_manager == mock_vs

    @patch('src.training.CVParentRetriever')
    def test_initialize_retriever(self, mock_retriever_class, training_pipeline):
        """Test retriever initialization"""
        # Mock loader
        mock_loader = Mock()
        mock_candidates = [Candidate(name="Alice", full_cv_text="CV", file_path="a.docx")]
        mock_loader.load_all_cvs.return_value = mock_candidates
        mock_loader.convert_to_langchain_documents.return_value = [Mock()]

        # Mock vector store manager
        mock_vs_mgr = Mock()

        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.get_stats.return_value = {
            "parent_chunks": 5,
            "child_chunks": 20
        }
        mock_retriever_class.return_value = mock_retriever

        retriever = training_pipeline.initialize_retriever(mock_loader, mock_vs_mgr)

        mock_retriever.initialize_retriever.assert_called_once()
        assert training_pipeline.metrics.total_parent_chunks == 5
        assert training_pipeline.metrics.total_child_chunks == 20
        assert retriever == mock_retriever

    def test_test_retrieval(self, training_pipeline):
        """Test retrieval testing"""
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.metadata = {"candidate_name": "Alice"}
        mock_doc.page_content = "Test content"
        mock_retriever.retrieve.return_value = [mock_doc]

        # Should not raise
        training_pipeline.test_retrieval(mock_retriever, ["test query"])

        mock_retriever.retrieve.assert_called_once()

    def test_test_retrieval_with_error(self, training_pipeline):
        """Test retrieval testing with error"""
        mock_retriever = Mock()
        mock_retriever.retrieve.side_effect = Exception("Retrieval error")

        # Should not raise, but log error
        training_pipeline.test_retrieval(mock_retriever, ["test query"])

        assert len(training_pipeline.metrics.errors) > 0

    def test_save_metrics(self, training_pipeline, tmp_path):
        """Test metrics saving"""
        output_file = tmp_path / "metrics.json"

        training_pipeline.metrics.total_documents = 10
        training_pipeline.save_metrics(str(output_file))

        assert output_file.exists()

        import json
        with open(output_file, 'r') as f:
            data = json.load(f)

        assert data["total_documents"] == 10

    @patch('src.training.TrainingPipeline.load_documents')
    @patch('src.training.TrainingPipeline.setup_embeddings')
    @patch('src.training.TrainingPipeline.create_vector_store')
    @patch('src.training.TrainingPipeline.initialize_retriever')
    @patch('src.training.TrainingPipeline.test_retrieval')
    def test_run_full_pipeline_success(
        self,
        mock_test_retrieval,
        mock_init_retriever,
        mock_create_vs,
        mock_setup_emb,
        mock_load_docs,
        training_pipeline
    ):
        """Test successful full pipeline run"""
        mock_load_docs.return_value = Mock()
        mock_setup_emb.return_value = Mock()
        mock_create_vs.return_value = Mock()
        mock_init_retriever.return_value = Mock()

        result = training_pipeline.run_full_pipeline(save_metrics=False)

        assert result["status"] == "success"
        assert "metrics" in result
        mock_load_docs.assert_called_once()
        mock_setup_emb.assert_called_once()
        mock_create_vs.assert_called_once()
        mock_init_retriever.assert_called_once()
        mock_test_retrieval.assert_called_once()

    @patch('src.training.TrainingPipeline.load_documents')
    def test_run_full_pipeline_failure(self, mock_load_docs, training_pipeline):
        """Test pipeline failure"""
        mock_load_docs.side_effect = Exception("Pipeline error")

        with pytest.raises(Exception, match="Pipeline error"):
            training_pipeline.run_full_pipeline(save_metrics=False)

        assert len(training_pipeline.metrics.errors) > 0
