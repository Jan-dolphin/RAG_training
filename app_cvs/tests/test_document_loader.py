"""Unit tests for document_loader module"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import os

from src.document_loader import CVDocumentLoader
from src.models import Candidate


class TestCVDocumentLoader:
    """Test CVDocumentLoader class"""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        return data_dir

    @pytest.fixture
    def loader(self, temp_data_dir):
        """Create CVDocumentLoader instance"""
        return CVDocumentLoader(str(temp_data_dir))

    def test_init_valid_directory(self, temp_data_dir):
        """Test initialization with valid directory"""
        loader = CVDocumentLoader(str(temp_data_dir))
        assert loader.data_directory == temp_data_dir

    def test_init_invalid_directory(self):
        """Test initialization with non-existent directory"""
        with pytest.raises(ValueError, match="Data directory does not exist"):
            CVDocumentLoader("/non/existent/path")

    @patch('src.document_loader.docx2txt.process')
    @patch('src.document_loader.os.path.getsize')
    def test_load_single_cv_success(self, mock_getsize, mock_process, loader, temp_data_dir):
        """Test successful loading of single CV"""
        # Setup
        test_file = temp_data_dir / "Doe_John_CV_EN.docx"
        test_file.touch()

        mock_process.return_value = "John Doe\nSoftware Engineer\nPython, Java, AWS"
        mock_getsize.return_value = 1024

        # Execute
        candidate = loader.load_single_cv(str(test_file))

        # Assert
        assert candidate is not None
        assert candidate.name == "Doe John"
        assert "Software Engineer" in candidate.full_cv_text
        assert candidate.file_path == str(test_file)
        assert candidate.metadata["filename"] == "Doe_John_CV_EN.docx"
        assert candidate.metadata["file_size"] == 1024

    @patch('src.document_loader.docx2txt.process')
    def test_load_single_cv_empty_file(self, mock_process, loader, temp_data_dir):
        """Test loading empty CV file"""
        test_file = temp_data_dir / "Empty_CV_EN.docx"
        test_file.touch()

        mock_process.return_value = ""

        candidate = loader.load_single_cv(str(test_file))

        assert candidate is None

    @patch('src.document_loader.docx2txt.process')
    def test_load_single_cv_exception(self, mock_process, loader, temp_data_dir):
        """Test handling exception during CV loading"""
        test_file = temp_data_dir / "Error_CV_EN.docx"
        test_file.touch()

        mock_process.side_effect = Exception("DOCX parsing error")

        candidate = loader.load_single_cv(str(test_file))

        assert candidate is None

    @patch('src.document_loader.CVDocumentLoader.load_single_cv')
    def test_load_all_cvs(self, mock_load_single, loader, temp_data_dir):
        """Test loading all CVs from directory"""
        # Create test files
        (temp_data_dir / "Person1_CV_EN.docx").touch()
        (temp_data_dir / "Person2_CV_EN.docx").touch()
        (temp_data_dir / "Person3_CV_EN.docx").touch()

        # Mock load_single_cv to return test candidates
        mock_load_single.side_effect = [
            Candidate(name="Person1", full_cv_text="CV1", file_path="p1.docx"),
            Candidate(name="Person2", full_cv_text="CV2", file_path="p2.docx"),
            None  # One file fails to load
        ]

        candidates = loader.load_all_cvs()

        assert len(candidates) == 2
        assert candidates[0].name == "Person1"
        assert candidates[1].name == "Person2"

    def test_convert_to_langchain_documents(self, loader):
        """Test conversion of Candidates to LangChain Documents"""
        candidates = [
            Candidate(
                name="Alice Smith",
                full_cv_text="Alice CV content with Python and AWS skills",
                file_path="/path/to/alice.docx",
                metadata={"filename": "alice.docx"}
            ),
            Candidate(
                name="Bob Jones",
                full_cv_text="Bob CV content with Java and Docker skills",
                file_path="/path/to/bob.docx",
                metadata={"filename": "bob.docx"}
            )
        ]

        documents = loader.convert_to_langchain_documents(candidates)

        assert len(documents) == 2

        # Check first document
        assert documents[0].page_content == candidates[0].full_cv_text
        assert documents[0].metadata["candidate_name"] == "Alice Smith"
        assert documents[0].metadata["type"] == "cv_parent"
        assert documents[0].metadata["filename"] == "alice.docx"

        # Check second document
        assert documents[1].page_content == candidates[1].full_cv_text
        assert documents[1].metadata["candidate_name"] == "Bob Jones"
        assert documents[1].metadata["type"] == "cv_parent"

    def test_convert_empty_list(self, loader):
        """Test conversion of empty candidate list"""
        documents = loader.convert_to_langchain_documents([])
        assert len(documents) == 0
