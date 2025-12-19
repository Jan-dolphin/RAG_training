"""Document loader for CV documents supporting multiple formats"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Type
import csv

# LangChain loaders
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader
)
from langchain_core.documents import Document

from .models import Candidate

logger = logging.getLogger(__name__)


class CVDocumentLoader:
    """
    Robust document loader for CVs.
    Supports recursive loading and multiple formats: .docx, .pdf, .txt, .csv
    """

    def __init__(self, data_directory: str):
        """
        Initialize the document loader

        Args:
            data_directory: Path to directory containing files
        """
        self.data_directory = Path(data_directory)
        if not self.data_directory.exists():
            raise ValueError(f"Data directory does not exist: {data_directory}")

        # Map extensions to LangChain loaders
        self.loaders: Dict[str, Type] = {
            ".docx": Docx2txtLoader,
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": TextLoader,
            ".csv": CSVLoader
        }

    def _get_candidate_name_from_path(self, file_path: Path) -> str:
        """
        Extract candidate name from filename using heuristic.
        Falls back to filename without extension.
        """
        filename = file_path.name
        # Heuristic: Remove common suffixes like _CV, _EN, etc.
        name_part = file_path.stem
        for suffix in ["_CV_EN", "_CV", "_EN", "_cv", "_en"]:
             if name_part.endswith(suffix):
                 name_part = name_part[:-len(suffix)]
        
        # Replace underscores with spaces
        return name_part.replace("_", " ").strip()

    def _load_file_content(self, file_path: Path) -> str:
        """
        Load text content from file using appropriate LangChain loader
        """
        suffix = file_path.suffix.lower()
        if suffix not in self.loaders:
            raise ValueError(f"Unsupported file format: {suffix}")

        LoaderClass = self.loaders[suffix]
        
        # CSV handling might be specific if we want all rows as one text or separate
        # For CVs, we assume one file = one candidate context usually, 
        # but for CSV it might be one row = one candidate. 
        # Requirement says "load different file formats", let's stick to standard loading.
        # If CSV, we assume it's a document. 
        
        loader_kwargs = {}
        if suffix == ".txt" or suffix == ".md":
            loader_kwargs["encoding"] = "utf-8"
            
        loader = LoaderClass(str(file_path), **loader_kwargs)
        docs = loader.load()
        
        # Combine all pages/rows into one text
        return "\n\n".join([doc.page_content for doc in docs])

    def load_single_cv(self, file_path: str) -> Optional[Candidate]:
        """
        Load a single CV from file
        """
        path = Path(file_path)
        try:
            text = self._load_file_content(path)

            if not text or len(text.strip()) < 50:
                logger.warning(f"File {path.name} is empty or too short")
                return None

            candidate_name = self._get_candidate_name_from_path(path)

            candidate = Candidate(
                name=candidate_name,
                full_cv_text=text.strip(),
                file_path=str(path),
                metadata={
                    "filename": path.name,
                    "file_size": path.stat().st_size,
                    "text_length": len(text),
                    "extension": path.suffix.lower()
                }
            )

            logger.info(f"Loaded CV for {candidate.name} ({candidate.metadata['extension']}, {len(text)} chars)")
            return candidate

        except Exception as e:
            logger.error(f"Failed to load {path.name}: {e}")
            return None

    def load_all_cvs(self) -> List[Candidate]:
        """
        Load all CV documents from data directory (recursive)
        """
        candidates = []
        
        logger.info(f"Scanning directory {self.data_directory} recursively...")
        
        # Recursive walk
        for root, _, files in os.walk(self.data_directory):
            for filename in files:
                file_path = Path(root) / filename
                
                # Check extension
                if file_path.suffix.lower() in self.loaders:
                    candidate = self.load_single_cv(str(file_path))
                    if candidate:
                        candidates.append(candidate)
                        
        logger.info(f"Successfully loaded {len(candidates)} documents")
        return candidates

    def convert_to_langchain_documents(self, candidates: List[Candidate]) -> List[Document]:
        """
        Convert Candidate objects to LangChain Document objects
        """
        documents = []

        for candidate in candidates:
            doc = Document(
                page_content=candidate.full_cv_text,
                metadata={
                    "candidate_name": candidate.name,
                    "source": candidate.file_path,
                    "type": "cv_parent",
                    **candidate.metadata
                }
            )
            documents.append(doc)

        logger.info(f"Converted {len(documents)} candidates to LangChain documents")
        return documents
