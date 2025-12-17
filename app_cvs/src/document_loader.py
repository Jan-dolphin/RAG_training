"""Document loader for CV DOCX files"""

import os
import logging
from pathlib import Path
from typing import List, Optional
import docx2txt
from langchain_core.documents import Document

from .models import Candidate

logger = logging.getLogger(__name__)


class CVDocumentLoader:
    """Loads CV documents from DOCX files"""

    def __init__(self, data_directory: str):
        """
        Initialize the document loader

        Args:
            data_directory: Path to directory containing DOCX files
        """
        self.data_directory = Path(data_directory)
        if not self.data_directory.exists():
            raise ValueError(f"Data directory does not exist: {data_directory}")

    def load_single_cv(self, file_path: str) -> Optional[Candidate]:
        """
        Load a single CV from DOCX file

        Args:
            file_path: Path to DOCX file

        Returns:
            Candidate object or None if loading fails
        """
        try:
            text = docx2txt.process(file_path)

            if not text or len(text.strip()) < 50:
                logger.warning(f"File {file_path} is empty or too short")
                return None

            # Extract candidate name from filename
            filename = os.path.basename(file_path)
            name_part = filename.replace("_CV_EN.docx", "").replace("_", " ")

            candidate = Candidate(
                name=name_part,
                full_cv_text=text.strip(),
                file_path=file_path,
                metadata={
                    "filename": filename,
                    "file_size": os.path.getsize(file_path),
                    "text_length": len(text)
                }
            )

            logger.info(f"Loaded CV for {candidate.name} ({len(text)} characters)")
            return candidate

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None

    def load_all_cvs(self) -> List[Candidate]:
        """
        Load all CV documents from data directory

        Returns:
            List of Candidate objects
        """
        candidates = []
        docx_files = list(self.data_directory.glob("*.docx"))

        logger.info(f"Found {len(docx_files)} DOCX files in {self.data_directory}")

        for file_path in docx_files:
            candidate = self.load_single_cv(str(file_path))
            if candidate:
                candidates.append(candidate)

        logger.info(f"Successfully loaded {len(candidates)} CVs")
        return candidates

    def convert_to_langchain_documents(self, candidates: List[Candidate]) -> List[Document]:
        """
        Convert Candidate objects to LangChain Document objects

        Args:
            candidates: List of Candidate objects

        Returns:
            List of LangChain Document objects with parent metadata
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
