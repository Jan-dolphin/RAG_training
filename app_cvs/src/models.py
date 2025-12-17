"""Data models for RAG CV Application"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Candidate:
    """Represents a candidate with their CV information"""
    name: str
    full_cv_text: str
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Extract name from file_path if not provided"""
        if not self.name and self.file_path:
            # Extract name from filename like "Baláček_Daniel_CV_EN.docx"
            filename = self.file_path.split("\\")[-1].split("/")[-1]
            name_part = filename.replace("_CV_EN.docx", "").replace("_", " ")
            self.name = name_part


@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    candidate_name: str
    content: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    """User query for CV search"""
    text: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 5
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RAGResponse:
    """Response from RAG pipeline"""
    query: str
    answer: str
    retrieved_contexts: List[RetrievalResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieved_contexts": [
                {
                    "candidate_name": ctx.candidate_name,
                    "content": ctx.content,
                    "score": ctx.score,
                    "metadata": ctx.metadata
                }
                for ctx in self.retrieved_contexts
            ],
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class TrainingMetrics:
    """Metrics from training/indexing process"""
    total_documents: int = 0
    total_parent_chunks: int = 0
    total_child_chunks: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def add_error(self, error: str) -> None:
        """Add an error message"""
        self.errors.append(error)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "total_documents": self.total_documents,
            "total_parent_chunks": self.total_parent_chunks,
            "total_child_chunks": self.total_child_chunks,
            "duration_seconds": round(self.duration_seconds, 2),
            "errors_count": len(self.errors),
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat()
        }
