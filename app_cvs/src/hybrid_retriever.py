"""Hybrid Retriever combining BM25 (keyword) and semantic (embedding) search"""

import logging
from typing import List, Optional, Dict
from collections import defaultdict
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from .config import RAGConfig

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining BM25 keyword search with semantic embedding search

    Uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers.

    Architecture:
    1. BM25 Retriever: Exact keyword matching (perfect for "React", "SQL", etc.)
    2. Embedding Retriever: Semantic similarity (catches "PostgreSQL" for "SQL database")
    3. RRF Fusion: Combines both result sets with configurable weights
    """

    def __init__(
        self,
        config: RAGConfig,
        vectorstore: Chroma,
        documents: List[Document]
    ):
        """
        Initialize hybrid retriever

        Args:
            config: RAG configuration with hybrid search settings
            vectorstore: ChromaDB vector store for semantic search
            documents: All documents for BM25 index (parent chunks)
        """
        self.config = config
        self.vectorstore = vectorstore

        logger.info("Initializing Hybrid Retriever (BM25 + Embeddings)")

        # Create BM25 retriever for keyword matching
        logger.info(f"Creating BM25 retriever with {len(documents)} documents")
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=config.bm25_k
        )

        # Create embedding retriever from vectorstore
        logger.info(f"Creating embedding retriever (top_k={config.embedding_k})")
        self.embedding_retriever = vectorstore.as_retriever(
            search_kwargs={"k": config.embedding_k}
        )

        logger.info(
            f"Hybrid Retriever initialized: "
            f"BM25 weight={config.bm25_weight}, "
            f"Embedding weight={config.embedding_weight}"
        )

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Document],
        embedding_results: List[Document],
        k: int = 60
    ) -> List[Document]:
        """
        Perform Reciprocal Rank Fusion (RRF) on two result lists

        RRF formula: score(d) = sum(1 / (k + rank(d)))
        where k=60 is a constant to prevent division by zero

        Args:
            bm25_results: Results from BM25 retriever
            embedding_results: Results from embedding retriever
            k: RRF constant (default 60)

        Returns:
            Fused and sorted list of documents
        """
        # Calculate RRF scores
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        # Add BM25 scores (weighted)
        for rank, doc in enumerate(bm25_results):
            doc_id = id(doc)  # Use Python object ID as unique identifier
            doc_key = f"{doc.metadata.get('candidate_name', '')}_{doc.page_content[:50]}"
            rrf_scores[doc_key] += self.config.bm25_weight / (k + rank + 1)
            doc_map[doc_key] = doc

        # Add embedding scores (weighted)
        for rank, doc in enumerate(embedding_results):
            doc_key = f"{doc.metadata.get('candidate_name', '')}_{doc.page_content[:50]}"
            rrf_scores[doc_key] += self.config.embedding_weight / (k + rank + 1)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc

        # Sort by RRF score (descending)
        sorted_doc_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Return documents in sorted order
        fused_results = [doc_map[doc_key] for doc_key in sorted_doc_keys]

        return fused_results

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve documents using hybrid search (BM25 + embeddings) with RRF fusion

        Args:
            query: Search query
            top_k: Number of final results to return (uses config default if None)

        Returns:
            List of documents sorted by RRF score
        """
        k = top_k or self.config.top_k

        logger.info(f"Hybrid search for query: '{query}' (top {k})")

        # Get results from both retrievers
        bm25_results = self.bm25_retriever.invoke(query)
        embedding_results = self.embedding_retriever.invoke(query)

        logger.info(f"BM25 returned {len(bm25_results)} results, Embeddings returned {len(embedding_results)} results")

        # Fuse results using RRF
        fused_results = self._reciprocal_rank_fusion(bm25_results, embedding_results)

        # Limit to top_k
        final_results = fused_results[:k]

        logger.info(f"Hybrid search returned {len(final_results)} fused documents")

        return final_results

    def retrieve_with_method_breakdown(self, query: str, top_k: Optional[int] = None) -> dict:
        """
        Retrieve with breakdown showing which results came from which method

        Useful for debugging and understanding hybrid search behavior.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Dictionary with:
                - final_results: Fused results
                - bm25_results: Results from BM25 only
                - embedding_results: Results from embeddings only
                - overlap: Documents found by both methods
        """
        k = top_k or self.config.top_k

        # Get BM25 results
        bm25_results = self.bm25_retriever.invoke(query)
        bm25_ids = {doc.metadata.get('doc_id', id(doc)) for doc in bm25_results}

        # Get embedding results
        embedding_results = self.embedding_retriever.invoke(query)
        embedding_ids = {doc.metadata.get('doc_id', id(doc)) for doc in embedding_results}

        # Get fused results
        fused_results = self._reciprocal_rank_fusion(bm25_results, embedding_results)
        final_results = fused_results[:k]

        # Calculate overlap
        overlap_ids = bm25_ids.intersection(embedding_ids)

        logger.info(
            f"Breakdown: BM25={len(bm25_results)}, "
            f"Embeddings={len(embedding_results)}, "
            f"Overlap={len(overlap_ids)}, "
            f"Final={len(final_results)}"
        )

        return {
            "final_results": final_results,
            "bm25_results": bm25_results,
            "embedding_results": embedding_results,
            "overlap_count": len(overlap_ids),
            "bm25_count": len(bm25_results),
            "embedding_count": len(embedding_results),
            "final_count": len(final_results)
        }

    def update_bm25_documents(self, documents: List[Document]) -> None:
        """
        Update BM25 index with new documents

        Call this after adding new CVs to the system.

        Args:
            documents: New list of all documents
        """
        logger.info(f"Updating BM25 index with {len(documents)} documents")
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=self.config.bm25_k
        )
        logger.info("BM25 index updated")

    def get_stats(self) -> dict:
        """Get retriever statistics"""
        return {
            "type": "hybrid",
            "bm25_k": self.config.bm25_k,
            "embedding_k": self.config.embedding_k,
            "bm25_weight": self.config.bm25_weight,
            "embedding_weight": self.config.embedding_weight,
            "top_k": self.config.top_k
        }
