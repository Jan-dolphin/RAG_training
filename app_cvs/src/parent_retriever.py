"""Parent Document Retriever implementation for CV search with batch processing"""

import logging
import time
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage.file_system import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.stores import BaseStore

from .config import RAGConfig, AzureConfig
from .models import RetrievalResult

logger = logging.getLogger(__name__)


class DocumentStore(BaseStore[str, Document]):
    """
    Wrapper around LocalFileStore to handle Document serialization.

    LocalFileStore expects bytes, but ParentDocumentRetriever works with Documents.
    This wrapper converts Document <-> bytes using JSON.
    """

    def __init__(self, root_path: str):
        """Initialize with a LocalFileStore backend"""
        self._store = LocalFileStore(root_path)

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        """Get Documents by keys"""
        results = []
        for key in keys:
            byte_data = self._store.mget([key])[0]
            if byte_data is None:
                results.append(None)
            else:
                # Deserialize bytes to Document
                data = json.loads(byte_data.decode('utf-8'))
                doc = Document(
                    page_content=data['page_content'],
                    metadata=data['metadata']
                )
                results.append(doc)
        return results

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        """Set Documents by keys"""
        byte_pairs = []
        for key, doc in key_value_pairs:
            # Serialize Document to bytes
            data = {
                'page_content': doc.page_content,
                'metadata': doc.metadata
            }
            byte_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            byte_pairs.append((key, byte_data))
        self._store.mset(byte_pairs)

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete Documents by keys"""
        self._store.mdelete(keys)

    def yield_keys(self, prefix: Optional[str] = None):
        """Yield all keys in the store"""
        return self._store.yield_keys(prefix=prefix)


class CVParentRetriever:
    """
    Parent Document Retriever specifically designed for CV search

    Architecture:
    - Parent: Full CV of candidate (large chunk with all information)
    - Child: Small chunks with specific skills/experience (used for search)
    """

    def __init__(self, config: RAGConfig, vectorstore: Chroma, azure_config: Optional[AzureConfig] = None):
        """
        Initialize CV Parent Retriever

        Args:
            config: RAG configuration with chunk sizes
            vectorstore: ChromaDB vector store instance
            azure_config: Optional Azure config for rate limiting settings
        """
        self.config = config
        self.vectorstore = vectorstore
        self.azure_config = azure_config
        self._retriever = None

        # Setup persistent docstore with Document wrapper
        docstore_path = Path(config.persist_directory) / "docstore"
        docstore_path.mkdir(parents=True, exist_ok=True)
        self.docstore = DocumentStore(str(docstore_path))

        logger.info(
            f"Initializing Parent Retriever - "
            f"Parent chunks: {config.parent_chunk_size}, "
            f"Child chunks: {config.child_chunk_size}"
        )
        logger.info(f"Docstore path: {docstore_path}")

    def _create_splitters(self):
        """Create parent and child text splitters"""
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.parent_chunk_size,
            chunk_overlap=self.config.parent_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.child_chunk_size,
            chunk_overlap=self.config.child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        return parent_splitter, child_splitter

    def initialize_retriever(self, documents: List[Document]) -> None:
        """
        Initialize the parent document retriever with CV documents

        Args:
            documents: List of CV documents (one per candidate)
        """
        logger.info(f"Initializing retriever with {len(documents)} CV documents")

        parent_splitter, child_splitter = self._create_splitters()

        self._retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )

        # Add documents - use batch processing if configured
        if self.azure_config and self.azure_config.batch_size > 0 and len(documents) > self.azure_config.batch_size:
            logger.info(f"Using batch processing for retriever initialization: batch_size={self.azure_config.batch_size}")
            self._add_documents_batched(documents)
        else:
            logger.info("Adding all documents at once")
            self._retriever.add_documents(documents)

        logger.info("Parent retriever initialized successfully")

        # Log statistics
        try:
            child_count = self.vectorstore._collection.count()
            parent_count = len(list(self.docstore.yield_keys()))
            logger.info(f"Created {parent_count} parent chunks and {child_count} child chunks")
        except Exception as e:
            logger.warning(f"Could not retrieve chunk statistics: {e}")

    def _add_documents_batched(self, documents: List[Document]) -> None:
        """
        Add documents in batches based on child chunk count to avoid rate limits

        This method:
        1. Pre-splits all documents into child chunks
        2. Batches child chunks (not parent documents) according to batch_size
        3. Processes each batch with delay to respect rate limits

        Args:
            documents: List of parent documents to add
        """
        batch_size = self.azure_config.batch_size
        batch_delay = self.azure_config.batch_delay

        logger.info(f"Pre-splitting {len(documents)} documents into child chunks...")

        # Get child splitter to pre-calculate chunks
        _, child_splitter = self._create_splitters()

        # Split all documents and track which parent they belong to
        all_child_chunks = []
        for doc in documents:
            child_chunks = child_splitter.split_documents([doc])
            all_child_chunks.extend(child_chunks)

        total_chunks = len(all_child_chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size

        logger.info(f"Total child chunks: {total_chunks}")
        logger.info(f"Processing in {total_batches} batches of ~{batch_size} chunks each")

        # Now process documents in batches based on chunk count
        processed_chunks = 0
        batch_num = 1

        for doc in documents:
            # Split document to child chunks
            child_chunks = child_splitter.split_documents([doc])
            num_chunks = len(child_chunks)

            logger.info(f"Document '{doc.metadata.get('candidate_name', 'Unknown')}': {num_chunks} child chunks")

            # Add this document (which internally creates child chunks)
            try:
                self._retriever.add_documents([doc])
                processed_chunks += num_chunks

                # Check if we should delay after this document
                # Delay when we've processed approximately batch_size chunks
                if processed_chunks >= batch_size * batch_num and processed_chunks < total_chunks:
                    logger.info(f"Processed {processed_chunks}/{total_chunks} chunks ({batch_num} batches)")
                    if batch_delay > 0:
                        logger.info(f"Waiting {batch_delay}s before next batch...")
                        time.sleep(batch_delay)
                    batch_num += 1

            except Exception as e:
                logger.error(f"Error processing document: {e}")
                raise

        logger.info(f"All {total_chunks} child chunks processed successfully in {batch_num} batches")

    def load_from_existing_store(self) -> None:
        """
        Load retriever from existing vector store and docstore.

        The docstore (parent chunks) is automatically loaded from disk via LocalFileStore.
        This method just initializes the ParentDocumentRetriever with the existing stores.
        """
        logger.info("Loading retriever from existing stores")

        parent_splitter, child_splitter = self._create_splitters()

        self._retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )

        # Log statistics
        try:
            child_count = self.vectorstore._collection.count()
            parent_count = len(list(self.docstore.yield_keys()))
            logger.info(f"Loaded retriever with {parent_count} parent chunks and {child_count} child chunks")
        except Exception as e:
            logger.warning(f"Could not retrieve chunk statistics: {e}")

        logger.info("Retriever loaded successfully")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant CV documents for query

        Args:
            query: Search query (e.g., "candidates with Python skills")
            top_k: Number of results to return (uses config default if None)

        Returns:
            List of parent documents (complete CV chunks)
        """
        if self._retriever is None:
            raise ValueError("Retriever not initialized. Call initialize_retriever or load_from_existing_store first.")

        k = top_k or self.config.top_k

        logger.info(f"Retrieving documents for query: '{query}' (top {k})")

        # Use ParentDocumentRetriever to get parent chunks
        results = self._retriever.invoke(query)

        # Limit to top_k
        results = results[:k]

        logger.info(f"Retrieved {len(results)} parent documents")

        return results

    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve documents with similarity scores

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects with scores
        """
        k = top_k or self.config.top_k

        # Use vectorstore directly for similarity search with scores
        raw_results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Map child chunks back to parent documents
        retrieval_results = []

        for doc, score in raw_results:
            # Get parent document from docstore if available
            parent_doc = doc  # If no parent mapping, use the doc itself

            result = RetrievalResult(
                candidate_name=doc.metadata.get("candidate_name", "Unknown"),
                content=parent_doc.page_content,
                score=float(score),
                metadata=parent_doc.metadata
            )
            retrieval_results.append(result)

        logger.info(f"Retrieved {len(retrieval_results)} results with scores")

        return retrieval_results

    def retrieve_relevant(self, query: str, top_k: Optional[int] = None,
                         threshold: Optional[float] = None) -> List[Document]:
        """
        Retrieve only relevant documents based on similarity threshold

        Filters out results with similarity score above threshold.
        Lower score = higher similarity (distance metric).

        Args:
            query: Search query
            top_k: Number of results to return (before filtering)
            threshold: Maximum similarity score (uses config default if None)

        Returns:
            List of relevant parent documents (may be empty if no relevant matches)
        """
        k = top_k or self.config.top_k
        max_score = threshold if threshold is not None else self.config.similarity_threshold

        logger.info(f"Retrieving relevant documents for query: '{query}' (top {k}, threshold {max_score})")

        # Get results with scores
        # Fetch more results to have options after filtering
        raw_results = self.vectorstore.similarity_search_with_score(query, k=k * 2)

        # Filter by threshold
        relevant_child_docs = []
        filtered_count = 0

        for doc, score in raw_results:
            if score <= max_score:
                relevant_child_docs.append(doc)
            else:
                filtered_count += 1

        # Limit to top_k after filtering
        relevant_child_docs = relevant_child_docs[:k]

        # Map child chunks to parent documents using ParentDocumentRetriever
        if relevant_child_docs:
            # Get parent documents for relevant child chunks
            parent_ids = set()
            parent_docs = []

            for doc in relevant_child_docs:
                parent_id = doc.metadata.get(self._retriever.id_key, None)
                if parent_id and parent_id not in parent_ids:
                    parent_ids.add(parent_id)
                    # Load parent from docstore
                    parent_list = self.docstore.mget([parent_id])
                    if parent_list and parent_list[0]:
                        parent_docs.append(parent_list[0])

            logger.info(f"Retrieved {len(parent_docs)} relevant parent documents "
                       f"(filtered out {filtered_count} irrelevant results)")

            return parent_docs[:k]
        else:
            logger.info(f"No relevant documents found (all {filtered_count} results below threshold)")
            return []

    def get_stats(self) -> dict:
        """
        Get retriever statistics

        Returns:
            Dictionary with statistics
        """
        try:
            child_count = self.vectorstore._collection.count()
            parent_count = len(list(self.docstore.yield_keys()))

            return {
                "status": "initialized" if self._retriever else "not_initialized",
                "parent_chunks": parent_count,
                "child_chunks": child_count,
                "parent_chunk_size": self.config.parent_chunk_size,
                "child_chunk_size": self.config.child_chunk_size
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}
