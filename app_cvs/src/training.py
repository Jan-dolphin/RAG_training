"""Training module for RAG pipeline with comprehensive logging"""

import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from .config import AppConfig
from .document_loader import CVDocumentLoader
from .embeddings import EmbeddingsManager
from .vector_store import VectorStoreManager
from .parent_retriever import CVParentRetriever
from .models import TrainingMetrics
from .retriever_factory import RetrieverFactory
from langchain_openai import AzureChatOpenAI
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline for RAG system

    Handles:
    - Document loading from DOCX files
    - Vector store creation
    - Parent document retriever initialization
    - Comprehensive logging and metrics
    """

    def __init__(self, config: AppConfig, log_file: Optional[str] = None):
        """
        Initialize training pipeline

        Args:
            config: Application configuration
            log_file: Path to log file for training logs (optional)
        """
        self.config = config
        self.metrics = TrainingMetrics()
        self.log_file = log_file

        if self.log_file:
            self._setup_file_logging(log_file)

        logger.info("=" * 80)
        logger.info("Training Pipeline Initialized")
        logger.info("=" * 80)

    def _setup_file_logging(self, log_file: str) -> None:
        """Setup file handler for logging"""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # Add to root logger
        logging.getLogger().addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    def _log_config(self) -> None:
        """Log current configuration"""
        logger.info("\n" + "=" * 80)
        logger.info("CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Data Directory: {self.config.rag.data_directory}")
        logger.info(f"Vector Store: {self.config.rag.persist_directory}")
        logger.info(f"Collection Name: {self.config.rag.collection_name}")
        logger.info(f"Parent Chunk Size: {self.config.rag.parent_chunk_size}")
        logger.info(f"Child Chunk Size: {self.config.rag.child_chunk_size}")
        logger.info(f"Azure Endpoint: {self.config.azure.endpoint}")
        logger.info(f"Embedding Model: {self.config.azure.embedding_deployment}")
        logger.info("=" * 80 + "\n")

    def load_documents(self) -> CVDocumentLoader:
        """
        Load CV documents from data directory

        Returns:
            CVDocumentLoader with loaded documents
        """
        logger.info("\n" + "-" * 80)
        logger.info("STEP 1: Loading Documents")
        logger.info("-" * 80)

        try:
            loader = CVDocumentLoader(self.config.rag.data_directory)
            candidates = loader.load_all_cvs()

            self.metrics.total_documents = len(candidates)

            logger.info(f"Successfully loaded {len(candidates)} CV documents")

            for i, candidate in enumerate(candidates[:5], 1):  # Log first 5
                logger.info(f"  {i}. {candidate.name} ({len(candidate.full_cv_text)} chars)")

            if len(candidates) > 5:
                logger.info(f"  ... and {len(candidates) - 5} more")

            return loader

        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            self.metrics.add_error(f"Document loading failed: {str(e)}")
            raise

    def setup_embeddings(self) -> EmbeddingsManager:
        """
        Setup and test embeddings

        Returns:
            EmbeddingsManager instance
        """
        logger.info("\n" + "-" * 80)
        logger.info("STEP 2: Setting Up Embeddings")
        logger.info("-" * 80)

        try:
            embeddings_mgr = EmbeddingsManager(self.config.azure)

            logger.info("Testing embeddings connection...")
            if embeddings_mgr.test_connection():
                logger.info("✓ Embeddings connection successful")
            else:
                logger.warning("✗ Embeddings connection test failed")
                self.metrics.add_error("Embeddings connection test failed")

            return embeddings_mgr

        except Exception as e:
            logger.error(f"Failed to setup embeddings: {e}")
            self.metrics.add_error(f"Embeddings setup failed: {str(e)}")
            raise

    def setup_vector_store(self, embeddings_mgr: EmbeddingsManager, clear_existing: bool = True) -> VectorStoreManager:
        """
        Setup vector store (create empty or load existing)

        Args:
            embeddings_mgr: Embeddings manager
            clear_existing: Whether to clear existing vectorstore

        Returns:
            VectorStoreManager instance
        """
        logger.info("\n" + "-" * 80)
        logger.info("STEP 3: Setting Up Vector Store")
        logger.info("-" * 80)

        try:
            vs_manager = VectorStoreManager(
                self.config.rag,
                embeddings_mgr.get_embeddings()
            )

            # Clear existing store if requested
            if clear_existing:
                logger.info("Clearing existing vector store...")
                vs_manager.clear_vectorstore()

            logger.info("Creating empty vector store...")
            start_time = time.time()

            vs_manager.create_or_load_vectorstore()

            elapsed = time.time() - start_time
            logger.info(f"✓ Vector store ready in {elapsed:.2f} seconds")

            return vs_manager

        except Exception as e:
            logger.error(f"Failed to setup vector store: {e}")
            self.metrics.add_error(f"Vector store setup failed: {str(e)}")
            raise

    def initialize_retriever(
        self,
        loader: CVDocumentLoader,
        vs_manager: VectorStoreManager
    ) -> CVParentRetriever:
        """
        Initialize Parent Document Retriever

        Args:
            loader: Document loader
            vs_manager: Vector store manager

        Returns:
            CVParentRetriever instance
        """
        logger.info("\n" + "-" * 80)
        logger.info("STEP 4: Initializing Parent Document Retriever")
        logger.info("-" * 80)

        try:
            # Pass azure_config for rate limiting settings
            retriever = CVParentRetriever(
                config=self.config.rag,
                vectorstore=vs_manager.get_vectorstore(),
                azure_config=self.config.azure
            )

            # Load and convert documents
            candidates = loader.load_all_cvs()
            documents = loader.convert_to_langchain_documents(candidates)

            logger.info(f"Initializing retriever with {len(documents)} documents...")
            logger.info(f"Rate limiting settings: batch_size={self.config.azure.batch_size}, batch_delay={self.config.azure.batch_delay}s")
            start_time = time.time()

            retriever.initialize_retriever(documents)

            elapsed = time.time() - start_time
            logger.info(f"✓ Retriever initialized in {elapsed:.2f} seconds")

            # Get stats
            stats = retriever.get_stats()
            self.metrics.total_parent_chunks = stats.get("parent_chunks", 0)
            self.metrics.total_child_chunks = stats.get("child_chunks", 0)

            logger.info(f"Retriever stats: {stats}")
            logger.info(f"  - Parent chunks: {self.metrics.total_parent_chunks}")
            logger.info(f"  - Child chunks: {self.metrics.total_child_chunks}")

            return retriever

        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            self.metrics.add_error(f"Retriever initialization failed: {str(e)}")
            raise

    def test_retrieval(self, retriever: BaseRetriever, test_queries: Optional[list] = None) -> None:
        """
        Test retrieval with sample queries using the configured strategy

        Args:
            retriever: Initialized retriever (BaseRetriever compatible)
            test_queries: Optional list of test queries
        """
        logger.info("\n" + "-" * 80)
        logger.info("STEP 5: Testing Retrieval (Strategy Check)")
        logger.info("-" * 80)

        if test_queries is None:
            test_queries = [
                "candidates with Python skills",
                "who has AWS experience",
                "Java developers"
            ]

        for query in test_queries:
            try:
                logger.info(f"\nTest Query: '{query}'")

                # Use standard invoke
                results = retriever.invoke(query)
                # Limit manually if retriever doesn't support top_k arg directly in invoke
                # (Factory retrievers usually configured with k)
                
                logger.info(f"  Retrieved {len(results)} results:")
                for i, doc in enumerate(results[:3], 1): # Show top 3
                    candidate_name = doc.metadata.get("candidate_name", "Unknown")
                    content_preview = doc.page_content[:100].replace("\n", " ")
                    logger.info(f"    {i}. {candidate_name}")
                    logger.info(f"       Preview: {content_preview}...")

            except Exception as e:
                logger.error(f"Test query failed: {e}")
                self.metrics.add_error(f"Test query '{query}' failed: {str(e)}")

    def save_metrics(self, output_file: Optional[str] = None) -> None:
        """
        Save training metrics to JSON file

        Args:
            output_file: Path to output file (defaults to training_metrics.json)
        """
        if output_file is None:
            output_file = "training_metrics.json"

        metrics_dict = self.metrics.to_dict()

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"\nMetrics saved to: {output_file}")

    def run_full_pipeline(
        self,
        test_queries: Optional[list] = None,
        save_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline

        Args:
            test_queries: Optional test queries for retrieval testing
            save_metrics: Whether to save metrics to file

        Returns:
            Dictionary with pipeline results
        """
        start_time = time.time()

        logger.info("\n" + "=" * 80)
        logger.info("STARTING FULL TRAINING PIPELINE")
        logger.info("=" * 80)

        self._log_config()

        try:
            # Step 1: Load documents
            loader = self.load_documents()

            # Step 2: Setup embeddings
            embeddings_mgr = self.setup_embeddings()

            # Step 3: Setup vector store (empty)
            vs_manager = self.setup_vector_store(embeddings_mgr, clear_existing=True)

            # Step 4: Initialize retriever (Index Builder) and populate vectorstore
            # This uses CVParentRetriever solely for handling the complex Parent/Child indexing logic
            index_builder = self.initialize_retriever(loader, vs_manager)

            # Step 5: Initialize Retrieval Strategy for Testing
            # (This is what we use for actual queries)
            llm = None
            # Step 5: Initialize Retrieval Strategy for Testing
            # (This is what we use for actual queries)
            llm = None
            # Check if any LLM-dependent strategy is enabled
            needs_llm = (self.config.rag.use_multi_query or 
                        self.config.rag.use_contextual_compression or 
                        self.config.rag.use_self_query)
            
            if needs_llm:
                 # Initialize LLM only if needed for strategy (optimization)
                 logger.info("Initializing LLM for advanced retrieval strategy...")
                 llm = AzureChatOpenAI(
                    azure_deployment=self.config.azure.llm_deployment,
                    openai_api_version=self.config.azure.api_version,
                    azure_endpoint=self.config.azure.endpoint,
                    api_key=self.config.azure.api_key,
                    temperature=0
                )

            testing_retriever = RetrieverFactory.create_retriever(
                self.config,
                vs_manager.get_vectorstore(),
                llm=llm,
                parent_retriever=index_builder
            )

            # Step 6: Test retrieval
            self.test_retrieval(testing_retriever, test_queries)

            # Calculate duration
            self.metrics.duration_seconds = time.time() - start_time

            logger.info("\n" + "=" * 80)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Total Duration: {self.metrics.duration_seconds:.2f} seconds")
            logger.info(f"Documents Processed: {self.metrics.total_documents}")
            logger.info(f"Parent Chunks: {self.metrics.total_parent_chunks}")
            logger.info(f"Child Chunks: {self.metrics.total_child_chunks}")
            logger.info(f"Errors: {len(self.metrics.errors)}")
            
            # Log active strategies
            active_strategies = []
            if self.config.rag.use_parent_document_retrieval: active_strategies.append("ParentDocument")
            else: active_strategies.append("Vector")
            if self.config.rag.use_hybrid_search: active_strategies.append("Hybrid")
            if self.config.rag.use_multi_query: active_strategies.append("MultiQuery")
            if self.config.rag.use_contextual_compression: active_strategies.append("Compression")
            
            logger.info(f"Active Strategy Chain: {' -> '.join(active_strategies)}")

            if save_metrics:
                self.save_metrics()

            return {
                "status": "success",
                "metrics": self.metrics.to_dict(),
                "loader": loader,
                "embeddings_manager": embeddings_mgr,
                "vector_store_manager": vs_manager,
                "retriever": testing_retriever # Return the configured strategy retriever
            }

        except Exception as e:
            self.metrics.duration_seconds = time.time() - start_time
            self.metrics.add_error(f"Pipeline failed: {str(e)}")

            logger.error("\n" + "=" * 80)
            logger.error("TRAINING PIPELINE FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {e}")
            logger.error(f"Duration: {self.metrics.duration_seconds:.2f} seconds")

            if save_metrics:
                self.save_metrics()

            raise
