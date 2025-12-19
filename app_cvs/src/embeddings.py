"""Azure OpenAI Embeddings wrapper with rate limit handling"""

import logging
import time
from typing import List
from langchain_openai import AzureOpenAIEmbeddings
from openai import RateLimitError

from .config import AzureConfig

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """Manages Azure OpenAI embeddings"""

    def __init__(self, config: AzureConfig):
        """
        Initialize embeddings manager

        Args:
            config: Azure configuration
        """
        self.config = config
        self._embeddings = None

    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry logic

        Args:
            func: Function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Result of the function

        Raises:
            Last exception if all retries fail
        """
        delay = self.config.retry_delay
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{self.config.max_retries}). "
                        f"Waiting {delay:.1f}s before retry..."
                    )
                    time.sleep(delay)
                    # Exponential backoff with max limit
                    delay = min(delay * 2, self.config.max_retry_delay)
                else:
                    logger.error(f"All {self.config.max_retries} retry attempts failed")
            except Exception as e:
                # For non-rate-limit errors, fail immediately
                logger.error(f"Non-retryable error: {e}")
                raise

        raise last_exception

    def get_embeddings(self) -> AzureOpenAIEmbeddings:
        """
        Get or create embeddings instance (singleton pattern)

        Returns:
            AzureOpenAIEmbeddings instance
        """
        if self._embeddings is None:
            logger.info(f"Initializing Azure OpenAI Embeddings with deployment: {self.config.embedding_deployment}")

            self._embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=self.config.endpoint,
                api_key=self.config.api_key,
                azure_deployment=self.config.embedding_deployment,
                openai_api_version=self.config.api_version,
                max_retries=self.config.max_retries
            )

            logger.info("Embeddings initialized successfully")

        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text with retry logic

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        embeddings = self.get_embeddings()
        return self._retry_with_backoff(embeddings.embed_query, text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents with retry logic

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.get_embeddings()
        return self._retry_with_backoff(embeddings.embed_documents, texts)

    def test_connection(self) -> bool:
        """
        Test embeddings connection with a simple query

        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_vector = self.embed_query("test connection")

            if len(test_vector) != 1536:  # Ada-002 embedding dimension
                logger.warning(f"Unexpected embedding dimension: {len(test_vector)}")
                return False

            logger.info("Embeddings connection test successful")
            return True

        except Exception as e:
            logger.error(f"Embeddings connection test failed: {e}")
            return False
