"""RAG Chain implementation using LCEL (LangChain Expression Language)"""

import logging
from typing import List, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .config import AzureConfig
from .parent_retriever import CVParentRetriever
from .models import RAGResponse, RetrievalResult, Query

logger = logging.getLogger(__name__)


class CVRAGChain:
    """
    RAG Chain for CV question answering

    Uses LCEL pattern: retriever -> context formatting -> prompt -> LLM -> parser
    """

    DEFAULT_PROMPT_TEMPLATE = """You are an expert HR assistant helping to find candidates based on their CVs.

Use ONLY the following CV excerpts to answer the question. If you cannot find the answer in the provided context, say "I don't have enough information to answer this question."

Context from CVs:
{context}

Question: {question}

Answer in a clear and concise manner. When mentioning candidates, always include their names."""

    def __init__(self, config: AzureConfig, retriever: BaseRetriever, prompt_template: Optional[str] = None):
        """
        Initialize RAG Chain

        Args:
            config: Azure configuration
            retriever: LangChain BaseRetriever (or compatible interface)
            prompt_template: Custom prompt template (uses default if None)
        """
        self.config = config
        self.retriever = retriever
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

        logger.info("Initializing RAG Chain")
        self._llm = self._create_llm()
        self._chain = self._create_chain()

    def _create_llm(self) -> AzureChatOpenAI:
        """Create Azure ChatOpenAI instance"""
        logger.info(f"Creating LLM with deployment: {self.config.llm_deployment}")

        return AzureChatOpenAI(
            azure_deployment=self.config.llm_deployment,
            openai_api_version=self.config.api_version,
            azure_endpoint=self.config.endpoint,
            api_key=self.config.api_key,
            temperature=self.config.temperature
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """
        Format retrieved documents into context string

        Args:
            docs: List of documents

        Returns:
            Formatted context string
        """
        formatted = []

        for i, doc in enumerate(docs, 1):
            candidate_name = doc.metadata.get("candidate_name", "Unknown")
            content = doc.page_content

            formatted.append(f"--- CV #{i}: {candidate_name} ---\n{content}")

        return "\n\n".join(formatted)

    def _create_chain(self):
        """
        Create LCEL chain: retriever -> format -> prompt -> LLM -> parser

        Returns:
            Runnable chain
        """
        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        # Note: self.retriever must support Runnable interface (invoke)
        chain = (
            {
                "context": lambda x: self._format_docs(self.retriever.invoke(x)),
                "question": RunnablePassthrough()
            }
            | prompt
            | self._llm
            | StrOutputParser()
        )

        logger.info("RAG Chain created successfully")
        return chain

    def invoke(self, query: str, use_relevance_filter: bool = True) -> RAGResponse:
        """
        Execute RAG chain with query

        Args:
            query: User question about CVs
            use_relevance_filter: Deprecated/Ignored. Filtering is now handled by the retriever strategy.

        Returns:
            RAGResponse with answer and retrieved contexts
        """
        logger.info(f"Processing query: '{query}'")

        # Retrieve documents using standard invoke
        retrieved_docs = self.retriever.invoke(query)

        # Check if we have relevant results
        if not retrieved_docs:
            logger.warning("No relevant documents found for query")
            return RAGResponse(
                query=query,
                answer="I couldn't find any candidates matching your criteria. Please try a different search or broaden your requirements.",
                retrieved_contexts=[],
                metadata={
                    "num_contexts": 0,
                    "llm_model": self.config.llm_deployment,
                    "no_relevant_results": True
                }
            )

        # Convert to RetrievalResult objects
        retrieved_contexts = [
            RetrievalResult(
                candidate_name=doc.metadata.get("candidate_name", "Unknown"),
                content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in retrieved_docs
        ]

        logger.info(f"Retrieved {len(retrieved_contexts)} contexts")

        # Generate answer
        answer = self._chain.invoke(query)

        logger.info("Answer generated successfully")

        # Create response
        response = RAGResponse(
            query=query,
            answer=answer,
            retrieved_contexts=retrieved_contexts,
            metadata={
                "num_contexts": len(retrieved_contexts),
                "llm_model": self.config.llm_deployment,
                "temperature": self.config.temperature,
                "relevance_filter": use_relevance_filter
            }
        )

        return response

    def batch_invoke(self, queries: List[str]) -> List[RAGResponse]:
        """
        Execute RAG chain for multiple queries

        Args:
            queries: List of user questions

        Returns:
            List of RAGResponse objects
        """
        logger.info(f"Processing batch of {len(queries)} queries")

        responses = []
        for query in queries:
            try:
                response = self.invoke(query)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                # Create error response
                responses.append(RAGResponse(
                    query=query,
                    answer=f"Error processing query: {str(e)}",
                    retrieved_contexts=[],
                    metadata={"error": str(e)}
                ))

        logger.info(f"Batch processing complete: {len(responses)} responses")
        return responses

    def update_prompt(self, new_template: str) -> None:
        """
        Update prompt template and recreate chain

        Args:
            new_template: New prompt template string
        """
        logger.info("Updating prompt template")
        self.prompt_template = new_template
        self._chain = self._create_chain()
        logger.info("Chain recreated with new prompt")

    def get_llm(self) -> AzureChatOpenAI:
        """Get LLM instance"""
        return self._llm
