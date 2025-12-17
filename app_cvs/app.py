"""Chainlit frontend for CV RAG Application"""

import sys
import logging
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Select, Slider

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.embeddings import EmbeddingsManager
from src.vector_store import VectorStoreManager
from src.parent_retriever import CVParentRetriever
from src.rag_chain import CVRAGChain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for application state
config = None
rag_chain = None


@cl.on_chat_start
async def start():
    """Initialize application when chat starts"""
    global config, rag_chain

    await cl.Message(content="🚀 Initializing CV Search Assistant...").send()

    try:
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded")

        # Initialize embeddings
        await cl.Message(content="📊 Loading embeddings...").send()
        embeddings_mgr = EmbeddingsManager(config.azure)

        # Load vector store (no need for azure_config when only loading)
        await cl.Message(content="💾 Loading vector store...").send()
        vs_manager = VectorStoreManager(config.rag, embeddings_mgr.get_embeddings())
        vectorstore = vs_manager.load_vectorstore()

        if vectorstore is None:
            await cl.Message(
                content="❌ **Vector store not found!**\n\n"
                        "Please run training first:\n"
                        "```bash\n"
                        "python train.py\n"
                        "```"
            ).send()
            return

        # Get vector store stats
        stats = vs_manager.get_stats()
        await cl.Message(
            content=f"✅ Vector store loaded: {stats['document_count']} documents indexed"
        ).send()

        # Initialize retriever and load from existing stores
        await cl.Message(content="🔍 Initializing retriever...").send()
        retriever = CVParentRetriever(config.rag, vectorstore, config.azure)

        # Load from existing stores (vectorstore + docstore)
        # No need to load documents again - they're already persisted!
        retriever.load_from_existing_store()

        # Get stats
        retriever_stats = retriever.get_stats()
        await cl.Message(
            content=f"✅ Retriever initialized: {retriever_stats['parent_chunks']} parent chunks, "
                    f"{retriever_stats['child_chunks']} child chunks"
        ).send()

        # Create RAG chain
        await cl.Message(content="⚙️ Creating RAG chain...").send()
        rag_chain = CVRAGChain(config.azure, retriever)

        # Store in user session
        cl.user_session.set("rag_chain", rag_chain)
        cl.user_session.set("retriever", retriever)

        # Success message with instructions
        await cl.Message(
            content="✨ **CV Search Assistant Ready!**\n\n"
                    "I can help you find candidates based on their CVs.\n\n"
                    "**Example queries:**\n"
                    "- Who has Python and AWS experience?\n"
                    "- Find candidates with Java skills\n"
                    "- Which candidates know Docker?\n"
                    "- Show me candidates with machine learning background\n\n"
                    "Ask me anything about the CVs!"
        ).send()

        # Setup settings
        settings = await cl.ChatSettings(
            [
                Slider(
                    id="top_k",
                    label="Number of CVs to retrieve",
                    initial=5,
                    min=1,
                    max=10,
                    step=1,
                ),
                Select(
                    id="temperature",
                    label="Response creativity",
                    values=["0.0", "0.3", "0.7"],
                    initial_value="0.0",
                ),
            ]
        ).send()

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        await cl.Message(
            content=f"❌ **Initialization failed:**\n\n```\n{str(e)}\n```\n\n"
                    "Please check the logs and configuration."
        ).send()


@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates"""
    logger.info(f"Settings updated: {settings}")

    rag_chain = cl.user_session.get("rag_chain")
    retriever = cl.user_session.get("retriever")

    if rag_chain and retriever:
        # Update top_k in config
        config.rag.top_k = settings["top_k"]

        # Update temperature
        config.azure.temperature = float(settings["temperature"])

        # Recreate RAG chain with new settings
        new_rag_chain = CVRAGChain(config.azure, retriever)
        cl.user_session.set("rag_chain", new_rag_chain)

        await cl.Message(content=f"⚙️ Settings updated: retrieving {settings['top_k']} CVs").send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages"""
    rag_chain = cl.user_session.get("rag_chain")

    if rag_chain is None:
        await cl.Message(
            content="❌ RAG chain not initialized. Please restart the chat."
        ).send()
        return

    # Show loading message
    msg = cl.Message(content="")
    await msg.send()

    try:
        # Get user query
        query = message.content

        logger.info(f"Processing query: {query}")

        # Execute RAG chain with relevance filtering
        response = rag_chain.invoke(query, use_relevance_filter=True)

        # Format response with sources
        answer_parts = [f"**Answer:**\n\n{response.answer}"]

        # Add sources
        if response.retrieved_contexts:
            answer_parts.append("\n\n---\n\n**📋 Retrieved CVs:**\n")

            for i, ctx in enumerate(response.retrieved_contexts, 1):
                candidate_name = ctx.candidate_name
                content_preview = ctx.content[:200].replace("\n", " ") + "..."

                answer_parts.append(
                    f"\n**{i}. {candidate_name}**\n"
                    f"_{content_preview}_\n"
                )
        elif response.metadata.get('no_relevant_results'):
            # No relevant results found
            answer_parts.append(
                "\n\n---\n\n"
                "💡 **Tip:** Try different keywords or broader criteria."
            )

        # Send formatted response
        msg.content = "".join(answer_parts)
        await msg.update()

        logger.info(f"Response sent for query: {query}")

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        msg.content = f"❌ **Error processing query:**\n\n```\n{str(e)}\n```"
        await msg.update()


if __name__ == "__main__":
    # This block is for IDE compatibility, actual execution is via chainlit CLI
    print("Run with: chainlit run app.py")
