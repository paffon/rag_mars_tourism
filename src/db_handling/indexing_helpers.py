# src/db_handling/indexing_helpers.py
from typing import List, Tuple

# Make sure Settings is imported directly
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from chromadb.api.models.Collection import Collection as ChromaCollection
from chromadb.api.client import ClientAPI as ChromaClientAPI

from src import config
from src.logger.logger import MyLogger
# Import updated parsing, hashing, and chroma_utils
from src.db_handling import chroma_utils

logger = MyLogger(config.LOG_NAME)

# --- LlamaIndex Global Settings Configuration ---

def _setup_llm():
    """Sets up the Language Model (LLM) in LlamaIndex global Settings."""
    logger.info("Configuring LlamaIndex LLM settings...")
    try:
        Settings.llm = OpenAI(
            model=config.OPENAI_MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            system_prompt=config.SYSTEM_PROMPT
        )
        logger.info(f"LLM configured successfully: Model={config.OPENAI_MODEL_NAME}")
    except Exception as e:
        logger.critical(f"Failed to configure LLM: {e}", exc_info=True)
        raise

def _setup_embed_model():
    """Sets up the Embedding Model in LlamaIndex global Settings."""
    logger.info("Configuring LlamaIndex Embedding Model settings...")
    try:
        # Ensure Settings is accessible for assignment
        embed_model = OpenAIEmbedding(
            model=config.EMBEDDING_MODEL_NAME,
            api_key=config.OPENAI_API_KEY
        )
        # Directly assign to the global Settings object
        Settings.embed_model = embed_model
        # Removed the line: config.Settings = Settings
        logger.info(f"Embedding model configured successfully: Model={config.EMBEDDING_MODEL_NAME}")

    except Exception as e:
        logger.critical(f"Failed to configure Embedding Model: {e}", exc_info=True)
        raise

def setup_llama_index_settings():
    """Configures LlamaIndex global settings for LLM and Embedding Model."""
    ACTION = "Configure LlamaIndex Settings"
    logger.start(ACTION)
    try:
        _setup_llm()
        _setup_embed_model()
        logger.info("LlamaIndex LLM and Embedding Model settings configured successfully.")
    except Exception as e:
        logger.critical(f"Critical failure during LlamaIndex settings configuration.", exc_info=False)
        raise
    finally:
        logger.close(ACTION)


# --- Combined Initialization ---

def initialize_db_and_index() -> Tuple[ChromaClientAPI, ChromaCollection, VectorStoreIndex, StorageContext]:
    """
    Initializes all necessary components: ChromaDB client, collection, and LlamaIndex Index/StorageContext.
    """
    ACTION = "Initialize Full DB and Index Structure"
    logger.start(ACTION)
    client, collection, index, storage_context = None, None, None, None # Initialize
    try:
        client = chroma_utils.get_chroma_client()
        collection = chroma_utils.get_or_create_chroma_collection(client)
        # LlamaIndex settings (like embed_model) must be configured *before* this step
        # setup_llama_index_settings() should be called before this in the main script flow
        index, storage_context = chroma_utils.get_index_and_storage_context(collection)
        logger.info("Full DB and Index Structure (Client, Collection, Index, StorageContext) initialized successfully.")
    except Exception as e:
         logger.critical(f"Failure during full DB/Index initialization wrapper: {e}", exc_info=True)
         raise
    finally:
        logger.close(ACTION)

    if not all([client, collection, index, storage_context]):
         logger.critical("DB/Index initialization failed to return all required components.")
         raise RuntimeError("DB/Index initialization failed to return all required components.")
    return client, collection, index, storage_context


# --- Index Interaction ---

def insert_documents_to_index(
    index: VectorStoreIndex, documents: List[Document]
) -> bool:
    """
    Inserts a list of LlamaIndex Documents into the specified index.
    Returns True on success, False on failure.
    """
    if not documents:
        logger.info("Document Insertion: No documents provided to insert.")
        return True # Nothing to do, success

    success = False
    num_docs = len(documents)
    logger.info(f"Starting insertion of {num_docs} new document nodes into the LlamaIndex index...")
    try:
        index.insert_nodes(documents)
        logger.info(f"Successfully inserted {num_docs} document nodes.")
        success = True
    except Exception as e:
        logger.error(f"Failed to insert documents into LlamaIndex index: {e}", exc_info=True)
        success = False
    return success
