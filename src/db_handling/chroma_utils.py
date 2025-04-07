# src/db_handling/chroma_utils.py
import os
import chromadb
from chromadb.api.models.Collection import Collection as ChromaCollection
from chromadb.api.client import ClientAPI as ChromaClientAPI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings, VectorStoreIndex, StorageContext, Document
from typing import Tuple, Set, Dict
from src import config
from src.logger.logger import MyLogger

logger = MyLogger(config.LOG_NAME)


def get_chroma_client() -> ChromaClientAPI:
    """Initializes and returns a persistent ChromaDB client."""
    ACTION = "Initialize ChromaDB Client"
    logger.start(ACTION)
    client = None
    try:
        persist_dir = config.PERSIST_DIR
        logger.info(f"Ensuring ChromaDB persistence directory exists: {persist_dir}")
        os.makedirs(persist_dir, exist_ok=True)
        logger.info(f"Initializing persistent ChromaDB client (Path: {persist_dir})")
        client = chromadb.PersistentClient(path=persist_dir)
        logger.info("ChromaDB client initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
        raise
    finally:
        logger.close(ACTION)

    if client is None:
        logger.critical("ChromaDB client initialization failed silently.")
        raise RuntimeError("ChromaDB client initialization failed.")
    return client


def get_or_create_chroma_collection(client: ChromaClientAPI) -> ChromaCollection:
    """Gets or creates the ChromaDB collection specified in config."""
    collection_name = config.CHROMA_COLLECTION_NAME
    ACTION = f"Get/Create Chroma Collection ('{collection_name}')"
    logger.start(ACTION)
    collection = None
    try:
        logger.info(f"Attempting to get or create Chroma collection: '{collection_name}'")
        collection = client.get_or_create_collection(collection_name)
        count = collection.count()
        logger.info(f"Collection '{collection_name}' ready. Current document count: {count}.")
    except Exception as e:
        logger.critical(f"Failed to get or create Chroma collection '{collection_name}': {e}", exc_info=True)
        raise
    finally:
        logger.close(ACTION)

    if collection is None:
        logger.critical(f"Chroma collection '{collection_name}' is None after get/create call.")
        raise RuntimeError("Chroma collection initialization failed.")
    return collection


def get_index_and_storage_context(collection: ChromaCollection) -> Tuple[VectorStoreIndex, StorageContext]:
    """Creates LlamaIndex objects (VectorStore, StorageContext, Index) from a Chroma collection."""
    ACTION = f"Create LlamaIndex Objects from Collection ('{collection.name}')"
    logger.start(ACTION)
    index = None
    storage_context = None
    try:
        logger.info("Setting up LlamaIndex VectorStore using the Chroma collection...")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        logger.info("Setting up LlamaIndex StorageContext...")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info("VectorStore and StorageContext created.")

        doc_count = collection.count()
        logger.info(f"Collection '{collection.name}' currently contains {doc_count} documents.")

        if not Settings.embed_model:
             logger.critical("Cannot load or create index: Embed model not configured in LlamaIndex Settings.")
             raise RuntimeError("Embed model not configured for index creation/loading.")

        if doc_count > 0:
            logger.info("Attempting to load LlamaIndex index from the existing vector store...")
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            logger.info("Successfully loaded index from vector store.")
        else:
            logger.info("Collection is empty. Creating a new, empty LlamaIndex index structure.")
            index = VectorStoreIndex.from_documents(
                documents=[],
                storage_context=storage_context,
                embed_model=Settings.embed_model
            )
            logger.info("New empty LlamaIndex index created.")

    except Exception as e:
         logger.critical(f"Failed to create/load LlamaIndex Index/StorageContext: {e}", exc_info=True)
         raise
    finally:
        logger.close(ACTION)

    if index is None or storage_context is None:
         logger.critical("Index or Storage Context is None after initialization attempt.")
         raise RuntimeError("Failed to create required LlamaIndex objects.")
    return index, storage_context


def get_all_qna_hashes_from_db(collection: ChromaCollection) -> Dict[str, str]:
    """Retrieves all document IDs and their associated 'qna_hash' metadata."""
    ACTION = f"Retrieve All QnA Hashes from DB ('{collection.name}')"
    logger.start(ACTION)
    qna_hash_to_id_map: Dict[str, str] = {}
    try:
        count = collection.count()
        if count == 0:
            logger.info("Collection is empty, no hashes to retrieve.")
            logger.close(ACTION)
            return {}

        logger.info(f"Retrieving default data (inc. metadata) for all {count} documents...")
        # Omit the 'include' parameter entirely to avoid type issues.
        results = collection.get() # Fetch default fields

        ids = results.get('ids', [])
        metadatas = results.get('metadatas', []) # Key in result dict is lowercase

        if not ids:
            logger.info("Retrieved 0 documents from the collection.")
            logger.close(ACTION)
            return {}

        if len(ids) != len(metadatas):
             logger.error(f"Mismatch between number of IDs ({len(ids)}) and metadatas ({len(metadatas)}) retrieved.")
        else:
            logger.info(f"Retrieved {len(ids)} IDs and metadatas (along with documents/embeddings).")

        processed_count = 0
        missing_hash_count = 0
        duplicate_hash_count = 0
        for doc_id, metadata in zip(ids, metadatas):
            if metadata and isinstance(metadata, dict):
                qna_hash = metadata.get('qna_hash')
                if qna_hash:
                    if qna_hash in qna_hash_to_id_map:
                        logger.warning(f"Duplicate qna_hash '{qna_hash[:8]}...' found in DB. Doc ID {doc_id} overwrites previous ID {qna_hash_to_id_map[qna_hash]}.")
                        duplicate_hash_count += 1
                    qna_hash_to_id_map[qna_hash] = doc_id
                    processed_count += 1
                else:
                    logger.warning(f"Document ID {doc_id} found in DB is missing 'qna_hash' in its metadata.")
                    missing_hash_count +=1
            else:
                logger.warning(f"Document ID {doc_id} has missing or invalid metadata: {metadata}")
                missing_hash_count += 1

        logger.info(f"Processed {processed_count} documents with qna_hash. "
                    f"Found {missing_hash_count} docs missing hash. "
                    f"Found {duplicate_hash_count} duplicate hashes (last one kept).")

    except Exception as e:
        logger.error(f"Error retrieving document metadata from ChromaDB: {e}", exc_info=True)
    finally:
        logger.close(ACTION)
    return qna_hash_to_id_map


def delete_documents_by_qna_hash(collection: ChromaCollection, qna_hashes_to_delete: Set[str]):
    """
    Deletes documents from a ChromaDB collection based on a set of 'qna_hash' values.
    """
    if not qna_hashes_to_delete:
        logger.info("No QnA hashes provided for deletion.")
        return

    ACTION = f"Delete {len(qna_hashes_to_delete)} Docs by QnA Hash"
    logger.start(ACTION)

    initial_doc_count = collection.count()
    logger.info(f"Attempting to delete documents for {len(qna_hashes_to_delete)} unique QnA hashes.")
    logger.debug(f"Hashes to delete: { {h[:8]+'...' for h in qna_hashes_to_delete} }")

    try:
        if qna_hashes_to_delete:
            where_filter = {"qna_hash": {"$in": list(qna_hashes_to_delete)}}
            logger.info(f"Initial doc count: {initial_doc_count}. Deleting documents where 'qna_hash' is in the target set...")

            collection.delete(where=where_filter) # type: ignore

            final_doc_count = collection.count()
            deleted_count = initial_doc_count - final_doc_count
            logger.info(f"Deletion complete. Actual documents removed: {deleted_count}. Final doc count: {final_doc_count}.")

    except Exception as e:
         logger.error(f"Error during document deletion by QnA hash: {e}", exc_info=True)
    finally:
        logger.close(ACTION)


def create_document_from_qna(
    file_path: str,
    subject: str,
    question: str,
    answer: str,
    qna_hash: str
) -> Document:
    """
    Creates a single LlamaIndex Document object with Q&A-specific metadata.
    Ensures text content is always a valid string.
    """
    # Ensure components are strings, default to empty string if None
    safe_subject = str(subject) if subject is not None else ""
    safe_question = str(question) if question is not None else ""
    safe_answer = str(answer) if answer is not None else ""

    # Construct the text safely
    doc_text = f"Subject: {safe_subject}\nQuestion: {safe_question}\nAnswer: {safe_answer}"

    doc_id = qna_hash
    doc = Document(
        id_=doc_id,
        text=doc_text, # Use the safe text
        metadata={
            "file_path": file_path,
            "subject": safe_subject, # Store safe subject
            "question": safe_question, # Store safe question
            "qna_hash": qna_hash
        }
    )
    return doc