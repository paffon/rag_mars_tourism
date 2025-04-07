import os
from typing import Tuple, Set, Dict

import chromadb
from chromadb.api.models.Collection import Collection as ChromaCollection
from chromadb.api.client import ClientAPI as ChromaClientAPI

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings, VectorStoreIndex, StorageContext, Document

from src import config
from src.logger.logger import MyLogger

logger = MyLogger(config.LOG_NAME)

def get_chroma_client() -> ChromaClientAPI:
    """
    Initializes and returns a persistent ChromaDB client.

    :return: A configured persistent ChromaDB client.
    :raises RuntimeError: If initialization fails.
    """
    ACTION = "Initialize ChromaDB Client"
    logger.start(ACTION)
    try:
        os.makedirs(config.PERSIST_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=config.PERSIST_DIR)
        return client
    except Exception as e:
        logger.critical(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
        raise RuntimeError("ChromaDB client initialization failed.")
    finally:
        logger.close(ACTION)

def get_or_create_chroma_collection(client: ChromaClientAPI) -> ChromaCollection:
    """
    Retrieves or creates a ChromaDB collection.

    :param client: Chroma client.
    :return: Existing or newly created Chroma collection.
    :raises RuntimeError: If collection retrieval fails.
    """
    ACTION = f"Get/Create Chroma Collection"
    logger.start(ACTION)
    try:
        collection = client.get_or_create_collection(config.CHROMA_COLLECTION_NAME)
        return collection
    except Exception as e:
        logger.critical(f"Failed to get/create collection: {e}", exc_info=True)
        raise RuntimeError("Chroma collection initialization failed.")
    finally:
        logger.close(ACTION)

def get_index_and_storage_context(collection: ChromaCollection) -> Tuple[VectorStoreIndex, StorageContext]:
    """
    Creates LlamaIndex index and storage context from a Chroma collection.

    :param collection: Chroma collection to wrap with LlamaIndex.
    :return: Tuple of VectorStoreIndex and StorageContext.
    :raises RuntimeError: If index creation fails.
    """
    ACTION = f"Create Index/StorageContext"
    logger.start(ACTION)
    try:
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if not Settings.embed_model:
            raise RuntimeError("Embed model not configured.")

        if collection.count() > 0:
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        else:
            index = VectorStoreIndex.from_documents(documents=[], storage_context=storage_context, embed_model=Settings.embed_model)

        return index, storage_context
    except Exception as e:
        logger.critical(f"Failed to create index/storage context: {e}", exc_info=True)
        raise RuntimeError("Index or StorageContext creation failed.")
    finally:
        logger.close(ACTION)

def get_all_qna_hashes_from_db(collection: ChromaCollection) -> Dict[str, str]:
    """
    Retrieves a mapping of all QnA hashes to their corresponding document IDs in the collection.

    :param collection: Chroma collection to query.
    :return: Dictionary mapping qna_hash -> document_id.
    """
    ACTION = f"Retrieve QnA Hashes"
    logger.start(ACTION)
    try:
        result = collection.get()
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])

        mapping = {}
        for doc_id, metadata in zip(ids, metadatas):
            if isinstance(metadata, dict):
                qna_hash = metadata.get("qna_hash")
                if qna_hash:
                    mapping[qna_hash] = doc_id

        return mapping
    except Exception as e:
        logger.error(f"Failed to retrieve QnA hashes: {e}", exc_info=True)
        return {}
    finally:
        logger.close(ACTION)

def delete_documents_by_qna_hash(collection: ChromaCollection, qna_hashes_to_delete: Set[str]) -> None:
    """
    Deletes documents in the collection matching any of the provided QnA hashes.

    :param collection: Chroma collection.
    :param qna_hashes_to_delete: Set of QnA hashes identifying documents to delete.
    """
    if not qna_hashes_to_delete:
        return

    ACTION = f"Delete Documents by QnA Hash"
    logger.start(ACTION)
    try:
        collection.delete(where={"qna_hash": {"$in": list(qna_hashes_to_delete)}})  # type: ignore
    except Exception as e:
        logger.error(f"Failed to delete documents: {e}", exc_info=True)
    finally:
        logger.close(ACTION)

def create_document_from_qna(file_path: str, subject: str, question: str, answer: str, qna_hash: str) -> Document:
    """
    Constructs a LlamaIndex Document using given QnA metadata.

    :param file_path: Source file of the QnA.
    :param subject: Topic of the QnA.
    :param question: The question text.
    :param answer: The answer text.
    :param qna_hash: Unique identifier for the QnA.
    :return: Document with structured text and metadata.
    """
    doc_text = f"Subject: {subject}\nQuestion: {question}\nAnswer: {answer}"
    return Document(
        id_=qna_hash,
        text=doc_text,
        metadata={
            "file_path": file_path,
            "subject": subject,
            "question": question,
            "qna_hash": qna_hash,
        },
    )
