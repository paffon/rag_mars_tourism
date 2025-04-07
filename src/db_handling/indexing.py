import os
from typing import Dict, List, Any
from src import config
from src.logger.logger import MyLogger
from src.db_handling import hashing, parsing, chroma_utils, indexing_helpers
from llama_index.core import Document

logger = MyLogger(config.LOG_NAME)


# Structure to hold Q&A data from files
QnAMetadata = Dict[str, Any] # TypedDict could be used for more rigor
# { qna_hash: {"file_path": str, "subject": str, "question": str, "answer": str} }
QnADataFromFile = Dict[str, QnAMetadata]


def _scan_data_dir_for_qna() -> QnADataFromFile:
    """Scans the data directory, parses files, generates QnA hashes."""
    ACTION = f"Scan Data Directory for QnAs ({config.DATA_DIR})"
    logger.start(ACTION)

    all_qna_data: QnADataFromFile = {}
    files_processed = 0
    files_skipped_error = 0
    total_qna_pairs = 0
    duplicate_qna_hashes = 0

    if not os.path.isdir(config.DATA_DIR):
        logger.error(f"Data directory '{config.DATA_DIR}' not found.")
        logger.close(ACTION)
        return {}

    try:
        for filename in os.listdir(config.DATA_DIR):
            filepath = os.path.join(config.DATA_DIR, filename)

            if not os.path.isfile(filepath) or not filename.lower().endswith(".txt"):
                # logger.debug(f"Skipping non-txt file: {filename}")
                continue

            try:
                logger.debug(f"Processing file: {filename}")
                subject, qna_pairs = parsing.parse_faq_file(filepath)
                files_processed += 1

                if subject is None and not qna_pairs:
                    logger.warning(f"File {filename} parsing yielded no subject or Q&As. Skipping.")
                    continue
                if not qna_pairs:
                     logger.info(f"File {filename} parsed with subject '{subject}' but no valid Q&A pairs.")
                     continue # Skip files with no Q&As

                logger.debug(f"Parsed {len(qna_pairs)} Q&A pairs from {filename} (Subject: {subject}).")
                total_qna_pairs += len(qna_pairs)

                for question, answer in qna_pairs:
                    qna_hash = hashing.generate_qna_hash(question, answer)

                    if qna_hash in all_qna_data:
                         existing_meta = all_qna_data[qna_hash]
                         logger.warning(
                             f"Duplicate QnA Hash '{qna_hash[:8]}...' found!\n"
                             f"  New: File='{filename}', Q='{question[:50]}...'\n"
                             f"  Existing: File='{os.path.basename(existing_meta['file_path'])}', Q='{existing_meta['question'][:50]}...'\n"
                             f"  Skipping duplicate from '{filename}'."
                         )
                         duplicate_qna_hashes += 1
                         continue # Skip duplicate hash

                    all_qna_data[qna_hash] = {
                        "file_path": filepath,
                        "subject": subject or "Unknown", # Handle case where subject might be None but pairs exist
                        "question": question,
                        "answer": answer
                    }

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}", exc_info=True)
                files_skipped_error += 1

    except OSError as e:
        logger.error(f"OS Error scanning directory {config.DATA_DIR}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during directory scan: {e}", exc_info=True)

    logger.info(f"Scan Complete."
                f"\n\tFiles Processed:              {files_processed}"
                f"\n\tFiles Skipped (Error):        {files_skipped_error}"
                f"\n\tTotal QnA Pairs Found:        {total_qna_pairs}"
                f"\n\tUnique QnAs Added:            {len(all_qna_data)}"
                f"\n\tDuplicate QnA Hashes Skipped: {duplicate_qna_hashes}.")
    logger.close(ACTION)
    return all_qna_data


def synchronize_vector_db():
    """
    Main function to synchronize Q&As from files with the vector database.
    Ensures 1-to-1 correlation between Q&As in files and docs in the DB.
    """
    ACTION = "Synchronize QnAs with Vector DB"
    logger.start(ACTION)

    try:
        # 0. Setup LlamaIndex LLM and Embedding Model
        indexing_helpers.setup_llama_index_settings()

        # 1. Scan data directory and get all current Q&A data and hashes
        qna_data_from_files = _scan_data_dir_for_qna()
        current_qna_hashes_in_files = set(qna_data_from_files.keys())
        logger.info(f"Found {len(current_qna_hashes_in_files)} unique QnA hashes in data files.")

        # 2. Initialize DB connection and get hashes currently in the DB
        client, collection, index, storage_context = indexing_helpers.initialize_db_and_index()
        qna_hash_to_id_in_db = chroma_utils.get_all_qna_hashes_from_db(collection)
        qna_hashes_in_db = set(qna_hash_to_id_in_db.keys())
        logger.info(f"Found {len(qna_hashes_in_db)} unique QnA hashes in the database.")

        # 3. Determine differences
        hashes_to_add = current_qna_hashes_in_files - qna_hashes_in_db
        hashes_to_delete = qna_hashes_in_db - current_qna_hashes_in_files

        if not hashes_to_add and not hashes_to_delete:
            logger.info("No changes detected. DB is synchronized with data files.")
            return # Exit early if no changes

        if hashes_to_delete or hashes_to_add:
            add_and_delete(collection, index, qna_data_from_files, hashes_to_delete, hashes_to_add)

        # 6. Final Verification (Optional but Recommended)
        # Re-fetch hashes from DB to confirm additions/deletions worked as expected
        final_qna_hash_to_id_in_db = chroma_utils.get_all_qna_hashes_from_db(collection)
        final_qna_hashes_in_db = set(final_qna_hash_to_id_in_db.keys())
        if final_qna_hashes_in_db == current_qna_hashes_in_files:
            logger.info("Post-sync verification successful: DB hashes match file hashes.")
            logger.info(f"Final document count in DB: {len(final_qna_hashes_in_db)}")
        else:
            logger.warning("Post-sync verification FAILED: Discrepancy between DB hashes and file hashes.")
            logger.warning(f"  Hashes in DB only: {len(final_qna_hashes_in_db - current_qna_hashes_in_files)}")
            logger.warning(f"  Hashes in Files only: {len(current_qna_hashes_in_files - final_qna_hashes_in_db)}")


    except Exception as e:
        logger.critical(f"Critical error during QnA synchronization: {e}", exc_info=True)
        # Depending on the error, the DB state might be inconsistent.

    finally:
        logger.close(ACTION)


def add_and_delete(collection, index, qna_data_from_files, hashes_to_delete, hashes_to_add):
    ACTION = 'Adding / Deleting'
    logger.start(ACTION)

    if hashes_to_delete:
        chroma_utils.delete_documents_by_qna_hash(collection, hashes_to_delete)

    if hashes_to_add:
        prep_for_adding_docs(index, hashes_to_add, qna_data_from_files)

    logger.close(ACTION)


def prep_for_adding_docs(index, hashes_to_add, qna_data_from_files):
    documents_to_insert: List[Document] = []
    logger.info(f"Preparing {len(hashes_to_add)} new documents for insertion...")
    for qna_hash in hashes_to_add:
        metadata = qna_data_from_files.get(qna_hash)
        if metadata:
            doc = chroma_utils.create_document_from_qna(
                file_path=metadata["file_path"],
                subject=metadata["subject"],
                question=metadata["question"],
                answer=metadata["answer"],
                qna_hash=qna_hash
            )
            documents_to_insert.append(doc)
        else:
            # This should not happen if hashes_to_add comes from qna_data_from_files keys
            logger.error(
                f"Internal inconsistency: Cannot find metadata for qna_hash '{qna_hash[:8]}...' marked for addition.")

    if documents_to_insert:
        insertion_success = indexing_helpers.insert_documents_to_index(index, documents_to_insert)
        if not insertion_success:
            logger.error("One or more documents failed to insert. DB might be inconsistent.")

    else:
        logger.warning("No valid documents were created for the hashes marked for addition.")