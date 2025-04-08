import os
from typing import Dict, List, Any
from src import config
from src.logger.logger import MyLogger
from src.db_handling import hashing, parsing, chroma_utils, indexing_helpers
from llama_index.core import Document

logger = MyLogger(config.LOG_NAME)

QnAMetadata = Dict[str, Any]
QnADataFromFile = Dict[str, QnAMetadata]

def _scan_data_dir_for_qna() -> QnADataFromFile:
    """Scans the data directory, parses files, generates QnA hashes."""
    ACTION = f"Scan Data Directory for QnAs ({config.DATA_DIR})"
    logger.start(ACTION)

    all_qna_data: QnADataFromFile = {}
    stats = _initialize_stats()

    if not os.path.isdir(config.DATA_DIR):
        logger.error(f"Data directory '{config.DATA_DIR}' not found.")
        logger.close(ACTION)
        return {}

    _process_all_files(all_qna_data, stats)
    _log_final_stats(stats, len(all_qna_data))

    logger.close(ACTION)
    return all_qna_data

def _initialize_stats() -> Dict[str, int]:
    return {
        "files_processed": 0,
        "files_skipped_error": 0,
        "total_qna_pairs": 0,
        "duplicate_qna_hashes": 0
    }

def _process_all_files(all_qna_data: QnADataFromFile, stats: Dict[str, int]):
    try:
        for filename in os.listdir(config.DATA_DIR):
            filepath = os.path.join(config.DATA_DIR, filename)

            if _should_skip_file(filepath, filename):
                continue

            result = _process_single_file(filepath, filename, all_qna_data)
            if result:
                _update_stats(stats, result)
            else:
                stats["files_skipped_error"] += 1

    except OSError as e:
        logger.error(f"OS Error scanning directory {config.DATA_DIR}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during directory scan: {e}", exc_info=True)

def _should_skip_file(filepath: str, filename: str) -> bool:
    return not os.path.isfile(filepath) or not filename.lower().endswith(".txt")

def _process_single_file(filepath: str, filename: str, all_qna_data: QnADataFromFile):
    try:
        logger.debug(f"Processing file: {filename}")
        subject, qna_pairs = parsing.parse_faq_file(filepath)

        if not _is_valid_qna_file(subject, qna_pairs, filename):
            return None

        logger.debug(f"Parsed {len(qna_pairs)} Q&A pairs from {filename} (Subject: {subject}).")
        return _add_qna_pairs(filepath, filename, subject, qna_pairs, all_qna_data)

    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}", exc_info=True)
        return None

def _is_valid_qna_file(subject: str, qna_pairs: List, filename: str) -> bool:
    if subject is None and not qna_pairs:
        logger.warning(f"File {filename} parsing yielded no subject or Q&As. Skipping.")
        return False
    if not qna_pairs:
        logger.info(f"File {filename} parsed with subject '{subject}' but no valid Q&A pairs.")
        return False
    return True

def _add_qna_pairs(filepath: str, filename: str, subject: str, qna_pairs: List, all_qna_data: QnADataFromFile):
    duplicate_qna_hashes = 0
    for question, answer in qna_pairs:
        qna_hash = hashing.generate_qna_hash(question, answer)
        if qna_hash in all_qna_data:
            _log_duplicate(qna_hash, filename, question, all_qna_data[qna_hash])
            duplicate_qna_hashes += 1
            continue

        all_qna_data[qna_hash] = {
            "file_path": filepath,
            "subject": subject or "Unknown",
            "question": question,
            "answer": answer
        }

    return True, duplicate_qna_hashes, len(qna_pairs), 1

def _update_stats(stats: Dict[str, int], result: tuple):
    _, dupes, total, processed = result
    stats["total_qna_pairs"] += total
    stats["duplicate_qna_hashes"] += dupes
    stats["files_processed"] += processed

def _log_duplicate(qna_hash: str, filename: str, question: str, existing_meta: Dict[str, Any]):
    logger.warning(
        f"Duplicate QnA Hash '{qna_hash[:8]}...' found!\n"
        f"  New: File='{filename}', Q='{question[:50]}...'\n"
        f"  Existing: File='{os.path.basename(existing_meta['file_path'])}', Q='{existing_meta['question'][:50]}...'\n"
        f"  Skipping duplicate from '{filename}'."
    )

def _log_final_stats(stats: Dict[str, int], unique_count: int):
    logger.info(f"Scan Complete."
                f"\n\tFiles Processed:              {stats['files_processed']}"
                f"\n\tFiles Skipped (Error):        {stats['files_skipped_error']}"
                f"\n\tTotal QnA Pairs Found:        {stats['total_qna_pairs']}"
                f"\n\tUnique QnAs Added:            {unique_count}"
                f"\n\tDuplicate QnA Hashes Skipped: {stats['duplicate_qna_hashes']}.")

def synchronize_vector_db():
    ACTION = "Synchronize QnAs with Vector DB"
    logger.start(ACTION)

    try:
        _synchronize_qnas()
    except Exception as e:
        logger.critical(f"Critical error during QnA synchronization: {e}", exc_info=True)
    finally:
        logger.close(ACTION)

def _synchronize_qnas():
    indexing_helpers.setup_llama_index_settings()
    qna_data_from_files = _scan_data_dir_for_qna()
    current_qna_hashes_in_files = set(qna_data_from_files.keys())
    logger.info(f"Found {len(current_qna_hashes_in_files)} unique QnA hashes in data files.")

    client, collection, index, storage_context = indexing_helpers.initialize_db_and_index()
    qna_hash_to_id_in_db = chroma_utils.get_all_qna_hashes_from_db(collection)
    qna_hashes_in_db = set(qna_hash_to_id_in_db.keys())
    logger.info(f"Found {len(qna_hashes_in_db)} unique QnA hashes in the database.")

    _sync_db_with_files(collection, index, qna_data_from_files, current_qna_hashes_in_files, qna_hashes_in_db)

def _sync_db_with_files(collection, index, qna_data_from_files, file_hashes, db_hashes):
    hashes_to_add = file_hashes - db_hashes
    hashes_to_delete = db_hashes - file_hashes

    if not hashes_to_add and not hashes_to_delete:
        logger.info("No changes detected. DB is synchronized with data files.")
        return

    add_and_delete(collection, index, qna_data_from_files, hashes_to_delete, hashes_to_add)
    _verify_db_sync(collection, file_hashes)

def _verify_db_sync(collection, expected_hashes):
    final_qna_hash_to_id_in_db = chroma_utils.get_all_qna_hashes_from_db(collection)
    final_qna_hashes_in_db = set(final_qna_hash_to_id_in_db.keys())

    if final_qna_hashes_in_db == expected_hashes:
        logger.info("Post-sync verification successful: DB hashes match file hashes.")
        logger.info(f"Final document count in DB: {len(final_qna_hashes_in_db)}")
    else:
        logger.warning("Post-sync verification FAILED: Discrepancy between DB hashes and file hashes.")
        logger.warning(f"  Hashes in DB only: {len(final_qna_hashes_in_db - expected_hashes)}")
        logger.warning(f"  Hashes in Files only: {len(expected_hashes - final_qna_hashes_in_db)}")

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
            logger.error(f"Internal inconsistency: Cannot find metadata for qna_hash '{qna_hash[:8]}...' marked for addition.")

    if documents_to_insert:
        insertion_success = indexing_helpers.insert_documents_to_index(index, documents_to_insert)
        if not insertion_success:
            logger.error("One or more documents failed to insert. DB might be inconsistent.")
    else:
        logger.warning("No valid documents were created for the hashes marked for addition.")
