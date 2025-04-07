import os
from typing import List, Tuple, Optional
from src import config
from src.logger.logger import MyLogger

logger = MyLogger(config.LOG_NAME)


def _read_and_clean_lines(filepath: str) -> List[str]:
    """Reads a file, removes whitespace, and filters empty lines."""
    lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}", exc_info=True)
    return lines


def _validate_question(question: str, line_num: int, filename: str) -> bool:
    """Validates if a question string ends with '?'."""
    if not question.endswith('?'):
        logger.warning( # Changed to warning as it might skip the file, not crash
            f"Invalid format in {filename}: Question on line {line_num} "
            f"does not end with '?'. Ignoring this Q&A pair and subsequent pairs in this file."
            f"\n   Question: '{question}'"
        )
        return False
    return True


def _extract_subject_and_qna(
    cleaned_lines: List[str], filename: str
) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """Extracts Subject and Q&A pairs from cleaned lines."""
    qna_pairs = []
    subject = None

    if not cleaned_lines:
        logger.warning(f"File {filename} is empty or contains only whitespace.")
        return None, []

    # First non-empty line is the subject
    subject = cleaned_lines[0]
    logger.debug(f"Extracted Subject: '{subject}' from {filename}")

    # Check if there are enough lines for at least one Q&A pair after the subject
    if len(cleaned_lines) < 3:
        logger.warning(f"File {filename} has a subject but no Q&A pairs (less than 3 non-empty lines).")
        return subject, []

    # Iterate through lines starting from index 1 for Q/A pairs
    for i in range(1, len(cleaned_lines) - 1, 2):
        question_line_num = i + 1 # Line number in the original file
        answer_line_num = i + 2
        question = cleaned_lines[i]
        answer = cleaned_lines[i+1]

        if not _validate_question(question, question_line_num, filename):
            # Stop processing this file if a question is invalid, return what we have so far
            logger.warning(f"Stopping parsing for {filename} due to invalid question format.")
            break # Exit the loop for this file

        if question and answer:
            qna_pairs.append((question, answer))
        else:
             # This case should be rare due to the initial line stripping/filtering
             logger.warning(f"Found empty question or answer around line {question_line_num}"
                            f"/{answer_line_num} in {filename} unexpectedly.")

    # Log if the number of content lines (excluding subject) was odd
    if (len(cleaned_lines) - 1) % 2 != 0 and len(cleaned_lines) > 1:
         last_line_num = len(cleaned_lines)
         logger.warning(f"File {filename} has an odd number of content lines after the subject. "
                        f"The last content line (line {last_line_num}) was ignored.")

    return subject, qna_pairs


def parse_faq_file(filepath: str) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """
    Parses a FAQ text file into a subject and a list of (question, answer) tuples.

    Format Expected:
     Line 1: Subject/Topic
     Line 2: Question 1 (must end with '?')
     Line 3: Answer 1
     Line 4: Question 2 (must end with '?')
     Line 5: Answer 2
     ... (Empty lines between entries are ignored)
    """
    filename = os.path.basename(filepath)

    subject = None
    qna_pairs = []

    try:
        cleaned_lines = _read_and_clean_lines(filepath)

        if cleaned_lines:
            subject, qna_pairs = _extract_subject_and_qna(cleaned_lines, filename)
        else:
            logger.warning(f"No content read from {filename}. Parsing skipped.")

        if subject and not qna_pairs and len(cleaned_lines) >= 3:
             logger.warning(f"Parsing {filename} resulted in 0 valid Q&A pairs "
                            f"(check format/validation warnings). Subject was '{subject}'.")
        elif qna_pairs:
             logger.debug(f"Successfully parsed subject and {len(qna_pairs)} Q&A pairs from {filename}.")

    except Exception as e:
        logger.error(f"Unexpected error during parsing orchestration for {filepath}: {e}", exc_info=True)
        subject = None # Ensure return consistency on error
        qna_pairs = []

    finally:
        return subject, qna_pairs
