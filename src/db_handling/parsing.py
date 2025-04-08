import os
from typing import List, Tuple, Optional
from src import config
from src.logger.logger import MyLogger

logger = MyLogger(config.LOG_NAME)


def _read_and_clean_lines(filepath: str) -> List[str]:
    """
    Read file and return stripped, non-empty lines.

    :param filepath: Path to the file.
    :return: List of cleaned lines.
    """
    lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}", exc_info=True)
    return lines


def _validate_question(question: str, line_num: int, filename: str) -> bool:
    """
    Validate that a question ends with a question mark.

    :param question: Question string to validate.
    :param line_num: Line number in the original file.
    :param filename: Name of the file being processed.
    :return: True if valid, False otherwise.
    """
    if not question.endswith('?'):
        logger.warning(
            f"Invalid format in {filename}: Question on line {line_num} "
            f"does not end with '?'. Skipping this and subsequent Q&A pairs.\n"
            f"   Question: '{question}'"
        )
        return False
    return True


def _parse_qna_pairs(lines: List[str], filename: str) -> List[Tuple[str, str]]:
    """
    Parse (question, answer) pairs from cleaned lines (excluding subject).

    :param lines: Cleaned lines excluding the subject.
    :param filename: Name of the file for logging context.
    :return: List of (question, answer) pairs.
    """
    qna_pairs = []

    for i in range(0, len(lines) - 1, 2):
        question_line_num = i + 2  # Account for 1-based line numbers + subject
        answer_line_num = i + 3

        question = lines[i]
        answer = lines[i + 1]

        if not _validate_question(question, question_line_num, filename):
            logger.warning(f"Stopping parsing for {filename} due to invalid question.")
            break

        if question and answer:
            qna_pairs.append((question, answer))
        else:
            logger.warning(f"Empty question or answer near line {question_line_num}/{answer_line_num} in {filename}.")

    if len(lines) % 2 != 0:
        logger.warning(
            f"File {filename} has an unpaired last content line (line {len(lines) + 1}). Ignored."
        )

    return qna_pairs


def _extract_subject(cleaned_lines: List[str], filename: str) -> Optional[str]:
    """
    Extract the subject line from the cleaned lines.

    :param cleaned_lines: Non-empty lines from the file.
    :param filename: Filename for logging context.
    :return: Subject string or None.
    """
    if not cleaned_lines:
        logger.warning(f"File {filename} is empty or whitespace only.")
        return None

    subject = cleaned_lines[0]
    logger.debug(f"Extracted Subject: '{subject}' from {filename}")
    return subject


def _extract_qna_data(cleaned_lines: List[str], filename: str) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """
    Extract subject and Q&A pairs from cleaned lines.

    :param cleaned_lines: Preprocessed lines.
    :param filename: File context for logging.
    :return: Tuple of subject and Q&A pairs.
    """
    subject = _extract_subject(cleaned_lines, filename)
    if not subject or len(cleaned_lines) < 3:
        logger.warning(f"File {filename} has a subject but no Q&A pairs.")
        return subject, []

    qna_pairs = _parse_qna_pairs(cleaned_lines[1:], filename)
    return subject, qna_pairs


def parse_faq_file(filepath: str) -> Tuple[Optional[str], List[Tuple[str, str]]]:
    """
    Parse a FAQ-style text file into a subject and list of (question, answer) tuples.

    :param filepath: Path to the FAQ file.
    :return: Tuple of subject and list of Q&A pairs. Subject is None if parsing fails.
    """
    filename = os.path.basename(filepath)
    subject = None
    qna_pairs = []

    try:
        cleaned_lines = _read_and_clean_lines(filepath)

        if not cleaned_lines:
            logger.warning(f"No content read from {filename}. Skipping parsing.")
            return None, []

        subject, qna_pairs = _extract_qna_data(cleaned_lines, filename)

        if subject and not qna_pairs and len(cleaned_lines) >= 3:
            logger.warning(f"Parsed subject from {filename} but found no valid Q&A pairs.")
        elif qna_pairs:
            logger.debug(f"Parsed subject and {len(qna_pairs)} Q&A pairs from {filename}.")

    except Exception as e:
        logger.error(f"Unexpected error during parsing {filepath}: {e}", exc_info=True)

    return subject, qna_pairs
