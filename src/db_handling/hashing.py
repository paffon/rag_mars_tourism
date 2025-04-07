import hashlib
from src import config
from src.logger.logger import MyLogger

logger = MyLogger(config.LOG_NAME)


def generate_qna_hash(question: str, answer: str) -> str:
    """
    Generates an SHA-256 hash for a question-answer pair.

    :param question: The question string to hash.
    :param answer: The answer string to include in the hash.
    :return: A hexadecimal SHA-256 hash representing the combined input.
    """
    hasher = hashlib.sha256()
    hasher.update(question.encode('utf-8'))
    hasher.update(answer.encode('utf-8'))  # Include answer in hash
    return hasher.hexdigest()
