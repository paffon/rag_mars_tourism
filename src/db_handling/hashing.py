import hashlib
from src import config
from src.logger.logger import MyLogger

logger = MyLogger(config.LOG_NAME)

def generate_qna_hash(question: str, answer: str) -> str:
    """Generates a SHA-256 hash for a question-answer pair."""
    hasher = hashlib.sha256()
    hasher.update(question.encode('utf-8'))
    hasher.update(answer.encode('utf-8')) # Include answer in hash
    hex_hash = hasher.hexdigest()
    return hex_hash
