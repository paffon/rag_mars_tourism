import logging
import time
from threading import Lock
from typing import List, Tuple, Optional
import os

from src.logger.format import format_duration  # Ensure this import is correct


class MyLogger:
    """
    A custom logger that tracks actions and indents messages based on action levels.
    Implements the Singleton pattern to ensure only one instance exists.
    """
    _instance = None
    _lock = Lock()  # Ensures thread-safe singleton initialization

    def __new__(cls, name: str = 'MyLogger', level: int = logging.INFO):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-checked locking
                    cls._instance = super(MyLogger, cls).__new__(cls)
                    cls._instance._initialize_logger(name, level)
                else:
                    pass
        return cls._instance

    def _initialize_logger(self, name: str = 'MyCustomLogger', level: int = logging.INFO):
        """
        Initialize the MyLogger instance.

        :param name: Name of the logger.
        :param level: Logging level.
        """
        self.actions: List[Tuple[str, float]] = []
        self.sep = ' - '
        self.spacer = '   '

        # Create or get the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove any existing handlers to prevent duplicate logs
        if not self.logger.handlers:
            # Create console handler
            handler = logging.StreamHandler()
            handler.setLevel(level)

            # Create formatter
            self.formatter = logging.Formatter(
                f'%(asctime)s{self.sep}%(name)s{self.sep}%(levelname)s: %(message)s'
            )

            # Add formatter to handler
            handler.setFormatter(self.formatter)

            # Add handler to logger
            self.logger.addHandler(handler)

        self._initialized = True  # Flag to indicate that initialization has been done

    def _indent_message(self, level_msg: str, message: str) -> str:
        """
        Indent the message based on the current action level.

        :param level_msg: The logging level as a string (e.g., 'INFO', 'DEBUG').
        :param message: Original log message.
        :return: Indented message.
        """
        # Calculate the length of the timestamp and logger name
        # Assuming the timestamp format 'YYYY-MM-DD HH:MM:SS,sss'
        timestamp_length = len('2024-12-04 10:01:34,252')  # Example length
        longest_level = 'CRITICAL'
        # Calculate the fixed part length
        fixed_length = (
            timestamp_length + len(self.sep) + len(self.logger.name) +
            len(self.sep) + len(longest_level) + len(': ')
        )
        level_length = len(level_msg)
        # Calculate the difference to align messages
        diff = fixed_length - (
            timestamp_length + len(self.sep) + len(self.logger.name) +
            len(self.sep) + level_length + len(': ')
        )
        level = self.spacer * len(self.actions)

        indented_message = ' ' * diff + level + str(message).replace(
            '\n', '\n' + ' ' * fixed_length + level
        )

        return indented_message

    def start(self, action: str):
        """
        Start a new action, logging the action and increasing indentation.

        :param action: Description of the action.
        """
        self.info(f"{action} {{")
        self.actions.append((action, time.time()))

    def close(self, action: str) -> Tuple[Optional[str], Optional[float]]:
        """
        If the most recent action is as the action given, close it, logging the time it took.
        Otherwise, log a warning and return None.
        """
        if not self.actions:
            self.warning("No action to close.")
            return None, None

        action, start_time = self.actions.pop()

        if action != action:
            self.warning(f"Action mismatch: '{action}' does not match the most recent action.")
            return None, None

        duration = time.time() - start_time
        formatted_duration = format_duration(duration)
        self.info(f"}} ({formatted_duration})")
        return action, duration

    # Overriding logging methods to include indentation
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(self._indent_message('DEBUG', msg), *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(self._indent_message('INFO', msg), *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(self._indent_message('WARNING', msg), *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(self._indent_message('ERROR', msg), *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(self._indent_message('CRITICAL', msg), *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        level_name = logging.getLevelName(level)
        self.logger.log(level, self._indent_message(level_name, msg), *args, **kwargs)

    # Additional methods to access logger properties if needed
    def setLevel(self, level):
        self.logger.setLevel(level)
        # Update level for all handlers
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def addHandler(self, handler):
        self.logger.addHandler(handler)

    def removeHandler(self, handler):
        self.logger.removeHandler(handler)

    def getEffectiveLevel(self):
        return self.logger.getEffectiveLevel()

    def set_log_file(self, log_file: str):
        """
        Sets the log file by adding a FileHandler to the logger.

        :param log_file: Path to the log file.
        """
        # Make sure the directory exists
        self.info(f'Making sure that {os.path.dirname(log_file)} exists')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Check if a FileHandler for this log file already exists
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == os.path.abspath(log_file):
                self.info(f"File handler for {log_file} already exists.")
                return

        # Create and add a new FileHandler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.logger.level)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

        self.info(f"File handler for {log_file} added.")


logger = MyLogger('MyDayBreak')


if __name__ == "__main__":
    # Since MyLogger is now a singleton, all instances will refer to the same logger
    logger = MyLogger('MyCustomLogger', logging.DEBUG)

    logger.start("Main Task")
    logger.info("Processing data...")
    logger.start("Subtask 1")
    time.sleep(0.7)
    logger.debug("Debugging info")
    logger.close("Subtask 1")
    logger.start("Subtask 2")
    logger.critical('Message:\n \tItem A\n\t\tItem A.1\n\tItem B')
    logger.error("An error occurred")
    time.sleep(0.4)
    logger.critical("A critical error occurred")

    # Attempting to create another logger with a different name
    another_logger = MyLogger('AnotherLogger')  # This will return the same instance as `logger`
    another_logger.info('TEST INFO TEST INFO')  # This will use the initial logger name 'MyCustomLogger'

    logger.warning("A warning message")
    logger.close("Subtask 1")
    time.sleep(0.2)
    logger.close("Main Task")
