"""
Centralized logging configuration module.

This module automatically configures logging when imported.
It provides a singleton LoggerManager instance and convenience functions
for getting loggers and managing TQDM compatibility.
"""

from typing import Union, Optional
import logging
import sys
from tqdm import tqdm
import sys
from contextlib import contextmanager

class LoggerManager:
    """
    Singleton class for managing logging configuration across the application.
    Provides support for both standard logging and TQDM-compatible logging.
    """
    _instance = None
    _initialized = False

    class TqdmLoggingHandler(logging.Handler):
        """Custom logging handler that writes to tqdm output."""
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
            except Exception:
                self.handleError(record)

    def __new__(cls, 
                log_level: Union[str, int] = logging.INFO,
                log_file: Optional[str] = None,
                use_tqdm: bool = False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, 
                 log_level: Union[str, int] = logging.INFO,
                 log_file: Optional[str] = None,
                 use_tqdm: bool = False):
        """
        Initialize logging configuration if not already done.

        Args:
            log_level: Logging level as string ('DEBUG', 'INFO', etc.) or logging constant
            log_file: Optional file path for logging to file
            use_tqdm: If True, use TQDM-compatible logging handler
        """
        if not self._initialized:
            self._configure_logging(log_level, log_file, use_tqdm)
            LoggerManager._initialized = True

    def _configure_logging(self, log_level: Union[str, int], log_file: Optional[str], use_tqdm: bool):
        """
        Configure logging with the specified settings.

        Args:
            log_level: Logging level to set
            log_file: Optional file path for logging output
            use_tqdm: If True, use TQDM-compatible logging handler
        """
        if isinstance(log_level, str):
            numeric_level = getattr(logging, log_level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {log_level}")
        else:
            numeric_level = log_level

        handlers = []
        if use_tqdm:
            handlers.append(self.TqdmLoggingHandler())
        else:
            handlers.append(logging.StreamHandler(sys.stdout))
            
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            handlers.append(file_handler)

        logging.basicConfig(
            level=numeric_level,
            format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=handlers,
            force=True
        )

        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)

    @staticmethod
    def get_logger(name: Optional[str] = None) -> logging.Logger:
        """Get a configured logger instance."""
        return logging.getLogger(name)

    def enable_tqdm_logging(self):
        """Switch to TQDM-compatible logging handler."""
        self._configure_logging(logging.getLogger().level, None, use_tqdm=True)

    def disable_tqdm_logging(self):
        """Switch back to standard logging handler."""
        self._configure_logging(logging.getLogger().level, None, use_tqdm=False)


# Create default logger manager instance
default_logger_manager = LoggerManager(log_level="INFO")

# Convenience functions
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name, typically __name__ for module-level logging
    """
    return default_logger_manager.get_logger(name)

def enable_tqdm():
    """Enable TQDM-compatible logging."""
    default_logger_manager.enable_tqdm_logging()

def disable_tqdm():
    """Disable TQDM-compatible logging and return to standard output."""
    default_logger_manager.disable_tqdm_logging()


# Usage example:
if __name__ == "__main__":
    # Get a logger
    logger = get_logger(__name__)
    
    # Normal logging
    logger.info("Starting process...")
    
    # With TQDM
    enable_tqdm()
    for i in tqdm(range(100)):
        if i % 25 == 0:
            logger.info(f"Processing item {i}")
    disable_tqdm()
    
    # Back to normal logging
    logger.info("Process completed!")

# TODO
@contextmanager
def LogFile(filename, append=True):
    stdout = sys.stdout
    mode = 'a' if append else 'w'
    with open(filename, mode) as f:
        class TeeStdout:
            def write(self, data):
                f.write(data)
                stdout.write(data)
            def flush(self):
                f.flush()
                stdout.flush()
        sys.stdout = TeeStdout()
        yield
        sys.stdout = stdout