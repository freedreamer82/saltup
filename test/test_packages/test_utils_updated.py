import pytest
import os
from pathlib import Path
from saltup.utils.misc import match_patterns, count_files, unify_files
from saltup.utils.configure_logging import LoggerManager, get_logger

class TestMiscUtils:
    @pytest.fixture
    def sample_dir(self, tmp_path):
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        
        (dir1 / "test1.txt").touch()
        (dir1 / "test2.jpg").touch()
        (dir2 / "test3.txt").touch()
        
        return tmp_path

    def test_match_patterns(self):
        test_cases = [
            ("test.txt", ["*.txt"], True),
            ("test.jpg", ["*.txt"], False),
            ("test.txt", [["test*", "*.txt"]], True),
        ]
        for target, patterns, expected in test_cases:
            assert match_patterns(target, patterns) == expected

    def test_count_files(self, sample_dir):
        count, files = count_files(str(sample_dir))
        assert count == 3
        assert len(files) == 3

        count, files = count_files(str(sample_dir), filters=["*.txt"])
        assert count == 2
        assert all(f.endswith(".txt") for f in files)

class TestLogging:
    def test_logger_singleton(self):
        manager1 = LoggerManager()
        manager2 = LoggerManager()
        assert manager1 is manager2

    def test_get_logger(self):
        logger1 = get_logger("test_logger")
        logger2 = get_logger("test_logger")
        assert logger1 is logger2

    def test_log_level_setting(self, tmp_path):
        log_file = tmp_path / "test.log"
        manager = LoggerManager()
        manager._configure_logging("DEBUG", str(log_file), False)  # Reconfigure directly
        logger = manager.get_logger("test")
        
        test_message = "Test log message"
        logger.debug(test_message)
        
        assert log_file.exists()
        assert test_message in log_file.read_text()
