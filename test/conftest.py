import pytest
import tempfile
import shutil
from pathlib import Path

@pytest.fixture(scope="session")
def root_dir():
    """Fixture che fornisce il percorso della directory root dei test."""
    return Path(__file__).parent

@pytest.fixture(scope="session")
def test_data_dir():
    """Base temporary directory for test data that persists across all tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def temp_dir():
    """Temporary directory cleaned up after each test."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)
