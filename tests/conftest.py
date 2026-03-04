"""
Pytest configuration and shared fixtures
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory"""
    return Path(__file__).parent / 'data'


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset imports between tests"""
    yield
