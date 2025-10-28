"""
Pytest configuration file.

Defines custom markers and shared fixtures for the test suite.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require external dependencies)"
    )
    config.addinivalue_line(
        "markers", "requires_credentials: marks tests that require HuggingFace or other credentials"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU access"
    )
