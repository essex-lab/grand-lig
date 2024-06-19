"""
Unit and regression test for the grandlig package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import grandlig


def test_grandlig_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "grandlig" in sys.modules
