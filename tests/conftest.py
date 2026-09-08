from __future__ import annotations

import sys
from pathlib import Path
import uuid

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


@pytest.fixture
def tmp_path():
    """Ordinary directory permissions also work in managed Windows workspaces."""
    path = SOURCE.parent / "_test_work" / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path
