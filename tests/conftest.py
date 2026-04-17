from __future__ import annotations
 
import pytest
 
from squishy.tools.base import ToolContext
 
 
@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(working_dir=str(tmp_path), permission_mode="yolo", use_sandbox=False)
