"""Load MCP server configs from .mcp.json files (project + user level).

Config search order (project-level overrides user-level by server name):
  1. ~/.squishy/mcp.json             — user-level, lowest priority
  2. <cwd>/.mcp.json                 — project-level, highest priority
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

from .types import MCPServerConfig


USER_MCP_CONFIG  = Path.home() / ".squishy" / "mcp.json"
PROJECT_MCP_NAME = ".mcp.json"


def _atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_file(path: Path) -> Dict[str, dict]:
    """Read a single mcp.json file and return the mcpServers dict."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("mcpServers", {})
    except Exception:
        return {}


def load_mcp_configs() -> Dict[str, MCPServerConfig]:
    """Return all MCP server configs, project-level overriding user-level."""
    servers: Dict[str, dict] = _load_file(USER_MCP_CONFIG)

    # Walk up from cwd to find .mcp.json (up to 10 levels)
    p = Path.cwd()
    for _ in range(10):
        candidate = p / PROJECT_MCP_NAME
        if candidate.exists():
            project_servers = _load_file(candidate)
            servers.update(project_servers)
            break
        parent = p.parent
        if parent == p:
            break
        p = parent

    return {
        name: MCPServerConfig.from_dict(name, raw)
        for name, raw in servers.items()
    }


def save_user_mcp_config(servers: Dict[str, dict]) -> None:
    """Write (or update) the user-level MCP config file."""
    USER_MCP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if USER_MCP_CONFIG.exists():
        try:
            existing = json.loads(USER_MCP_CONFIG.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing["mcpServers"] = servers
    _atomic_write(USER_MCP_CONFIG, json.dumps(existing, indent=2))


def add_server_to_user_config(name: str, raw: dict) -> None:
    """Append or update one server entry in the user MCP config."""
    existing: dict = {}
    if USER_MCP_CONFIG.exists():
        try:
            existing = json.loads(USER_MCP_CONFIG.read_text(encoding="utf-8"))
        except Exception:
            pass
    mcp_servers = existing.get("mcpServers", {})
    mcp_servers[name] = raw
    existing["mcpServers"] = mcp_servers
    USER_MCP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(USER_MCP_CONFIG, json.dumps(existing, indent=2))


def remove_server_from_user_config(name: str) -> bool:
    """Remove a server from the user MCP config. Returns True if found."""
    if not USER_MCP_CONFIG.exists():
        return False
    try:
        existing = json.loads(USER_MCP_CONFIG.read_text(encoding="utf-8"))
        mcp_servers = existing.get("mcpServers", {})
        if name not in mcp_servers:
            return False
        del mcp_servers[name]
        existing["mcpServers"] = mcp_servers
        _atomic_write(USER_MCP_CONFIG, json.dumps(existing, indent=2))
        return True
    except Exception:
        return False


def list_config_files() -> List[Path]:
    """Return paths of all mcp.json config files that exist."""
    found = []
    if USER_MCP_CONFIG.exists():
        found.append(USER_MCP_CONFIG)
    p = Path.cwd()
    for _ in range(10):
        candidate = p / PROJECT_MCP_NAME
        if candidate.exists():
            found.append(candidate)
            break
        parent = p.parent
        if parent == p:
            break
        p = parent
    return found
