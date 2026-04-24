"""Register MCP tools into squishy's tool registry.

MCP tool qualified names follow the pattern: mcp__<server_name>__<tool_name>
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

from squishy.tools.base import Tool, ToolContext, ToolResult
from .client import get_mcp_manager
from .config import load_mcp_configs
from .types import MCPTool

log = logging.getLogger("squishy.mcp")

_initialized = False
_init_lock = threading.Lock()
_connect_errors: Dict[str, Optional[str]] = {}
_mcp_tools: List[Tool] = []


def _make_mcp_runner(qualified_name: str):
    """Return an async tool runner that delegates to the MCP server."""
    async def _run(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        mgr = get_mcp_manager()
        try:
            result = await asyncio.to_thread(mgr.call_tool, qualified_name, args)
            return ToolResult(True, data={"content": result})
        except Exception as e:
            return ToolResult(False, error=f"MCP tool error: {e}")
    return _run


def _build_tool(mcp_tool: MCPTool) -> Tool:
    """Convert an MCPTool descriptor into a squishy Tool."""
    return Tool(
        name=mcp_tool.qualified_name,
        description=f"[MCP:{mcp_tool.server_name}] {mcp_tool.description}",
        parameters=mcp_tool.input_schema or {"type": "object", "properties": {}},
        run=_make_mcp_runner(mcp_tool.qualified_name),
    )


def _register_tools_into_squishy(tools: list[Tool]) -> None:
    """Add MCP tools to squishy's global ALL_TOOLS and REGISTRY."""
    from squishy.tools import ALL_TOOLS, REGISTRY

    # Remove previously-registered MCP tools (for reload).
    mcp_ids = {id(t) for t in _mcp_tools}
    ALL_TOOLS[:] = [t for t in ALL_TOOLS if id(t) not in mcp_ids]
    for old in _mcp_tools:
        REGISTRY.pop(old.name, None)
    _mcp_tools.clear()

    # Add new tools.
    for tool in tools:
        ALL_TOOLS.append(tool)
        REGISTRY[tool.name] = tool
        _mcp_tools.append(tool)


def initialize_mcp(verbose: bool = False) -> Dict[str, Optional[str]]:
    """Load configs, connect servers, register tools. Idempotent."""
    global _initialized, _connect_errors

    with _init_lock:
        if _initialized:
            return _connect_errors

        configs = load_mcp_configs()
        if not configs:
            _initialized = True
            return {}

        mgr = get_mcp_manager()
        for cfg in configs.values():
            mgr.add_server(cfg)

        errors = mgr.connect_all()
        _connect_errors = errors

        new_tools = []
        for client in mgr.list_servers():
            if client.state.value == "connected":
                for mcp_tool in client._tools:
                    new_tools.append(_build_tool(mcp_tool))
                if verbose:
                    log.info("%s: %d tool(s)", client.config.name, len(client._tools))

        _register_tools_into_squishy(new_tools)
        _initialized = True
        return errors


def reload_mcp() -> Dict[str, Optional[str]]:
    """Force reload: re-read configs, reconnect, re-register."""
    global _initialized
    with _init_lock:
        _initialized = False
    return initialize_mcp()


def get_mcp_tools() -> list[Tool]:
    """Return the currently registered MCP tools."""
    return list(_mcp_tools)


def get_connect_errors() -> Dict[str, Optional[str]]:
    return dict(_connect_errors)
