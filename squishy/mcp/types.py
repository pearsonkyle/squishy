"""MCP type definitions: server configs, tool descriptors, connection state."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ── Server config ─────────────────────────────────────────────────────────────

class MCPTransport(str, Enum):
    STDIO = "stdio"
    SSE   = "sse"
    HTTP  = "http"


_TRANSPORT_ALIASES: dict[str, str] = {
    "remote": "http",
    "streamable-http": "http",
}


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server.

    Stdio example:
        {"type": "stdio", "command": "uvx", "args": ["mcp-server-git"]}

    HTTP/remote example:
        {"type": "remote", "url": "https://mcp.context7.com/mcp",
         "headers": {"CONTEXT7_API_KEY": "..."}}
    """
    name: str
    transport: MCPTransport = MCPTransport.STDIO
    # stdio fields
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # sse / http / ws fields
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    # optional
    timeout: int = 30
    disabled: bool = False

    @classmethod
    def from_dict(cls, name: str, d: dict) -> MCPServerConfig:
        transport_str = d.get("type", "stdio").lower()
        transport_str = _TRANSPORT_ALIASES.get(transport_str, transport_str)
        try:
            transport = MCPTransport(transport_str)
        except ValueError:
            transport = MCPTransport.STDIO
        return cls(
            name=name,
            transport=transport,
            command=d.get("command", ""),
            args=d.get("args", []),
            env=d.get("env", {}),
            url=d.get("url", ""),
            headers=d.get("headers", {}),
            timeout=int(d.get("timeout", 30)),
            disabled=bool(d.get("disabled", False)),
        )


# ── Connection state ──────────────────────────────────────────────────────────

class MCPServerState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING   = "connecting"
    CONNECTED    = "connected"
    ERROR        = "error"


# ── Tool descriptor ───────────────────────────────────────────────────────────

@dataclass
class MCPTool:
    """A tool provided by an MCP server."""
    server_name: str
    tool_name: str
    qualified_name: str             # mcp__<server>__<tool>
    description: str
    input_schema: dict[str, Any]
    read_only: bool = False



# ── JSON-RPC helpers ──────────────────────────────────────────────────────────

def make_request(method: str, params: dict | None, req_id: int) -> dict:
    msg: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def make_notification(method: str, params: dict | None = None) -> dict:
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


MCP_PROTOCOL_VERSION = "2024-11-05"

CLIENT_INFO = {
    "name": "squishy",
    "version": "1.0.0",
}

INIT_PARAMS = {
    "protocolVersion": MCP_PROTOCOL_VERSION,
    "capabilities": {
        "tools": {},
        "roots": {"listChanged": False},
    },
    "clientInfo": CLIENT_INFO,
}
