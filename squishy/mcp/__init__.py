"""MCP (Model Context Protocol) support for squishy.

MCP servers are configured in JSON files:
  ~/.squishy/mcp.json         (user-level, all projects)
  .mcp.json                   (project-level, overrides user)

Supported transports: stdio, sse, http (also "remote" as alias for http).
MCP tools are registered as mcp__<server>__<tool> and callable like built-in tools.
"""
from .client import MCPClient, MCPManager, get_mcp_manager  # noqa: F401
from .config import (  # noqa: F401
    add_server_to_user_config,
    list_config_files,
    load_mcp_configs,
    remove_server_from_user_config,
    save_user_mcp_config,
)
from .tools import get_connect_errors, get_mcp_tools, initialize_mcp, reload_mcp  # noqa: F401
from .types import MCPServerConfig, MCPServerState, MCPTool, MCPTransport  # noqa: F401
