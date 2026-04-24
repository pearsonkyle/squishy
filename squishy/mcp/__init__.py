"""MCP (Model Context Protocol) support for squishy.

MCP servers are configured in JSON files:
  ~/.squishy/mcp.json         (user-level, all projects)
  .mcp.json                   (project-level, overrides user)

Supported transports: stdio, sse, http (also "remote" as alias for http).
MCP tools are registered as mcp__<server>__<tool> and callable like built-in tools.
"""
from .types import MCPServerConfig, MCPTool, MCPServerState, MCPTransport  # noqa: F401
from .client import MCPClient, MCPManager, get_mcp_manager                 # noqa: F401
from .config import (                                                       # noqa: F401
    load_mcp_configs,
    save_user_mcp_config,
    add_server_to_user_config,
    remove_server_from_user_config,
    list_config_files,
)
from .tools import initialize_mcp, reload_mcp, get_mcp_tools, get_connect_errors  # noqa: F401
