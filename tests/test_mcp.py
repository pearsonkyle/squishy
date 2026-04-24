"""Tests for squishy.mcp — MCP integration."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from squishy.mcp.types import MCPServerConfig, MCPTool, MCPTransport
from squishy.mcp.tools import _build_tool, _make_mcp_runner, _register_tools_into_squishy, _mcp_tools
from squishy.tools.base import Tool, ToolContext


# ── types.py ──────────────────────────────────────────────────────────────────

class TestMCPServerConfig:
    def test_from_dict_stdio(self):
        cfg = MCPServerConfig.from_dict("my-git", {
            "type": "stdio",
            "command": "uvx",
            "args": ["mcp-server-git"],
        })
        assert cfg.name == "my-git"
        assert cfg.transport == MCPTransport.STDIO
        assert cfg.command == "uvx"
        assert cfg.args == ["mcp-server-git"]

    def test_from_dict_remote_alias(self):
        cfg = MCPServerConfig.from_dict("ctx7", {
            "type": "remote",
            "url": "https://mcp.context7.com/mcp",
            "headers": {"KEY": "val"},
        })
        assert cfg.transport == MCPTransport.HTTP
        assert cfg.url == "https://mcp.context7.com/mcp"
        assert cfg.headers == {"KEY": "val"}

    def test_from_dict_streamable_http_alias(self):
        cfg = MCPServerConfig.from_dict("x", {"type": "streamable-http", "url": "http://localhost:9000"})
        assert cfg.transport == MCPTransport.HTTP

    def test_from_dict_sse(self):
        cfg = MCPServerConfig.from_dict("x", {"type": "sse", "url": "http://localhost:8080/sse"})
        assert cfg.transport == MCPTransport.SSE

    def test_from_dict_unknown_type_defaults_stdio(self):
        cfg = MCPServerConfig.from_dict("x", {"type": "websocket-v9"})
        assert cfg.transport == MCPTransport.STDIO

    def test_from_dict_disabled(self):
        cfg = MCPServerConfig.from_dict("x", {"disabled": True})
        assert cfg.disabled is True

    def test_from_dict_timeout(self):
        cfg = MCPServerConfig.from_dict("x", {"timeout": 120})
        assert cfg.timeout == 120


class TestMCPTool:
    def test_to_tool_schema(self):
        tool = MCPTool(
            server_name="ctx7",
            tool_name="resolve_library_id",
            qualified_name="mcp__ctx7__resolve_library_id",
            description="Resolves a library ID",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        schema = tool.to_tool_schema()
        assert schema["name"] == "mcp__ctx7__resolve_library_id"
        assert "[MCP:ctx7]" in schema["description"]
        assert schema["input_schema"]["properties"]["name"]["type"] == "string"


# ── tools.py ──────────────────────────────────────────────────────────────────

class TestBuildTool:
    def test_produces_squishy_tool(self):
        mcp_tool = MCPTool(
            server_name="test",
            tool_name="my_tool",
            qualified_name="mcp__test__my_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
        )
        tool = _build_tool(mcp_tool)
        assert isinstance(tool, Tool)
        assert tool.name == "mcp__test__my_tool"
        assert "[MCP:test]" in tool.description
        assert callable(tool.run)

    @pytest.mark.asyncio
    async def test_async_runner(self):
        """The MCP runner wraps call_tool in asyncio.to_thread."""
        runner = _make_mcp_runner("mcp__srv__fn")

        mock_mgr = MagicMock()
        mock_mgr.call_tool.return_value = "hello from mcp"

        with patch("squishy.mcp.tools.get_mcp_manager", return_value=mock_mgr):
            ctx = ToolContext(working_dir="/tmp")
            result = await runner({"arg": 1}, ctx)

        assert result.success is True
        assert result.data["content"] == "hello from mcp"
        mock_mgr.call_tool.assert_called_once_with("mcp__srv__fn", {"arg": 1})

    @pytest.mark.asyncio
    async def test_async_runner_error(self):
        runner = _make_mcp_runner("mcp__srv__fn")

        mock_mgr = MagicMock()
        mock_mgr.call_tool.side_effect = RuntimeError("server down")

        with patch("squishy.mcp.tools.get_mcp_manager", return_value=mock_mgr):
            ctx = ToolContext(working_dir="/tmp")
            result = await runner({}, ctx)

        assert result.success is False
        assert "server down" in result.error


class TestDynamicRegistration:
    def test_register_and_remove(self):
        from squishy.tools import ALL_TOOLS, REGISTRY

        mcp_tool = MCPTool(
            server_name="test",
            tool_name="reg_test",
            qualified_name="mcp__test__reg_test",
            description="registration test",
            input_schema={"type": "object", "properties": {}},
        )
        tool = _build_tool(mcp_tool)
        initial_count = len(ALL_TOOLS)

        _register_tools_into_squishy([tool])
        assert "mcp__test__reg_test" in REGISTRY
        assert len(ALL_TOOLS) == initial_count + 1

        # Re-register (reload) should replace, not duplicate.
        _register_tools_into_squishy([tool])
        assert len(ALL_TOOLS) == initial_count + 1

        # Clean up.
        _register_tools_into_squishy([])
        assert "mcp__test__reg_test" not in REGISTRY
        assert len(ALL_TOOLS) == initial_count


# ── tool_restrictions.py ─────────────────────────────────────────────────────

class TestMCPPermissions:
    def test_mcp_tools_allowed_in_all_modes(self):
        from squishy.tool_restrictions import check_permission

        for mode in ("plan", "edits", "yolo", "bench"):
            allowed, reason = check_permission("mcp__ctx7__resolve", mode)
            assert allowed is True, f"MCP tool rejected in {mode} mode"
            assert reason == ""


# ── tools/__init__.py schemas ────────────────────────────────────────────────

class TestSchemaInclusion:
    def test_mcp_tools_in_schemas_all_modes(self):
        from squishy.tools import ALL_TOOLS, REGISTRY, openai_schemas

        mcp_tool = MCPTool(
            server_name="test",
            tool_name="schema_test",
            qualified_name="mcp__test__schema_test",
            description="schema inclusion test",
            input_schema={"type": "object", "properties": {}},
        )
        tool = _build_tool(mcp_tool)
        _register_tools_into_squishy([tool])

        try:
            for mode in ("plan", "edits", "yolo", "bench"):
                schemas = openai_schemas(mode)
                names = [s["function"]["name"] for s in schemas]
                assert "mcp__test__schema_test" in names, f"MCP tool missing from {mode} schemas"
        finally:
            _register_tools_into_squishy([])


# ── config.py ────────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_user_config(self, tmp_path: Path):
        mcp_json = tmp_path / "mcp.json"
        mcp_json.write_text(json.dumps({
            "mcpServers": {
                "my-server": {
                    "type": "stdio",
                    "command": "echo",
                    "args": ["hello"],
                }
            }
        }))

        with patch("squishy.mcp.config.USER_MCP_CONFIG", mcp_json):
            from squishy.mcp.config import load_mcp_configs
            configs = load_mcp_configs()
            assert "my-server" in configs
            assert configs["my-server"].command == "echo"

    def test_add_and_remove(self, tmp_path: Path):
        mcp_json = tmp_path / "mcp.json"
        with patch("squishy.mcp.config.USER_MCP_CONFIG", mcp_json):
            from squishy.mcp.config import add_server_to_user_config, remove_server_from_user_config
            add_server_to_user_config("test", {"type": "stdio", "command": "echo"})
            assert mcp_json.exists()
            data = json.loads(mcp_json.read_text())
            assert "test" in data["mcpServers"]

            assert remove_server_from_user_config("test") is True
            data = json.loads(mcp_json.read_text())
            assert "test" not in data["mcpServers"]

            assert remove_server_from_user_config("nonexistent") is False


# ── dispatch integration ─────────────────────────────────────────────────────

class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_mcp_tool(self):
        from squishy.tools import dispatch

        mcp_tool = MCPTool(
            server_name="test",
            tool_name="dispatch_test",
            qualified_name="mcp__test__dispatch_test",
            description="dispatch test",
            input_schema={"type": "object", "properties": {}},
        )
        tool = _build_tool(mcp_tool)
        _register_tools_into_squishy([tool])

        mock_mgr = MagicMock()
        mock_mgr.call_tool.return_value = "dispatched ok"

        try:
            with patch("squishy.mcp.tools.get_mcp_manager", return_value=mock_mgr):
                ctx = ToolContext(working_dir="/tmp", permission_mode="yolo")
                result = await dispatch("mcp__test__dispatch_test", {}, ctx)
            assert result.success is True
            assert result.data["content"] == "dispatched ok"
        finally:
            _register_tools_into_squishy([])
