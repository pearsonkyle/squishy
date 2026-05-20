"""``squishy-acp`` — stdio ACP agent entry point.

Editors spawn this binary as a child process and speak ACP over stdin/stdout.
Logs go to stderr so they don't corrupt the JSON-RPC stream.

Examples
--------
Zed (in ``~/.config/zed/settings.json``)::

    "agent_servers": {
      "Squishy": {
        "command": "squishy-acp",
        "args": ["--base-url", "http://localhost:1234/v1"]
      }
    }

Neovim ACP plugins point at the same command. Configuration mirrors the
regular ``squishy`` CLI (``--model``, ``--base-url``, ``--temperature``, …)
but no TTY is needed — all output is structured.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys

from dotenv import load_dotenv

from squishy.acp.agent import SquishyAcpAgent
from squishy.config import Config


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="squishy-acp",
        description="Run squishy as an ACP agent over stdio.",
    )
    p.add_argument("--base-url", help="OpenAI-compatible endpoint (env SQUISHY_BASE_URL)")
    p.add_argument("--model", help="Model id (env SQUISHY_MODEL)")
    p.add_argument("--api-key", help="API key (env SQUISHY_API_KEY)")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument(
        "--mode",
        choices=("plan", "edits", "yolo"),
        default=None,
        help="Initial permission mode (default: plan)",
    )
    p.add_argument(
        "--sandbox",
        action="store_true",
        help="Use Docker sandbox for run_command when no editor terminal is available",
    )
    p.add_argument(
        "--log-level", default="warning",
        help="Log level for stderr diagnostics (debug/info/warning/error)",
    )
    return p.parse_args(argv)


def _build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.base_url:
        cfg.base_url = args.base_url
    if args.model:
        cfg.model = args.model
    if args.api_key:
        cfg.api_key = args.api_key
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.mode is not None:
        cfg.permission_mode = args.mode  # type: ignore[assignment]
    if args.sandbox:
        cfg.use_sandbox = True
    return cfg


async def _serve(args: argparse.Namespace) -> None:
    # Import here so the legacy CLI doesn't take the import cost when ACP is
    # never used. Errors here surface as a single line on stderr instead of
    # leaking a Python traceback through the JSON-RPC stream.
    import acp

    cfg = _build_config(args)
    agent = SquishyAcpAgent(cfg)
    try:
        await acp.run_agent(agent)
    finally:
        await agent.aclose()


def run(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_serve(args))


if __name__ == "__main__":
    run()
