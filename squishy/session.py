"""Session persistence for conversation history.

Saves every conversation to disk so sessions can be:
- Inspected for benchmark auditing
- Resumed by UUID
- Exported as LLM training data (JSONL with tool_calls, reasoning)

Storage layout::

    ~/.squishy/sessions/<uuid>/
        meta.json        — session metadata
        messages.jsonl   — append-only, one message per line
        tools.json       — tool schemas (written once at start)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEFAULT_DIR = os.path.expanduser("~/.squishy/sessions")


def session_dir(override: str | None = None) -> Path:
    """Return the root sessions directory, creating it if needed."""
    d = Path(override or os.environ.get("SQUISHY_SESSION_DIR", _DEFAULT_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Session:
    id: str
    model: str
    created_at: str
    updated_at: str
    working_dir: str
    mode: str
    status: str = "active"
    turns: int = 0
    tokens: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_path(session_id: str, root: str | None = None) -> Path:
    return session_dir(root) / session_id


def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Normalize a message for storage.

    Converts tool_calls[].function.arguments from JSON strings to dicts
    so the stored format matches trainer expectations.
    """
    out = dict(msg)
    if "tool_calls" in out and out["tool_calls"]:
        new_calls = []
        for tc in out["tool_calls"]:
            tc = dict(tc)
            func = tc.get("function")
            if isinstance(func, dict):
                func = dict(func)
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        func["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        pass
                tc["function"] = func
            new_calls.append(tc)
        out["tool_calls"] = new_calls
    return out


def create_session(
    model: str,
    working_dir: str,
    mode: str,
    tools: list[dict[str, Any]] | None = None,
    *,
    root: str | None = None,
) -> Session:
    """Create a new session directory and write initial metadata."""
    import uuid

    sid = uuid.uuid4().hex
    d = _session_path(sid, root)
    d.mkdir(parents=True, exist_ok=True)

    now = _now_iso()
    sess = Session(
        id=sid,
        model=model,
        created_at=now,
        updated_at=now,
        working_dir=working_dir,
        mode=mode,
    )
    (d / "meta.json").write_text(json.dumps(asdict(sess), indent=2))

    if tools:
        (d / "tools.json").write_text(json.dumps(tools, indent=2, ensure_ascii=False))

    return sess


def append_messages(
    session_id: str,
    messages: list[dict[str, Any]],
    *,
    root: str | None = None,
) -> None:
    """Append messages to the session's messages.jsonl (append-only)."""
    if not messages:
        return
    d = _session_path(session_id, root)
    with (d / "messages.jsonl").open("a", encoding="utf-8") as f:
        for msg in messages:
            normalized = _normalize_message(msg)
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def finish_session(
    session_id: str,
    *,
    status: str = "completed",
    turns: int = 0,
    tokens: int = 0,
    root: str | None = None,
) -> None:
    """Update session metadata with final stats."""
    d = _session_path(session_id, root)
    meta_path = d / "meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        meta = {"id": session_id}
    meta["status"] = status
    meta["turns"] = turns
    meta["tokens"] = tokens
    meta["updated_at"] = _now_iso()
    meta_path.write_text(json.dumps(meta, indent=2))


def load_session(session_id: str, *, root: str | None = None) -> Session:
    """Load session metadata by UUID."""
    d = _session_path(session_id, root)
    meta = json.loads((d / "meta.json").read_text())
    return Session(**{k: meta[k] for k in Session.__dataclass_fields__ if k in meta})


def load_messages(session_id: str, *, root: str | None = None) -> list[dict[str, Any]]:
    """Load all messages from a session."""
    d = _session_path(session_id, root)
    msgs_path = d / "messages.jsonl"
    if not msgs_path.exists():
        return []
    messages = []
    for line in msgs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            messages.append(json.loads(line))
    return messages


def load_tools(session_id: str, *, root: str | None = None) -> list[dict[str, Any]]:
    """Load tool schemas from a session."""
    d = _session_path(session_id, root)
    tools_path = d / "tools.json"
    if not tools_path.exists():
        return []
    return json.loads(tools_path.read_text())


def list_sessions(
    *,
    limit: int = 20,
    working_dir: str | None = None,
    root: str | None = None,
) -> list[Session]:
    """List recent sessions, newest first."""
    sd = session_dir(root)
    sessions: list[Session] = []
    for entry in sd.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            sess = Session(**{k: meta[k] for k in Session.__dataclass_fields__ if k in meta})
            if working_dir and sess.working_dir != working_dir:
                continue
            sessions.append(sess)
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    sessions.sort(key=lambda s: s.updated_at, reverse=True)
    return sessions[:limit]


def export_training(
    session_id: str,
    *,
    root: str | None = None,
    strip_system: bool = True,
) -> dict[str, Any]:
    """Export a session as a training-ready record.

    Returns ``{"messages": [...]}`` in the format expected by
    ``llmtk/pipeline/cleanup.py:format_for_training()`` and the trainer:
    - Tool schemas embedded in ``messages[0]["tools"]``
    - ``tool_calls[].function.arguments`` as dicts (already normalized on write)
    - ``think`` key preserved on assistant messages with reasoning
    - System messages stripped (except first user message gets tool schemas)

    The output can be appended to a ``.jsonl`` file and used directly as
    ``local_json:`` in ``train.yaml``.
    """
    messages = load_messages(session_id, root=root)
    tools = load_tools(session_id, root=root)

    if strip_system:
        messages = [m for m in messages if m.get("role") != "system"]

    if not messages:
        return {"messages": []}

    # Embed tool schemas in first message (trainer convention).
    if tools and messages:
        messages[0] = {**messages[0], "tools": tools}

    # Ensure conversation ends with assistant message (trainer requirement).
    # Drop trailing tool messages if needed.
    while messages and messages[-1].get("role") not in ("assistant",):
        if messages[-1].get("role") == "tool":
            messages.pop()
        else:
            break

    return {"messages": messages}


def export_training_to_file(
    session_id: str,
    output_path: str | Path,
    *,
    root: str | None = None,
) -> Path:
    """Export a session to a JSONL file. Returns the output path."""
    record = export_training(session_id, root=root)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out
