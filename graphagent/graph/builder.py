"""Build a :class:`CodeGraph` from a Python repository using ``ast``.

Two passes, in the spirit of CodeGraph's extract-then-resolve pipeline:

1. **Extraction** — parse every ``.py`` file, emit file/class/function/method
   nodes with source spans, and record raw import names, base-class names,
   and call names per symbol.
2. **Resolution** — map raw names to node ids: imports to files, base
   classes and calls to symbols (module scope first, imported names next,
   then a globally unique name match as a last resort).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from graphagent.graph.models import EdgeKind, Node, NodeKind
from graphagent.graph.store import CodeGraph

_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
}


@dataclass(slots=True)
class _RawSymbol:
    """A symbol plus the unresolved names it references."""

    node: Node
    calls: set[str] = field(default_factory=set)
    bases: set[str] = field(default_factory=set)


@dataclass(slots=True)
class _RawFile:
    """Per-file extraction output awaiting cross-file resolution."""

    path: str
    module: str
    line_count: int = 0
    imports: set[str] = field(default_factory=set)
    imported_names: dict[str, str] = field(default_factory=dict)
    symbols: list[_RawSymbol] = field(default_factory=list)


def iter_python_files(root: Path) -> list[Path]:
    """Yield indexable ``.py`` files under ``root``, skipping junk dirs."""
    files: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def build_graph(root: Path) -> CodeGraph:
    """Index every Python file under ``root`` into a new graph."""
    root = root.resolve()
    graph = CodeGraph()
    raw_files: list[_RawFile] = []

    for file_path in iter_python_files(root):
        raw = _extract_file(root, file_path)
        if raw is not None:
            raw_files.append(raw)

    _register_nodes(graph, raw_files)
    _resolve_edges(graph, raw_files)
    return graph


# ------------------------------------------------------------- extraction ---


def _extract_file(root: Path, file_path: Path) -> _RawFile | None:
    rel = file_path.relative_to(root).as_posix()
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None

    module = rel[: -len(".py")].replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    raw = _RawFile(path=rel, module=module,
                   line_count=len(source.splitlines()))

    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                raw.imports.add(alias.name)
                raw.imported_names[alias.asname or alias.name] = alias.name
        elif isinstance(stmt, ast.ImportFrom) and stmt.module and stmt.level == 0:
            raw.imports.add(stmt.module)
            for alias in stmt.names:
                raw.imported_names[alias.asname or alias.name] = (
                    f"{stmt.module}.{alias.name}"
                )

    for node in tree.body:
        _extract_symbol(raw, node, parent_id=rel, parent_kind=NodeKind.FILE)
    return raw


def _extract_symbol(
    raw: _RawFile,
    node: ast.stmt,
    parent_id: str,
    parent_kind: NodeKind,
) -> None:
    if isinstance(node, ast.ClassDef):
        node_id = _child_id(parent_id, node.name)
        symbol = _RawSymbol(
            node=Node(
                node_id=node_id,
                kind=NodeKind.CLASS,
                name=node.name,
                path=raw.path,
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                docstring=ast.get_docstring(node),
                signature=f"class {node.name}",
            )
        )
        symbol.bases = {
            base_name for base in node.bases if (base_name := _name_of(base))
        }
        raw.symbols.append(symbol)
        for child in node.body:
            _extract_symbol(raw, child, parent_id=node_id, parent_kind=NodeKind.CLASS)
    elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        kind = (
            NodeKind.METHOD if parent_kind is NodeKind.CLASS else NodeKind.FUNCTION
        )
        node_id = _child_id(parent_id, node.name)
        symbol = _RawSymbol(
            node=Node(
                node_id=node_id,
                kind=kind,
                name=node.name,
                path=raw.path,
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                docstring=ast.get_docstring(node),
                signature=_signature_of(node),
            )
        )
        symbol.calls = _collect_call_names(node)
        raw.symbols.append(symbol)


def _child_id(parent_id: str, name: str) -> str:
    if "::" in parent_id:
        return f"{parent_id}.{name}"
    return f"{parent_id}::{name}"


def _name_of(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def _signature_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [a.arg for a in node.args.args]
    return f"def {node.name}({', '.join(args)})"


def _collect_call_names(func: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            name = _name_of(node.func)
            if name:
                names.add(name)
    return names


# ------------------------------------------------------------- resolution ---


def _register_nodes(graph: CodeGraph, raw_files: list[_RawFile]) -> None:
    for raw in raw_files:
        graph.add_node(
            Node(
                node_id=raw.path,
                kind=NodeKind.FILE,
                name=Path(raw.path).name,
                path=raw.path,
                lineno=1,
                # The file's real length. It was pinned at 1, which made every
                # file look one line long — harmless until something needed to
                # ask whether a file fits in a single read.
                end_lineno=raw.line_count,
            )
        )
        for symbol in raw.symbols:
            graph.add_node(symbol.node)
            parent_id = symbol.node.node_id.rsplit(".", 1)[0]
            if "::" not in parent_id or parent_id == symbol.node.node_id:
                parent_id = raw.path
            if graph.get(parent_id) is None:
                parent_id = raw.path
            graph.add_edge(parent_id, symbol.node.node_id, EdgeKind.CONTAINS)


def _resolve_edges(graph: CodeGraph, raw_files: list[_RawFile]) -> None:
    module_to_file = {raw.module: raw.path for raw in raw_files}
    name_index: dict[str, list[str]] = {}
    per_file_names: dict[str, dict[str, str]] = {}
    for raw in raw_files:
        local: dict[str, str] = {}
        for symbol in raw.symbols:
            name_index.setdefault(symbol.node.name, []).append(symbol.node.node_id)
            local.setdefault(symbol.node.name, symbol.node.node_id)
        per_file_names[raw.path] = local

    for raw in raw_files:
        for module in raw.imports:
            target = _resolve_module(module, module_to_file)
            if target:
                graph.add_edge(raw.path, target, EdgeKind.IMPORTS)

        for symbol in raw.symbols:
            for base in symbol.bases:
                target = _resolve_name(base, raw, per_file_names, name_index)
                if target:
                    graph.add_edge(symbol.node.node_id, target, EdgeKind.INHERITS)
            for call in symbol.calls:
                target = _resolve_name(call, raw, per_file_names, name_index)
                if target and target != symbol.node.node_id:
                    graph.add_edge(symbol.node.node_id, target, EdgeKind.CALLS)


def _resolve_module(module: str, module_to_file: dict[str, str]) -> str | None:
    if module in module_to_file:
        return module_to_file[module]
    # ``from pkg.mod import name`` records ``pkg.mod``; also try the parent
    # in case ``pkg.mod`` was actually ``pkg`` + attribute access.
    parent = module.rsplit(".", 1)[0]
    return module_to_file.get(parent)


def _resolve_name(
    name: str,
    raw: _RawFile,
    per_file_names: dict[str, dict[str, str]],
    name_index: dict[str, list[str]],
) -> str | None:
    # 1. Same-file symbol.
    local = per_file_names.get(raw.path, {})
    if name in local:
        return local[name]
    # 2. Explicitly imported name -> symbol in the source module.
    dotted = raw.imported_names.get(name)
    if dotted:
        module, _, attr = dotted.rpartition(".")
        candidates = name_index.get(attr or dotted, [])
        for candidate in candidates:
            if module and candidate.startswith(module.replace(".", "/")):
                return candidate
        if len(candidates) == 1:
            return candidates[0]
    # 3. Globally unique name.
    candidates = name_index.get(name, [])
    if len(candidates) == 1:
        return candidates[0]
    return None
