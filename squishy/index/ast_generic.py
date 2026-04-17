"""Cheap regex fallback for non-Python source.
 
Covers the 80% case without pulling in tree-sitter: grab `function`, `class`,
`fn`, `func`, `def` declarations plus their leading `//` / `/** ... */` / `#`
comment block. Line numbers are approximate; good enough for `recall`.
"""
 
from __future__ import annotations
 
import re
from dataclasses import dataclass
 
from squishy.index.ast_py import Symbol
 
 
@dataclass
class _Lang:
    line_comment: str
    block_open: str = ""
    block_close: str = ""
    patterns: tuple[re.Pattern[str], ...] = ()
 
 
_JS_TS = _Lang(
    line_comment="//",
    block_open="/*",
    block_close="*/",
    patterns=(
        re.compile(r"^\s*export\s+(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"^\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"^\s*export\s+(?:default\s+)?class\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"^\s*class\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"^\s*export\s+const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\("),
        re.compile(r"^\s*const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\("),
    ),
)
 
_GO = _Lang(
    line_comment="//",
    block_open="/*",
    block_close="*/",
    patterns=(
        re.compile(r"^func\s+(?:\([^)]*\)\s+)?([A-Za-z_][\w]*)"),
        re.compile(r"^type\s+([A-Za-z_][\w]*)\s+(?:struct|interface)"),
    ),
)
 
_RUST = _Lang(
    line_comment="//",
    block_open="/*",
    block_close="*/",
    patterns=(
        re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][\w]*)"),
        re.compile(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_][\w]*)"),
        re.compile(r"^\s*(?:pub\s+)?enum\s+([A-Za-z_][\w]*)"),
        re.compile(r"^\s*(?:pub\s+)?trait\s+([A-Za-z_][\w]*)"),
    ),
)
 
_C_LIKE = _Lang(
    line_comment="//",
    block_open="/*",
    block_close="*/",
    patterns=(
        # crude: `type name(` at column 0
        re.compile(r"^[A-Za-z_][\w\s\*]+\s+([A-Za-z_][\w]*)\s*\([^;]*$"),
        re.compile(r"^\s*(?:class|struct)\s+([A-Za-z_][\w]*)"),
    ),
)
 
_SHELL = _Lang(
    line_comment="#",
    patterns=(
        re.compile(r"^\s*([A-Za-z_][\w]*)\s*\(\)\s*\{"),
        re.compile(r"^\s*function\s+([A-Za-z_][\w]*)"),
    ),
)
 
_RUBY = _Lang(
    line_comment="#",
    patterns=(
        re.compile(r"^\s*def\s+([A-Za-z_][\w]*[?!=]?)"),
        re.compile(r"^\s*class\s+([A-Za-z_][\w:]*)"),
        re.compile(r"^\s*module\s+([A-Za-z_][\w:]*)"),
    ),
)
 
_JAVA_KT = _Lang(
    line_comment="//",
    block_open="/*",
    block_close="*/",
    patterns=(
        re.compile(r"^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?(?:final\s+)?class\s+([A-Za-z_][\w]*)"),
        re.compile(r"^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?(?:suspend\s+)?fun\s+([A-Za-z_][\w]*)"),
    ),
)
 
_BY_EXT: dict[str, _Lang] = {
    ".js": _JS_TS, ".jsx": _JS_TS, ".mjs": _JS_TS, ".cjs": _JS_TS,
    ".ts": _JS_TS, ".tsx": _JS_TS,
    ".go": _GO,
    ".rs": _RUST,
    ".c": _C_LIKE, ".h": _C_LIKE,
    ".cc": _C_LIKE, ".hh": _C_LIKE,
    ".cpp": _C_LIKE, ".hpp": _C_LIKE,
    ".java": _JAVA_KT, ".kt": _JAVA_KT, ".kts": _JAVA_KT,
    ".rb": _RUBY,
    ".sh": _SHELL, ".bash": _SHELL,
}
 
 
def header_comment(source: str, ext: str) -> str:
    """Return the leading comment block of a source file, trimmed."""
    lang = _BY_EXT.get(ext.lower())
    if lang is None:
        return ""
    lines = source.splitlines()
    out: list[str] = []
 
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
 
    if lang.block_open and i < len(lines) and lang.block_open in lines[i]:
        while i < len(lines):
            text = lines[i]
            cleaned = text.strip().lstrip("/").lstrip("*").strip()
            if cleaned:
                out.append(cleaned)
            if lang.block_close in text:
                break
            i += 1
    else:
        while i < len(lines):
            line = lines[i].strip()
            if not line.startswith(lang.line_comment):
                break
            out.append(line.lstrip(lang.line_comment).strip())
            i += 1
 
    return " ".join(s for s in out if s)[:400]
 
 
def extract_symbols(source: str, ext: str) -> list[Symbol]:
    lang = _BY_EXT.get(ext.lower())
    if lang is None:
        return []
    out: list[Symbol] = []
    seen: set[tuple[str, int]] = set()
    lines = source.splitlines()
    for i, line in enumerate(lines, start=1):
        for pat in lang.patterns:
            m = pat.match(line)
            if m:
                name = m.group(1)
                key = (name, i)
                if key in seen:
                    continue
                seen.add(key)
                kind = "class" if ("class" in pat.pattern or "struct" in pat.pattern
                                   or "enum" in pat.pattern or "trait" in pat.pattern
                                   or "module" in pat.pattern or "type" in pat.pattern) else "function"
                out.append(Symbol(kind=kind, name=name, start_line=i, end_line=i))
                break
    return out
 
 
__all__ = ["extract_symbols", "header_comment"]
