"""Python symbol extraction via stdlib `ast`.
 
Surfaces top-level classes, functions, and async functions. Captures:
- name + line range
- docstring (if any) for a free LLM-free summary
- nested methods one level down (so `ClassName.method` shows up as a sub-symbol)
"""
 
from __future__ import annotations
 
import ast
from dataclasses import dataclass
 
 
@dataclass
class Symbol:
    kind: str  # "class" | "function" | "method"
    name: str
    start_line: int
    end_line: int
    docstring: str = ""
    children: list["Symbol"] | None = None
 
 
def _end_line(node: ast.AST, fallback: int) -> int:
    return getattr(node, "end_lineno", None) or fallback
 
 
def _docstring(node: ast.AST) -> str:
    if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        ds = ast.get_docstring(node)
        return ds.strip() if ds else ""
    return ""
 
 
def module_docstring(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    return ast.get_docstring(tree) or ""
 
 
def extract_symbols(source: str) -> list[Symbol]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[Symbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(Symbol(
                kind="function",
                name=node.name,
                start_line=node.lineno,
                end_line=_end_line(node, node.lineno),
                docstring=_docstring(node),
            ))
        elif isinstance(node, ast.ClassDef):
            methods: list[Symbol] = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(Symbol(
                        kind="method",
                        name=sub.name,
                        start_line=sub.lineno,
                        end_line=_end_line(sub, sub.lineno),
                        docstring=_docstring(sub),
                    ))
            out.append(Symbol(
                kind="class",
                name=node.name,
                start_line=node.lineno,
                end_line=_end_line(node, node.lineno),
                docstring=_docstring(node),
                children=methods or None,
            ))
    return out
 
 
__all__ = ["Symbol", "extract_symbols", "module_docstring"]
