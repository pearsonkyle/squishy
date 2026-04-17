"""Python AST symbol extraction."""
 
from __future__ import annotations
 
from squishy.index.ast_py import extract_symbols, module_docstring
 
 
def test_module_docstring() -> None:
    src = '"""Top-level doc."""\n\ndef x(): pass\n'
    assert module_docstring(src) == "Top-level doc."
 
 
def test_extract_functions_and_classes() -> None:
    src = (
        '"""mod."""\n'
        "def alpha():\n"
        '    "a"\n'
        "    return 1\n"
        "\n"
        "async def beta():\n"
        "    return 2\n"
        "\n"
        "class C:\n"
        '    """cls."""\n'
        "    def m1(self): return 1\n"
        "    async def m2(self): return 2\n"
    )
    syms = extract_symbols(src)
    names = [(s.kind, s.name) for s in syms]
    assert ("function", "alpha") in names
    assert ("function", "beta") in names
    assert ("class", "C") in names
 
    c = next(s for s in syms if s.name == "C")
    method_names = {m.name for m in (c.children or [])}
    assert {"m1", "m2"} <= method_names
    assert c.docstring == "cls."
 
 
def test_syntax_error_returns_empty() -> None:
    assert extract_symbols("def (:::") == []
    assert module_docstring("def (:::") == ""
 
 
def test_line_numbers_reasonable() -> None:
    src = "def first():\n    pass\n\ndef second():\n    pass\n"
    syms = extract_symbols(src)
    by_name = {s.name: s for s in syms}
    assert by_name["first"].start_line == 1
    assert by_name["second"].start_line == 4
    assert by_name["first"].end_line >= by_name["first"].start_line
