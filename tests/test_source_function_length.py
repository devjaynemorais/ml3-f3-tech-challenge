"""Architecture guard for the logical size of production functions."""

from __future__ import annotations

import ast
from pathlib import Path

MAX_LOGICAL_LINES = 20
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _collect_statement_lines(node: ast.AST, lines: set[int]) -> None:
    if isinstance(node, (*FUNCTION_NODES, ast.ClassDef, ast.Lambda)):
        return
    if isinstance(node, ast.stmt):
        lines.add(node.lineno)
    for child in ast.iter_child_nodes(node):
        _collect_statement_lines(child, lines)


def _logical_line_count(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    lines: set[int] = set()
    for statement in function.body:
        _collect_statement_lines(statement, lines)
    return len(lines)


def _function_violations(source_root: Path) -> list[str]:
    violations = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, FUNCTION_NODES):
                count = _logical_line_count(node)
                if count > MAX_LOGICAL_LINES:
                    violations.append(f"{path}:{node.lineno} {node.name} ({count})")
    return violations


def test_logical_line_counter_ignores_nested_functions(tmp_path: Path) -> None:
    source = "def outer():\n    def inner():\n" + "        value = 1\n" * 21
    path = tmp_path / "nested.py"
    path.write_text(source, encoding="utf-8")

    assert _function_violations(tmp_path) == [f"{path}:2 inner (21)"]


def test_src_functions_have_at_most_twenty_logical_lines() -> None:
    violations = _function_violations(Path("src"))

    assert not violations, "Functions above 20 logical lines:\n" + "\n".join(violations)
