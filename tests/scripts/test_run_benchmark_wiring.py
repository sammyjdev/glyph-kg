"""P3.5 wiring check: every render_markdown(...) call site in run_benchmark.py must
pass confusion=, or the AC2 error-rate section silently never reaches METRICS.md.

Static (ast) rather than a live run, since main() isn't easily unit-testable end-to-end.
"""

import ast
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_benchmark.py"


def _render_markdown_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "render_markdown"
    ]


def test_every_render_markdown_call_passes_confusion() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    calls = _render_markdown_calls(tree)
    assert calls, "expected at least one render_markdown(...) call in run_benchmark.py"

    missing = [
        call.lineno for call in calls if not any(kw.arg == "confusion" for kw in call.keywords)
    ]
    assert not missing, (
        f"render_markdown(...) call(s) at line(s) {missing} in {SCRIPT_PATH} "
        "do not pass confusion= — the P3.5 error-rate section will not reach METRICS.md"
    )
