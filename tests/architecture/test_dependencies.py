"""Architecture invariants from docs/ARCHITECTURE.md, enforced as tests.

Dependencies point inward: every layer may depend on ``model``; ``model``
depends on no one; no adapter imports another adapter.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

GLYPH = Path(__file__).resolve().parents[2] / "glyph"

INWARD_LAYERS = (
    "glyph.extract",
    "glyph.store",
    "glyph.retrieval",
    "glyph.baseline",
    "glyph.eval",
)


def _modules_under(package: str) -> Iterator[Path]:
    return (GLYPH / package).rglob("*.py")


def _imported_modules(pyfile: Path) -> set[str]:
    tree = ast.parse(pyfile.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_model_does_not_import_outer_layers() -> None:
    for pyfile in _modules_under("model"):
        offenders = {m for m in _imported_modules(pyfile) if m.startswith(INWARD_LAYERS)}
        assert not offenders, f"{pyfile.name} imports outer layer(s): {offenders}"


def test_store_adapter_does_not_import_extract() -> None:
    for pyfile in _modules_under("store"):
        offenders = {m for m in _imported_modules(pyfile) if m.startswith("glyph.extract")}
        assert not offenders, f"{pyfile.name} imports extract: {offenders}"


def test_extract_adapter_does_not_import_store() -> None:
    for pyfile in _modules_under("extract"):
        offenders = {m for m in _imported_modules(pyfile) if m.startswith("glyph.store")}
        assert not offenders, f"{pyfile.name} imports store: {offenders}"
