"""P4.4: the code-domain corpus for the vector arm.

Two granularities:

- ``code_documents``: one document per source file (posix relpath key). The
  original baseline — the graphify-vs-glyph benchmark (2026-07-26) measured
  this whole-file granularity as a handicap: it lost to symbol-level
  retrieval CI-separated on every stratum. Kept for comparison runs.
- ``code_symbol_documents`` (#51): one document per CLASS/FUNCTION symbol,
  sliced from the CP-1 provenance span and keyed by the graph node id — so
  hybrid RRF fusion keys align across the graph and vector arms.
"""

from collections.abc import Sequence
from pathlib import Path

from glyph.model.node import NodeType

_SOURCE_SUFFIXES = (".py", ".java")


def code_documents(repo_path: str | Path) -> list[tuple[str, str]]:
    """Return ``(relpath, source_text)`` for each non-empty source file under ``repo_path``."""
    root = Path(repo_path)
    files: Sequence[Path] = sorted(
        p for p in root.rglob("*") if p.suffix in _SOURCE_SUFFIXES and p.is_file()
    )
    documents: list[tuple[str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if text.strip():
            documents.append((path.relative_to(root).as_posix(), text))
    return documents


def code_symbol_documents(repo_path: str | Path) -> list[tuple[str, str]]:
    """Return ``(node_id, symbol_source_text)`` per CLASS/FUNCTION symbol (#51).

    Reuses the tree-sitter extraction; the text is the CP-1 provenance span
    sliced from the source file. Sorted by node id for determinism.
    """
    from glyph.extract.code import CodeExtractor

    root = Path(repo_path)
    nodes, _edges = CodeExtractor().extract(root)
    file_lines: dict[str, list[str]] = {}
    documents: list[tuple[str, str]] = []
    for node in nodes:
        if node.type not in (NodeType.CLASS, NodeType.FUNCTION):
            continue
        file = node.attrs["file"]
        if file not in file_lines:
            file_lines[file] = (root / file).read_text(encoding="utf-8").splitlines(keepends=True)
        text = "".join(file_lines[file][node.attrs["start_line"] - 1 : node.attrs["end_line"]])
        if text.strip():
            documents.append((node.id, text))
    return sorted(documents)
