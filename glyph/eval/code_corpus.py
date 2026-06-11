"""P4.4: the code-domain corpus for the vector arm — one document per source file.

The graph arm surfaces symbols + relations; the vector arm needs comparable text to
retrieve over. We index whole source files keyed by their posix relpath (matching the
code graph's file-node ids). Per-file is coarser than the graph's per-symbol units —
a declared granularity asymmetry, measured by the benchmark, not hidden.
"""

from collections.abc import Sequence
from pathlib import Path

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
