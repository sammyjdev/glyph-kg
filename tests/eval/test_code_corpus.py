"""Code corpus: per-file (relpath, source) documents for the code-domain vector arm."""

from pathlib import Path

from glyph.eval.code_corpus import code_documents


def test_code_documents_returns_relpath_and_source(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.java").write_text("class B {}\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignored", encoding="utf-8")

    docs = code_documents(tmp_path)

    assert (docs == [("b.java", "class B {}\n"), ("pkg/a.py", "def f():\n    return 1\n")]) or (
        docs == [("pkg/a.py", "def f():\n    return 1\n"), ("b.java", "class B {}\n")]
    )
    labels = [label for label, _ in docs]
    assert "notes.md" not in labels  # only source files
    assert set(labels) == {"pkg/a.py", "b.java"}


def test_code_documents_is_sorted_and_skips_empty(tmp_path: Path) -> None:
    (tmp_path / "z.py").write_text("z = 1\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "empty.py").write_text("", encoding="utf-8")

    docs = code_documents(tmp_path)

    assert [label for label, _ in docs] == ["a.py", "z.py"]  # sorted, empty skipped
