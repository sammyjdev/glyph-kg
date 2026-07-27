"""Code corpus: per-file (relpath, source) documents for the code-domain vector arm."""

from pathlib import Path

import pytest

from glyph.eval.code_corpus import code_documents, code_symbol_documents


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


def test_symbol_documents_slice_symbol_spans(tmp_path: Path) -> None:
    # #51: the retrieval unit is the symbol, keyed by node id so hybrid
    # fusion keys align with the graph arm (benchmark-verified handicap).
    pytest.importorskip("tree_sitter")
    (tmp_path / "a.py").write_text(
        "class Base:\n    def greet(self):\n        pass\n\ndef helper():\n    return 1\n",
        encoding="utf-8",
    )

    docs = dict(code_symbol_documents(tmp_path))

    assert docs["a.py::Base"] == "class Base:\n    def greet(self):\n        pass\n"
    assert docs["a.py::Base.greet"] == "    def greet(self):\n        pass\n"
    assert docs["a.py::helper"] == "def helper():\n    return 1\n"


def test_symbol_documents_exclude_file_and_module_nodes(tmp_path: Path) -> None:
    pytest.importorskip("tree_sitter")
    (tmp_path / "a.py").write_text("import os\n\ndef f():\n    return 1\n", encoding="utf-8")

    docs = code_symbol_documents(tmp_path)

    ids = [doc_id for doc_id, _ in docs]
    assert ids == ["a.py::f"]  # no FILE doc, no module:os doc, sorted


def test_symbol_documents_java(tmp_path: Path) -> None:
    pytest.importorskip("tree_sitter")
    (tmp_path / "W.java").write_text("class W {\n    void run() {\n    }\n}\n", encoding="utf-8")

    docs = dict(code_symbol_documents(tmp_path))

    assert docs["W.java::W"] == "class W {\n    void run() {\n    }\n}\n"
    assert docs["W.java::W.run"] == "    void run() {\n    }\n"


def test_symbol_documents_context_lines_pad_the_span(tmp_path: Path) -> None:
    # #55 H1: fixed-size chunks carry neighboring context that exact symbol
    # spans lack; context_lines pads the slice, clamped to file bounds.
    pytest.importorskip("tree_sitter")
    (tmp_path / "a.py").write_text(
        "import os\n\ndef helper():\n    return 1\n\nHELPER_DOC = 'doc'\n",
        encoding="utf-8",
    )

    docs = dict(code_symbol_documents(tmp_path, context_lines=2))

    assert docs["a.py::helper"] == (
        "import os\n\ndef helper():\n    return 1\n\nHELPER_DOC = 'doc'\n"
    )


def test_symbol_documents_context_lines_default_zero_keeps_exact_span(tmp_path: Path) -> None:
    pytest.importorskip("tree_sitter")
    (tmp_path / "a.py").write_text(
        "import os\n\ndef helper():\n    return 1\n\nHELPER_DOC = 'doc'\n",
        encoding="utf-8",
    )

    assert dict(code_symbol_documents(tmp_path))["a.py::helper"] == "def helper():\n    return 1\n"
