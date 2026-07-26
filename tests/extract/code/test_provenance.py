"""CP-1 (#50): CLASS/FUNCTION/FILE nodes carry file + line-span provenance.

A retrieved code segment must be addressable (file:line) for the consumer
to jump to; tree-sitter carries start_point/end_point for free.
"""

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")

from glyph.extract.code import CodeExtractor  # noqa: E402
from glyph.model.node import NodeType  # noqa: E402

PY_SOURCE = """\
class Base:
    def greet(self):
        pass

def helper():
    pass
"""

JAVA_SOURCE = """\
class Widget {
    void run() {
    }
}
"""


def _nodes_by_id(tmp_path: Path, files: dict[str, str]) -> dict[str, object]:
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    nodes, _edges = CodeExtractor().extract(tmp_path)
    return {node.id: node for node in nodes}


def test_python_symbols_carry_file_and_line_span(tmp_path: Path) -> None:
    nodes = _nodes_by_id(tmp_path, {"a.py": PY_SOURCE})

    base = nodes["a.py::Base"]
    assert base.attrs["file"] == "a.py"
    assert base.attrs["start_line"] == 1
    assert base.attrs["end_line"] == 3

    greet = nodes["a.py::Base.greet"]
    assert greet.attrs["file"] == "a.py"
    assert greet.attrs["start_line"] == 2
    assert greet.attrs["end_line"] == 3

    helper = nodes["a.py::helper"]
    assert helper.attrs["file"] == "a.py"
    assert helper.attrs["start_line"] == 5
    assert helper.attrs["end_line"] == 6


def test_java_symbols_carry_file_and_line_span(tmp_path: Path) -> None:
    nodes = _nodes_by_id(tmp_path, {"Widget.java": JAVA_SOURCE})

    widget = nodes["Widget.java::Widget"]
    assert widget.attrs["file"] == "Widget.java"
    assert widget.attrs["start_line"] == 1
    assert widget.attrs["end_line"] == 4

    run = nodes["Widget.java::Widget.run"]
    assert run.attrs["file"] == "Widget.java"
    assert run.attrs["start_line"] == 2
    assert run.attrs["end_line"] == 3


def test_file_node_spans_the_whole_file(tmp_path: Path) -> None:
    nodes = _nodes_by_id(tmp_path, {"a.py": PY_SOURCE})

    file_node = nodes["a.py"]
    assert file_node.type is NodeType.FILE
    assert file_node.attrs["file"] == "a.py"
    assert file_node.attrs["start_line"] == 1
    assert file_node.attrs["end_line"] >= 6


def test_module_nodes_have_no_fake_provenance(tmp_path: Path) -> None:
    nodes = _nodes_by_id(tmp_path, {"a.py": "import os\n"})

    module = nodes["module:os"]
    assert module.attrs == {}
