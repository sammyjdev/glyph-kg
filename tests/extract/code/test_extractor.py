"""CodeExtractor turns a source tree into a deterministic code graph."""

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")

from glyph.extract.code import CodeExtractor  # noqa: E402
from glyph.extract.port import Extractor  # noqa: E402
from glyph.model.edge import EdgeType  # noqa: E402
from glyph.model.node import NodeType  # noqa: E402

PY_SOURCE = """\
import os
from collections import OrderedDict


class Base:
    def greet(self):
        helper()


class Worker(Base):
    def run(self):
        self.greet()
        helper()


def helper():
    return 1
"""


def _extract(tmp_path: Path, files: dict[str, str]):  # type: ignore[no-untyped-def]
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    nodes, edges = CodeExtractor().extract(tmp_path)
    node_ids = {n.id for n in nodes}
    edge_set = {(e.src, e.dst, e.type) for e in edges}
    return nodes, node_ids, edge_set


def test_satisfies_extractor_protocol() -> None:
    assert isinstance(CodeExtractor(), Extractor)


def test_extracts_files_classes_and_functions(tmp_path: Path) -> None:
    nodes, ids, _ = _extract(tmp_path, {"a.py": PY_SOURCE})
    by_type = {t: {n.id for n in nodes if n.type == t} for t in NodeType}
    assert "a.py" in by_type[NodeType.FILE]
    assert {"a.py::Base", "a.py::Worker"} <= by_type[NodeType.CLASS]
    assert {"a.py::helper", "a.py::Base.greet", "a.py::Worker.run"} <= by_type[NodeType.FUNCTION]


def test_defines_edges_nest_class_and_methods(tmp_path: Path) -> None:
    _, _, edges = _extract(tmp_path, {"a.py": PY_SOURCE})
    assert ("a.py", "a.py::Base", EdgeType.DEFINES) in edges
    assert ("a.py::Base", "a.py::Base.greet", EdgeType.DEFINES) in edges
    assert ("a.py", "a.py::helper", EdgeType.DEFINES) in edges


def test_resolves_unambiguous_calls(tmp_path: Path) -> None:
    _, _, edges = _extract(tmp_path, {"a.py": PY_SOURCE})
    assert ("a.py::Base.greet", "a.py::helper", EdgeType.CALLS) in edges
    assert ("a.py::Worker.run", "a.py::Base.greet", EdgeType.CALLS) in edges  # self.greet()


def test_resolves_inheritance(tmp_path: Path) -> None:
    _, _, edges = _extract(tmp_path, {"a.py": PY_SOURCE})
    assert ("a.py::Worker", "a.py::Base", EdgeType.INHERITS) in edges


def test_imports_become_module_nodes(tmp_path: Path) -> None:
    nodes, ids, edges = _extract(tmp_path, {"a.py": PY_SOURCE})
    assert "module:os" in ids
    assert ("a.py", "module:os", EdgeType.IMPORTS) in edges
    assert ("a.py", "module:collections", EdgeType.IMPORTS) in edges


def test_ambiguous_call_is_not_resolved(tmp_path: Path) -> None:
    files = {
        "a.py": "def shared():\n    pass\n",
        "b.py": "def shared():\n    pass\n",
        "c.py": "def caller():\n    shared()\n",
    }
    _, _, edges = _extract(tmp_path, files)
    calls = [e for e in edges if e[2] == EdgeType.CALLS]
    assert calls == []  # 'shared' defined twice -> under-approximate, drop the edge


def test_unresolved_call_to_unknown_name_is_dropped(tmp_path: Path) -> None:
    _, _, edges = _extract(tmp_path, {"a.py": "def f():\n    nonexistent()\n"})
    assert not [e for e in edges if e[2] == EdgeType.CALLS]


def test_java_classes_methods_calls_and_inheritance(tmp_path: Path) -> None:
    java = (
        "package demo;\n"
        "import java.util.List;\n"
        "class Animal { void speak() {} }\n"
        "class Dog extends Animal {\n"
        "    void bark() { speak(); }\n"
        "}\n"
    )
    nodes, ids, edges = _extract(tmp_path, {"B.java": java})
    assert {"B.java::Animal", "B.java::Dog"} <= ids
    assert ("B.java::Dog", "B.java::Animal", EdgeType.INHERITS) in edges
    assert ("B.java::Dog.bark", "B.java::Animal.speak", EdgeType.CALLS) in edges
    assert ("B.java", "module:java.util.List", EdgeType.IMPORTS) in edges


def test_single_file_source(tmp_path: Path) -> None:
    path = tmp_path / "solo.py"
    path.write_text("def a():\n    b()\n\ndef b():\n    pass\n", encoding="utf-8")
    nodes, edges = CodeExtractor().extract(path)
    ids = {n.id for n in nodes}
    assert "solo.py::a" in ids
    assert ("solo.py::a", "solo.py::b", EdgeType.CALLS) in {(e.src, e.dst, e.type) for e in edges}


def test_non_source_files_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("# not code\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("def f():\n    pass\n", encoding="utf-8")
    nodes, _ = CodeExtractor().extract(tmp_path)
    assert {n.id for n in nodes if n.type == NodeType.FILE} == {"a.py"}


def test_single_non_source_file_yields_empty_graph(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not code\n", encoding="utf-8")
    nodes, edges = CodeExtractor().extract(path)
    assert nodes == []
    assert edges == []


def test_module_level_call_has_no_caller(tmp_path: Path) -> None:
    # a call outside any function (current_func is None) yields no CALLS edge
    _, _, edges = _extract(tmp_path, {"a.py": "def helper():\n    pass\n\nhelper()\n"})
    assert not [e for e in edges if e[2] == EdgeType.CALLS]


def test_aliased_import_uses_module_name(tmp_path: Path) -> None:
    nodes, ids, edges = _extract(tmp_path, {"a.py": "import os as o\n\ndef f():\n    pass\n"})
    assert "module:os" in ids
    assert ("a.py", "module:os", EdgeType.IMPORTS) in edges


def test_call_on_non_name_target_is_skipped(tmp_path: Path) -> None:
    # fns[0]() — the callee is a subscript, not a name, so no edge is produced
    src = "def f(fns):\n    fns[0]()\n"
    _, _, edges = _extract(tmp_path, {"a.py": src})
    assert not [e for e in edges if e[2] == EdgeType.CALLS]


def test_nested_function_is_scoped_under_its_parent(tmp_path: Path) -> None:
    src = "def outer():\n    def inner():\n        pass\n    inner()\n"
    nodes, ids, edges = _extract(tmp_path, {"a.py": src})
    assert "a.py::outer.inner" in ids
    assert ("a.py::outer", "a.py::outer.inner", EdgeType.DEFINES) in edges


def test_resolution_dedups_edges_and_skips_unresolved(tmp_path: Path) -> None:
    files = {
        "a.py": "import os\nimport os\n",  # duplicate import -> one edge (dedup)
        "b.py": "import os\n",  # same module from another file -> node already exists
        "c.py": "class Orphan(UnknownBase):\n    pass\n",  # base not in corpus -> unresolved
    }
    _, ids, edges = _extract(tmp_path, files)
    import_os = [e for e in edges if e[2] == EdgeType.IMPORTS and e[1] == "module:os"]
    assert sum(1 for e in import_os if e[0] == "a.py") == 1  # duplicate collapsed
    assert ("b.py", "module:os", EdgeType.IMPORTS) in edges
    assert not [e for e in edges if e[2] == EdgeType.INHERITS]  # UnknownBase undefined


def test_java_implements_interface(tmp_path: Path) -> None:
    java = (
        "interface Pet { void name(); }\nclass Dog implements Pet {\n    public void name() {}\n}\n"
    )
    _, ids, edges = _extract(tmp_path, {"B.java": java})
    assert "B.java::Pet" in ids
    assert ("B.java::Dog", "B.java::Pet", EdgeType.INHERITS) in edges
