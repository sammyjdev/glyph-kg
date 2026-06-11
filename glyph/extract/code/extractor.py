"""P4.1: a deterministic tree-sitter code extractor (Python + Java).

Walks each file's AST into FILE/CLASS/FUNCTION/MODULE nodes and DEFINES/IMPORTS/
CALLS/INHERITS edges. Symbol resolution is by unqualified name across the corpus:
a CALLS/INHERITS edge is emitted only when the callee/base name is defined exactly
once (under-approximation on ambiguity) — the declared limitation, ADR-G5. The
extraction itself is exact; only cross-symbol resolution is heuristic.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from glyph.extract.code.grammar import Grammar, for_path
from glyph.extract.port import Source
from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType


class CodeExtractor:
    """Extractor port adapter: source tree -> code knowledge graph."""

    def extract(self, source: Source) -> tuple[Sequence[Node], Sequence[Edge]]:
        root = Path(source)
        files = self._files(root)

        nodes: dict[str, Node] = {}
        defines: list[tuple[str, str]] = []
        imports: list[tuple[str, str]] = []
        raw_calls: list[tuple[str, str]] = []
        raw_inherits: list[tuple[str, str]] = []

        for path in files:
            grammar = for_path(path.name)
            if grammar is None:
                continue
            file_id = self._file_id(root, path)
            nodes[file_id] = Node(id=file_id, type=NodeType.FILE, label=path.name)
            tree = self._parse(grammar, path.read_bytes())
            self._walk(
                tree.root_node,
                grammar,
                file_id,
                file_id,
                None,
                nodes,
                defines,
                imports,
                raw_calls,
                raw_inherits,
            )

        edges = self._resolve(nodes, defines, imports, raw_calls, raw_inherits)
        return list(nodes.values()), edges

    # -- file discovery -------------------------------------------------------

    def _files(self, root: Path) -> list[Path]:
        if root.is_file():
            return [root]
        return sorted(p for p in root.rglob("*") if p.is_file() and for_path(p.name) is not None)

    def _file_id(self, root: Path, path: Path) -> str:
        rel = path.relative_to(root) if (root.is_dir() and path != root) else Path(path.name)
        return rel.as_posix()

    def _parse(self, grammar: Grammar, source: bytes) -> Any:
        import tree_sitter

        return tree_sitter.Parser(grammar.language()).parse(source)

    # -- AST walk -------------------------------------------------------------

    def _walk(
        self,
        node: Any,
        grammar: Grammar,
        file_id: str,
        scope_id: str,
        current_func: str | None,
        nodes: dict[str, Node],
        defines: list[tuple[str, str]],
        imports: list[tuple[str, str]],
        raw_calls: list[tuple[str, str]],
        raw_inherits: list[tuple[str, str]],
    ) -> None:
        for child in node.children:
            if child.type in grammar.class_types:
                name = grammar.name_of(child)
                if name is None:  # pragma: no cover - a class definition is always named
                    continue
                class_id = f"{file_id}::{name}"
                nodes[class_id] = Node(id=class_id, type=NodeType.CLASS, label=name)
                defines.append((scope_id, class_id))
                for base in grammar.superclass_names(child):
                    raw_inherits.append((class_id, base))
                self._walk(
                    child,
                    grammar,
                    file_id,
                    class_id,
                    None,
                    nodes,
                    defines,
                    imports,
                    raw_calls,
                    raw_inherits,
                )
            elif child.type in grammar.func_types:
                name = grammar.name_of(child)
                if name is None:  # pragma: no cover - a function definition is always named
                    continue
                sep = "::" if scope_id == file_id else "."
                func_id = f"{scope_id}{sep}{name}"
                nodes[func_id] = Node(id=func_id, type=NodeType.FUNCTION, label=name)
                defines.append((scope_id, func_id))
                self._walk(
                    child,
                    grammar,
                    file_id,
                    func_id,
                    func_id,
                    nodes,
                    defines,
                    imports,
                    raw_calls,
                    raw_inherits,
                )
            elif child.type in grammar.import_types:
                module = grammar.import_name(child)
                if module is None:  # pragma: no cover - import_name only None on malformed input
                    continue
                imports.append((file_id, module))
            else:
                if child.type in grammar.call_types and current_func is not None:
                    callee = grammar.callee_name(child)
                    if callee is not None:
                        raw_calls.append((current_func, callee))
                self._walk(
                    child,
                    grammar,
                    file_id,
                    scope_id,
                    current_func,
                    nodes,
                    defines,
                    imports,
                    raw_calls,
                    raw_inherits,
                )

    # -- resolution -----------------------------------------------------------

    def _resolve(
        self,
        nodes: dict[str, Node],
        defines: list[tuple[str, str]],
        imports: list[tuple[str, str]],
        raw_calls: list[tuple[str, str]],
        raw_inherits: list[tuple[str, str]],
    ) -> list[Edge]:
        by_name: dict[str, list[str]] = {}
        for node in nodes.values():
            if node.type in (NodeType.FUNCTION, NodeType.CLASS):
                by_name.setdefault(node.label, []).append(node.id)

        edges: list[Edge] = []
        seen: set[tuple[str, str, EdgeType]] = set()

        def add(src: str, dst: str, edge_type: EdgeType) -> None:
            key = (src, dst, edge_type)
            if key not in seen:
                seen.add(key)
                edges.append(Edge(src=src, dst=dst, type=edge_type))

        for src, dst in defines:
            add(src, dst, EdgeType.DEFINES)

        for file_id, module in imports:
            module_id = f"module:{module}"
            if module_id not in nodes:
                nodes[module_id] = Node(id=module_id, type=NodeType.MODULE, label=module)
            add(file_id, module_id, EdgeType.IMPORTS)

        for caller, callee in raw_calls:
            targets = by_name.get(callee, [])
            if len(targets) == 1:  # resolve only unambiguous names (ADR-G5)
                add(caller, targets[0], EdgeType.CALLS)

        for class_id, base in raw_inherits:
            targets = by_name.get(base, [])
            if len(targets) == 1:
                add(class_id, targets[0], EdgeType.INHERITS)

        return edges
