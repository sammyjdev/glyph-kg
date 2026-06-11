"""Per-language tree-sitter configuration for the code extractor.

Each grammar names the AST node types that mark definitions, calls and imports,
and knows how to pull the relevant name out of each. The walker in ``extractor``
is language-agnostic and drives itself from these. tree-sitter is imported lazily
so importing this module never requires the ``code`` extra.
"""

from __future__ import annotations

from typing import Any


def _text(node: Any) -> str:
    return str(node.text.decode("utf-8"))


def _descendants_of_type(node: Any, node_type: str) -> list[Any]:
    """All descendants of ``node`` with the given AST type (depth-first)."""
    found: list[Any] = []
    for child in node.children:
        if child.type == node_type:
            found.append(child)
        found.extend(_descendants_of_type(child, node_type))
    return found


class Grammar:
    """Base config: node-type sets + name extractors a language overrides."""

    name: str
    extensions: tuple[str, ...]
    func_types: frozenset[str]
    class_types: frozenset[str]
    call_types: frozenset[str]
    import_types: frozenset[str]

    def language(self) -> Any:  # pragma: no cover - thin lazy binding
        raise NotImplementedError

    def name_of(self, node: Any) -> str | None:
        field = node.child_by_field_name("name")
        return _text(field) if field is not None else None

    def callee_name(self, call_node: Any) -> str | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def import_name(self, import_node: Any) -> str | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def superclass_names(self, class_node: Any) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError


class PythonGrammar(Grammar):
    name = "python"
    extensions = (".py",)
    func_types = frozenset({"function_definition"})
    class_types = frozenset({"class_definition"})
    call_types = frozenset({"call"})
    import_types = frozenset({"import_statement", "import_from_statement"})

    def language(self) -> Any:  # pragma: no cover - thin lazy binding
        import tree_sitter
        import tree_sitter_python

        return tree_sitter.Language(tree_sitter_python.language())

    def callee_name(self, call_node: Any) -> str | None:
        target = call_node.child_by_field_name("function")
        if target is None:  # pragma: no cover - a call always has a function child
            return None
        if target.type == "identifier":
            return _text(target)
        if target.type == "attribute":  # obj.method(...)
            attribute = target.child_by_field_name("attribute")
            return _text(attribute) if attribute is not None else None
        return None

    def import_name(self, import_node: Any) -> str | None:
        if import_node.type == "import_from_statement":
            module = import_node.child_by_field_name("module_name")
            return _text(module) if module is not None else None
        for child in import_node.children:
            if child.type == "dotted_name":
                return _text(child)
            if child.type == "aliased_import":  # import x as y
                inner = child.child_by_field_name("name")
                return _text(inner) if inner is not None else None
        return None  # pragma: no cover - an import always carries a name

    def superclass_names(self, class_node: Any) -> list[str]:
        supers = class_node.child_by_field_name("superclasses")
        if supers is None:
            return []
        return [_text(c) for c in supers.children if c.type == "identifier"]


class JavaGrammar(Grammar):
    name = "java"
    extensions = (".java",)
    func_types = frozenset({"method_declaration", "constructor_declaration"})
    class_types = frozenset({"class_declaration", "interface_declaration"})
    call_types = frozenset({"method_invocation"})
    import_types = frozenset({"import_declaration"})

    def language(self) -> Any:  # pragma: no cover - thin lazy binding
        import tree_sitter
        import tree_sitter_java

        return tree_sitter.Language(tree_sitter_java.language())

    def callee_name(self, call_node: Any) -> str | None:
        field = call_node.child_by_field_name("name")
        return _text(field) if field is not None else None

    def import_name(self, import_node: Any) -> str | None:
        for child in import_node.children:
            if child.type in {"scoped_identifier", "identifier"}:
                return _text(child)
        return None  # pragma: no cover - an import always carries a name

    def superclass_names(self, class_node: Any) -> list[str]:
        # extends names sit directly under `superclass`; implements names are nested
        # in a type_list under `interfaces` — collect type_identifiers from both.
        names: list[str] = []
        for field in ("superclass", "interfaces"):
            node = class_node.child_by_field_name(field)
            if node is not None:
                names += [_text(c) for c in _descendants_of_type(node, "type_identifier")]
        return names


_GRAMMARS: tuple[Grammar, ...] = (PythonGrammar(), JavaGrammar())


def for_path(path: str) -> Grammar | None:
    """The grammar whose extension matches ``path``, or None if unsupported."""
    for grammar in _GRAMMARS:
        if path.endswith(grammar.extensions):
            return grammar
    return None
