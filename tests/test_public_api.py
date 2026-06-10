"""Phase 0 done-criterion: `import glyph` exposes model, store port, extract port."""

import glyph


def test_import_glyph_exposes_subpackages() -> None:
    assert glyph.model is not None
    assert glyph.store is not None
    assert glyph.extract is not None


def test_core_symbols_are_importable_from_glyph() -> None:
    from glyph import (
        Edge,
        EdgeType,
        Extractor,
        GraphStore,
        NetworkXStore,
        Node,
        NodeType,
    )

    assert {Edge, EdgeType, Extractor, GraphStore, NetworkXStore, Node, NodeType}
