"""GraphStore port and adapters (NetworkX default, Neo4j adapter)."""

from glyph.store.networkx_store import NetworkXStore
from glyph.store.port import GraphStore

__all__ = ["GraphStore", "NetworkXStore", "Neo4jStore"]

try:
    from glyph.store.neo4j_store import Neo4jStore  # noqa: F401
except ImportError:
    pass  # neo4j extra not installed; Neo4jStore unavailable
