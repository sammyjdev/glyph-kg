"""Shared fixtures for store contract tests.

The ``store`` and ``hub_store`` fixtures are parametrized over NetworkXStore
and Neo4jStore.  The Neo4j variants are marked ``neo4j`` and deselected from
the default test run (they require a running Docker daemon).
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from glyph.model.edge import Edge, EdgeType
from glyph.model.node import Node, NodeType
from glyph.store.networkx_store import NetworkXStore
from glyph.store.port import GraphStore

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _populate_linear(s: GraphStore) -> GraphStore:
    """a -> b -> c chain (directed RELATES_TO); anchor is a."""
    s.upsert_nodes(
        [
            Node(id="a", type=NodeType.ENTITY, label="a"),
            Node(id="b", type=NodeType.ENTITY, label="b"),
            Node(id="c", type=NodeType.ENTITY, label="c"),
        ]
    )
    s.upsert_edges(
        [
            Edge(src="a", dst="b", type=EdgeType.RELATES_TO),
            Edge(src="b", dst="c", type=EdgeType.RELATES_TO),
        ]
    )
    return s


def _populate_hub(s: GraphStore) -> GraphStore:
    """a->b->c->d plus a COMMUNITY super-hub linked to a and d via CONTAINS."""
    s.upsert_nodes(
        [
            Node(id="a", type=NodeType.ENTITY, label="a"),
            Node(id="b", type=NodeType.ENTITY, label="b"),
            Node(id="c", type=NodeType.ENTITY, label="c"),
            Node(id="d", type=NodeType.ENTITY, label="d"),
            Node(id="comm", type=NodeType.COMMUNITY, label="faction"),
        ]
    )
    s.upsert_edges(
        [
            Edge(src="a", dst="b", type=EdgeType.RELATES_TO),
            Edge(src="b", dst="c", type=EdgeType.RELATES_TO),
            Edge(src="c", dst="d", type=EdgeType.RELATES_TO),
            Edge(src="comm", dst="a", type=EdgeType.CONTAINS),
            Edge(src="comm", dst="d", type=EdgeType.CONTAINS),
        ]
    )
    return s


# ---------------------------------------------------------------------------
# Session-scoped Neo4j container
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def neo4j_container() -> Generator[Any, None, None]:
    """Start a Neo4j Testcontainer once per session (only when neo4j tests run)."""
    try:
        from testcontainers.neo4j import Neo4jContainer  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("testcontainers[neo4j] not installed")
        return

    with Neo4jContainer("neo4j:5") as container:
        yield container


# ---------------------------------------------------------------------------
# Shared backend factory (lazily acquires the container)
# ---------------------------------------------------------------------------


def _make_neo4j_store(request: pytest.FixtureRequest) -> tuple[Any, Any]:
    """Return (store, driver) for a freshly-wiped Neo4j DB."""
    container = request.getfixturevalue("neo4j_container")

    from neo4j import GraphDatabase  # noqa: PLC0415

    from glyph.store.neo4j_store import Neo4jStore  # noqa: PLC0415

    bolt_url = container.get_connection_url()
    username = container.username
    password = container.password

    driver = GraphDatabase.driver(bolt_url, auth=(username, password))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()

    return Neo4jStore(bolt_url, auth=(username, password)), None


# ---------------------------------------------------------------------------
# Parametrized ``store`` fixture (linear a->b->c)
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("networkx", id="networkx"),
        pytest.param("neo4j", id="neo4j", marks=pytest.mark.neo4j),
    ]
)
def store(request: pytest.FixtureRequest) -> Generator[GraphStore, None, None]:
    """Yield a fresh, pre-populated GraphStore (networkx or neo4j)."""
    if request.param == "networkx":
        yield _populate_linear(NetworkXStore())
        return

    s, _ = _make_neo4j_store(request)
    yield _populate_linear(s)
    s.close()


# ---------------------------------------------------------------------------
# Parametrized ``hub_store`` fixture (a->b->c->d + COMMUNITY super-hub)
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("networkx", id="networkx"),
        pytest.param("neo4j", id="neo4j", marks=pytest.mark.neo4j),
    ]
)
def hub_store(request: pytest.FixtureRequest) -> Generator[GraphStore, None, None]:
    """Yield a fresh hub-topology GraphStore (networkx or neo4j)."""
    if request.param == "networkx":
        yield _populate_hub(NetworkXStore())
        return

    s, _ = _make_neo4j_store(request)
    yield _populate_hub(s)
    s.close()
