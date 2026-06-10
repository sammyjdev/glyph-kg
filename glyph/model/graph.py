"""Graph-shaped return types used across the store and retrieval ports."""

from pydantic import BaseModel, ConfigDict, Field

from glyph.model.edge import Edge
from glyph.model.node import Node

NodeId = str


class Subgraph(BaseModel):
    """A bundle of nodes and edges, e.g. a neighborhood or a seed expansion."""

    model_config = ConfigDict(frozen=True)

    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


class Path(BaseModel):
    """An ordered walk through the graph, expressed as node ids."""

    model_config = ConfigDict(frozen=True)

    nodes: list[NodeId]
