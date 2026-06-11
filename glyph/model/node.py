"""Graph nodes shared by the code and document domains."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NodeType(StrEnum):
    """Node kinds, grouped by domain.

    Code structure is extracted deterministically; document entities are
    inferred by an LLM. Both live in one enum because the graph core is
    agnostic to a node's origin once it is built.
    """

    # code
    FILE = "file"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    # document
    ENTITY = "entity"
    CONCEPT = "concept"
    SECTION = "section"
    # overlay (graph-derived, not extracted): a detected community of member nodes
    COMMUNITY = "community"


class Node(BaseModel):
    """An immutable graph node identified by a stable ``id``."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: NodeType
    label: str
    attrs: dict[str, Any] = Field(default_factory=dict)
