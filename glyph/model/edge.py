"""Graph edges shared by the code and document domains."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EdgeType(str, Enum):
    """Edge kinds, grouped by domain.

    Code edges (``CALLS``, ``IMPORTS``, ...) are facts read from an AST.
    Document edges (``RELATES_TO``, ``RESISTS``, ...) are LLM-inferred and
    therefore carry extraction error — measured, not assumed, downstream.
    """

    # code
    DEFINES = "defines"
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    REFERENCES = "references"
    # document
    RELATES_TO = "relates_to"
    MENTIONS = "mentions"
    REQUIRES = "requires"
    RESISTS = "resists"
    IMMUNE_TO = "immune_to"
    VULNERABLE_TO = "vulnerable_to"
    INHABITS = "inhabits"
    SUMMONS = "summons"


class Edge(BaseModel):
    """An immutable directed edge between two node ids."""

    model_config = ConfigDict(frozen=True)

    src: str
    dst: str
    type: EdgeType
    attrs: dict[str, Any] = Field(default_factory=dict)
