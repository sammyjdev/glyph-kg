"""Embedding ports: turn text into vectors and search them by similarity."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

Vector = Sequence[float]


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Return one vector per input text."""
        ...


@runtime_checkable
class VectorIndex(Protocol):
    def add(self, key: str, vector: Vector) -> None:
        """Store a vector under ``key``."""
        ...

    def search(self, query: Vector, k: int) -> list[tuple[str, float]]:
        """Return the ``k`` nearest ``(key, cosine_score)`` pairs, best first."""
        ...
