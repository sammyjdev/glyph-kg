"""The Retriever port: every arm turns a query into a comparable ContextPack."""

from typing import Protocol, runtime_checkable

from glyph.model.contract import ContextPack


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, token_budget: int) -> ContextPack:
        """Return retrieved context for ``query`` within ``token_budget``."""
        ...
