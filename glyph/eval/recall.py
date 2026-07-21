"""Glyph-native retrieval instrumentation (ADR-G8): recall@budget.

Deterministic, judge-free measurement of whether a query's KG-derived relevant
sources actually made it into a ContextPack within its token budget. Out of
gnomon's scope by design (ADR-G8) - it needs target internals (retrieval
budgets, segment provenance) the judge contract cannot see.
"""

from collections.abc import Collection


def recall_at_budget(relevant: Collection[str], retrieved_sources: Collection[str]) -> float:
    """Fraction of ``relevant`` (deduplicated) whose id appears in ``retrieved_sources``."""
    relevant_set = set(relevant)
    assert relevant_set, "recall_at_budget requires at least one relevant source"  # noqa: S101
    return len(relevant_set & set(retrieved_sources)) / len(relevant_set)
