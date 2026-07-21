import pytest

from glyph.eval.recall import recall_at_budget


def test_recall_partial() -> None:
    assert recall_at_budget({"a", "b", "c", "d"}, {"a", "b", "x"}) == 0.5


def test_recall_full() -> None:
    assert recall_at_budget({"a", "b"}, {"a", "b", "c"}) == 1.0


def test_recall_zero() -> None:
    assert recall_at_budget({"a", "b"}, {"x", "y"}) == 0.0


def test_recall_dedups() -> None:
    assert recall_at_budget(["a", "a", "b"], ["a"]) == 0.5


def test_recall_empty_relevant_asserts() -> None:
    with pytest.raises(AssertionError):
        recall_at_budget([], {"a"})
