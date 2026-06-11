from glyph.model.contract import ContextPack, Segment, estimate_tokens, pack


def test_estimate_tokens_is_a_ceil_char_quarter() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_pack_keeps_segments_within_budget() -> None:
    segments = [
        Segment(text="a" * 40, source="s1", score=0.9),  # 10 tokens
        Segment(text="b" * 40, source="s2", score=0.8),  # 10 tokens
        Segment(text="c" * 40, source="s3", score=0.7),  # 10 tokens
    ]
    result = pack("vector", segments, token_budget=20)
    assert [s.source for s in result.segments] == ["s1", "s2"]
    assert result.token_estimate == 20
    assert result.mode == "vector"


def test_pack_includes_first_segment_even_if_over_budget() -> None:
    segments = [Segment(text="x" * 400, source="big", score=1.0)]
    result = pack("graph", segments, token_budget=5)
    assert [s.source for s in result.segments] == ["big"]


def test_contextpack_round_trips_through_json() -> None:
    result = pack("hybrid", [Segment(text="hi", source="s", score=0.5)], token_budget=100)
    assert ContextPack.model_validate_json(result.model_dump_json()) == result
