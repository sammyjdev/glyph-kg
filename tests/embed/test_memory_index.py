from glyph.embed.memory_index import InMemoryVectorIndex


def test_search_ranks_by_cosine_similarity() -> None:
    index = InMemoryVectorIndex()
    index.add("fire", [1.0, 0.0])
    index.add("cold", [0.0, 1.0])
    index.add("warm", [0.9, 0.1])
    results = index.search([1.0, 0.0], k=2)
    assert [key for key, _ in results] == ["fire", "warm"]
    assert results[0][1] > results[1][1]


def test_search_respects_k() -> None:
    index = InMemoryVectorIndex()
    for i in range(5):
        index.add(f"v{i}", [float(i), 1.0])
    assert len(index.search([1.0, 1.0], k=3)) == 3


def test_search_empty_index_returns_empty() -> None:
    assert InMemoryVectorIndex().search([1.0, 0.0], k=3) == []


def test_search_handles_zero_vector_without_error() -> None:
    index = InMemoryVectorIndex()
    index.add("z", [0.0, 0.0])
    results = index.search([1.0, 0.0], k=1)
    assert results[0][0] == "z"
