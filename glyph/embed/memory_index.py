"""In-memory cosine vector index (numpy). Zero servers; fits the target corpora."""

import numpy as np

from glyph.embed.port import Vector


class InMemoryVectorIndex:
    """A list-backed cosine index over float vectors."""

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._vectors: list[np.ndarray[tuple[int], np.dtype[np.float32]]] = []

    def add(self, key: str, vector: Vector) -> None:
        self._keys.append(key)
        self._vectors.append(np.asarray(vector, dtype=np.float32))

    def search(self, query: Vector, k: int) -> list[tuple[str, float]]:
        if not self._vectors:
            return []
        matrix = np.vstack(self._vectors)
        q = np.asarray(query, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        norms[norms == 0.0] = 1e-12
        scores = (matrix @ q) / norms
        order = np.argsort(-scores)[:k]
        return [(self._keys[i], float(scores[i])) for i in order]
