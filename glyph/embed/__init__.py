"""Embedding infrastructure: embedder + vector index ports and adapters."""

from glyph.embed.memory_index import InMemoryVectorIndex
from glyph.embed.port import Embedder, VectorIndex
from glyph.embed.sentence_transformers_embedder import SentenceTransformerEmbedder

__all__ = ["Embedder", "InMemoryVectorIndex", "SentenceTransformerEmbedder", "VectorIndex"]
