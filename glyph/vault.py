"""Vault builder: walk an Obsidian directory and extract a knowledge graph.

This module sits at the top of the ``glyph`` package so it can freely import
from both ``glyph.extract`` (document loading) and ``glyph.store``
(persistence) without violating the hexagonal architecture invariant that says
extract adapters must not import store adapters.

Import path: ``from glyph.vault import build_vault``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from glyph.extract.document.extractor import DocumentExtractor
from glyph.extract.document.llm import ExtractorConfig, LLMExtractor, make_extractor
from glyph.store.networkx_store import NetworkXStore


def build_vault(
    vault_path: Path,
    output_path: Path,
    *,
    provider: str = "litellm",
    model: str = "ollama/llama3",
    base_url: str | None = "http://localhost:11434",
    api_key: str | None = None,
    llm: LLMExtractor | None = None,
) -> tuple[int, int]:
    """Walk *vault_path* for ``.md`` files, extract entities/relations, persist graph.

    Parameters
    ----------
    vault_path:
        Root directory of the Obsidian vault (searched recursively for ``*.md``).
    output_path:
        Destination for the persisted graph JSON (parent dirs created as needed).
    provider:
        ``"litellm"`` (default, supports Ollama) or ``"anthropic"``.
    model:
        LiteLLM model string, e.g. ``"ollama/llama3"``, ``"gemini/gemini-1.5-flash"``.
    base_url:
        Base URL for the LiteLLM backend (default: ``http://localhost:11434`` for Ollama).
    api_key:
        API key; omit for local Ollama.
    llm:
        If provided, used directly (overrides *provider*/*model* — useful for tests).

    Returns
    -------
    (node_count, edge_count)
        The totals in the persisted graph after deduplication.
    """
    md_files = sorted(vault_path.rglob("*.md"))
    if not md_files:
        print(f"No .md files found under {vault_path}", file=sys.stderr)
        return 0, 0

    if llm is None:
        cfg = ExtractorConfig(
            provider=provider,
            model=model,
            base_url=base_url if provider == "litellm" else None,
            api_key=api_key,
            domain="notes",
        )
        llm = make_extractor(cfg)

    extractor = DocumentExtractor(llm=llm, domain="notes")
    store = NetworkXStore()
    total_usages = 0

    started = time.monotonic()
    for i, md_file in enumerate(md_files, 1):
        print(f"[{i}/{len(md_files)}] {md_file.name}", end=" … ", flush=True)
        try:
            nodes, edges, usages = extractor.extract_with_usage(md_file)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP ({exc})")
            continue
        store.upsert_nodes(list(nodes))
        store.upsert_edges(list(edges))
        total_usages += len(usages)
        print(f"{len(nodes)} nodes, {len(edges)} edges")

    elapsed = time.monotonic() - started
    node_count: int = store._g.number_of_nodes()
    edge_count: int = store._g.number_of_edges()

    print(f"\nvault:    {vault_path}")
    print(f"files:    {len(md_files)}")
    print(f"nodes:    {node_count}")
    print(f"edges:    {edge_count}")
    print(f"chunks:   {total_usages}")
    print(f"latency:  {elapsed:.1f}s")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    store.save(output_path)
    print(f"graph  →  {output_path}")

    return node_count, edge_count
