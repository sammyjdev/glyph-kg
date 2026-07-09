# GLYPH

![phases](https://img.shields.io/badge/phases-0--7%20complete-brightgreen) ![coverage](https://img.shields.io/badge/coverage-%E2%89%A590%25-brightgreen) ![python](https://img.shields.io/badge/python-3.11%2B-blue)

> GLYPH builds a knowledge graph from documents and code, then serves graph-aware context for retrieval. Document entities come from LLM extraction, code structure from tree-sitter, both behind one extractor port. Retrieval is benchmarked against a fair vector baseline with bootstrap confidence intervals — across **both** the document and code domains, with the honest result reported either way.

**Part of the [AXON](https://github.com/sammyjdev/axon) stack** — GLYPH is the graph-retrieval layer that AXON (the product front) consumes to decide *what* context to bring; [rtkx](https://github.com/sammyjdev/rtkx) decides *how compact*. GLYPH also stands alone as a benchmarked knowledge-graph retrieval library.

## Why it exists

Vector similarity retrieval ignores structure: who cites whom, what relates to what. For corpora with rich entities and relationships (rules, technical documents) and for code (calls, imports), a graph delivers context that vectors cannot see. GLYPH builds this graph from both domains under a single abstraction, and measures when the graph beats the vector and when it does not justify the cost.

## What it does

- Builds knowledge graphs from **documents** (LLM extraction) and from **code** (tree-sitter, deterministic).
- Serves **graph-aware retrieval**: given a query, anchors entities and expands neighborhood by `hops`.
- Compares against a **fair vector baseline** on the same corpus, measured with confidence intervals.

## What it does NOT do (yet)

- Does not perform complete cross-language type inference in code. Symbol resolution is by name in the import graph and intra-file. Declared limitation (ADR-G5).
- The *retrieval* token budget is estimated per character; **generation** tokens in the benchmark are actual model counts.
- Does not replace AXON parsing; GLYPH is the canonical graph source and AXON delegates to it (dec-116 / ADR-G6).

## Current status

**Phases 0–7 complete.** Model + ports + NetworkX adapter (F0); document extraction by LLM with persisted graph (F1); graph-aware retrieval + fair vector baseline + hybrid under a single contract (F2); benchmark against GNOMON with bootstrap CIs (F3); code extraction by tree-sitter (F4); product boundary `GraphContextSource` consumed by AXON (F5, ADR-G6); packaging & publication (F6); global community axis (F7, ADR-G7).

**Published metrics (validation-first):**
- **Documents** (Monster Manual, n=25): graph leads in faithfulness (0.987) at lowest token cost; context_precision ties within CIs. [METRICS.md](METRICS.md).
- **Code** (AXON graph, n=14, judged by **two independent judges**, Gemini + Qwen): on **faithfulness** (robust metric — same ordering across both judges) the **vector baseline leads and graph ranks last** (vector > hybrid > graph); **context_precision is judge-dependent** (ordering reverses between judges → no reliable winner). The thesis "graph wins on code" did not hold up on the robust metric. [METRICS-code.md](METRICS-code.md) (Gemini) + [METRICS-code-qwen.md](METRICS-code-qwen.md) (Qwen).
- **Global / sense-making** (community axis ADR-G7, n=8, two judges): no arm wins significantly in quality (overlapping CIs), but **community summary delivers equal-or-better quality at ~half the tokens** (5.3k vs 10k+) of vector/graph — *same answer, half the cost* for "how does this organize?" questions. [METRICS-code-global.md](METRICS-code-global.md) + `eval/code-global-qwen.json`.

Methodology in [ADR-G4](docs/decisions/dec-g4-eval-methodology.md). Plan in [docs/GLYPH_PLAN.md](docs/GLYPH_PLAN.md).

## Prerequisites

- Python 3.11+
- `ANTHROPIC_API_KEY` — only for **document extraction** by LLM (`DocumentExtractor`, Claude Haiku 4.5). Code extraction (tree-sitter) and retrieval do not require a key.
- To run the **benchmark**: a key for any OpenAI-compatible endpoint for judge/generation (NVIDIA NIM, Groq, or Gemini) — see [Environment variables](#environment-variables).
- The default graph backend is NetworkX (in-process, no server).

## Local setup

```bash
git clone https://github.com/sammyjdev/glyph-kg.git
cd glyph-kg
pip install -e ".[dev]"
pytest
```

## Architecture

```
              Extractor port
       +------------+------------+
       |                         |
DocumentExtractor          CodeExtractor
(LLM, probabilistic)       (tree-sitter, deterministic)
       |                         |
       +--------> Graph <--------+
                (Node/Edge)
                    |
              GraphStore port
            +-------+-------+
            |               |
      NetworkX          Neo4j
      (default)        (adapter)
            |
     graph-aware retrieval
            |
   GNOMON benchmark vs vector baseline
```

Details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Execution plan in [docs/GLYPH_PLAN.md](docs/GLYPH_PLAN.md).

## Environment variables

GLYPH does not define its own env vars — it reads those from the SDKs/endpoints it uses, and the benchmark selects the provider by flag (`--base-url`/`--api-key-env`), not by a fixed variable.

| Variable | When | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | document extraction by LLM | read by the `anthropic` SDK (`DocumentExtractor`/`AnthropicGenerator`) |
| `NVIDIA_NIM_API_KEY` | benchmark (judge/generation free OSS) | used via `--api-key-env NVIDIA_NIM_API_KEY` |
| `GROQ_API_KEY` | benchmark (OSS judge, default) | default for `OpenAICompatJudge` |
| `GEMINI_API_KEY` | benchmark (independent judge) | via `--api-key-env GEMINI_API_KEY` + `--judge-no-seed` |

Graph backend selection is in code (`NetworkXStore` default; optional Neo4j adapter), not by env var.

## Running the tests

```bash
pytest                      # full suite
pytest --cov=glyph          # with coverage
pytest tests/architecture   # architecture invariants
```

## Usage

```python
# 1. Build a code graph (deterministic, no LLM)
from glyph.extract.code import CodeExtractor
from glyph.store import NetworkXStore

nodes, edges = CodeExtractor().extract("path/to/src")     # Python + Java
store = NetworkXStore()
store.upsert_nodes(nodes)
store.upsert_edges(edges)
store.save("out/code.json")

# 2. Serve graph-aware context — the product boundary (ADR-G6), satisfying the Retriever port
from glyph.integration import GraphContextSource
from glyph.embed.sentence_transformers_embedder import SentenceTransformerEmbedder

source = GraphContextSource.from_graph_file("out/code.json", SentenceTransformerEmbedder())
pack = source.retrieve("how is retry handled?", token_budget=1000)   # -> ContextPack
for segment in pack.segments:
    print(segment.score, segment.source, segment.text)
```

For **document** graphs, replace `CodeExtractor` with `glyph.extract.document.extractor.DocumentExtractor` (requires `ANTHROPIC_API_KEY`). The **global** community axis (ADR-G7) lives in `glyph.retrieval.community`.

## Design decisions

ADRs in [docs/decisions/](docs/decisions/):

- **ADR-G1**: Extractor port + backend selection (NetworkX default, Neo4j adapter).
- **ADR-G2**: document extraction schema (D&D entities/relations).
- **ADR-G3**: fair vector baseline + unified output contract.
- **ADR-G4**: eval methodology (query set, OSS reference-free judge, bootstrap CI, cost).
- **ADR-G5**: symbol resolution in code extractor (by unique name, declared limitation).
- **ADR-G6**: `GraphContextSource` is the product boundary (satisfies the `Retriever` port).
- **ADR-G7**: global community axis (Louvain seeded; isolation by pruned traversal projection).

Results and reproduction: [METRICS.md](METRICS.md) (real run, n=25). Technical article (validation-first) in [docs/article.md](docs/article.md); portfolio claims in [docs/portfolio.md](docs/portfolio.md).

## Relationship to AXON

AXON consumes GLYPH as a dependency: AXON's `GraphContextSource` (ADR-102 in the AXON repo) delegates to `glyph.integration.GraphContextSource` from this lib. Code-graph and document-graph become two uses of the same core. Contract and boundary in [docs/axon-integration.md](docs/axon-integration.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). TDD per sub-task, ADR per architectural decision, invariants checked in CI.

## License

Apache-2.0. See [LICENSE](LICENSE).
