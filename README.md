# GLYPH

![phases](https://img.shields.io/badge/phases-0--7%20complete-brightgreen) ![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen) ![python](https://img.shields.io/badge/python-3.11%2B-blue)

> GLYPH builds a knowledge graph from documents and code, then serves graph-aware context for retrieval. Document entities come from LLM extraction, code structure from tree-sitter, both behind one extractor port. Retrieval is benchmarked against a fair vector baseline with bootstrap confidence intervals — across **both** the document and code domains, with the honest result reported either way.

## Por que existe

Retrieval por similaridade vetorial ignora a estrutura: quem cita quem, o que se relaciona com o quê. Para corpora com entidades e relações ricas (regras, documentos técnicos) e para código (calls, imports), um grafo entrega contexto que o vetor não vê. GLYPH constrói esse grafo dos dois domínios sob uma abstração só, e mede quando o grafo ganha do vetor e quando não vale o custo.

## O que faz

- Constrói knowledge graph de **documentos** (extração por LLM) e de **código** (tree-sitter, determinístico).
- Serve **retrieval graph-aware**: dado uma query, ancora entidades e expande vizinhança por `hops`.
- Compara contra um **baseline vetorial justo** sobre o mesmo corpus, medido com intervalos de confiança.

## O que NÃO faz (ainda)

- Não faz type inference cross-language completa no código. A resolução de símbolo é por nome no import graph e intra-file. Limitação declarada (ADR-G5).
- O orçamento de tokens do *retrieval* é estimado por char; os tokens de **geração** no benchmark são contagem real do modelo.
- Não substitui o parsing do AXON; o GLYPH é a fonte canônica de grafo e o AXON delega (dec-116 / ADR-G6).

## Estado atual

**Fases 0–7 completas.** Modelo + ports + adapter NetworkX (F0); extração documental por LLM com grafo persistido (F1); retrieval graph-aware + baseline vetorial justo + híbrido sob um contrato único (F2); benchmark contra o GNOMON com CIs de bootstrap (F3); extração de código por tree-sitter (F4); fronteira de produto `GraphContextSource` consumida pelo AXON (F5, ADR-G6); publicação (F6); eixo global por comunidades (F7, ADR-G7).

**Números publicados (validation-first):**
- **Documentos** (Monster Manual, n=25): o graph lidera faithfulness (0.987) ao menor custo de tokens; context_precision empata dentro dos CIs. [METRICS.md](METRICS.md).
- **Código** (grafo do AXON, n=14, judge independente Gemini): o **baseline vetorial supera o graph** nos dois metrics (faith 0.995 vs 0.839; cp 0.513 vs 0.180) — a tese "graph ganha em código" não se sustentou aqui, e isso é reportado. [METRICS-code.md](METRICS-code.md).

Metodologia em [ADR-G4](docs/decisions/dec-g4-eval-methodology.md). Plano em [docs/GLYPH_PLAN.md](docs/GLYPH_PLAN.md).

## Pré-requisitos

- Python 3.11+
- `ANTHROPIC_API_KEY` — só para a **extração documental** por LLM (`DocumentExtractor`, Claude Haiku 4.5). A extração de **código** (tree-sitter) e o retrieval não precisam de chave.
- Para rodar o **benchmark**: uma chave de qualquer endpoint OpenAI-compatible para o judge/geração (NVIDIA NIM, Groq ou Gemini) — ver [Variáveis de ambiente](#variáveis-de-ambiente).
- O backend de grafo default é NetworkX (in-process, sem servidor).

## Setup local

```bash
git clone https://github.com/sammyjdev/glyph-kg.git
cd glyph-kg
pip install -e ".[dev]"
pytest
```

## Arquitetura

```
              Extractor port
       +------------+------------+
       |                         |
DocumentExtractor          CodeExtractor
(LLM, probabilístico)      (tree-sitter, determinístico)
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

Detalhe em [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Plano de execução em [docs/GLYPH_PLAN.md](docs/GLYPH_PLAN.md).

## Variáveis de ambiente

O GLYPH não define env vars próprias — lê as dos SDKs/endpoints que usa, e o benchmark escolhe o provider por flag (`--base-url`/`--api-key-env`), não por variável fixa.

| Variável | Quando | Descrição |
|---|---|---|
| `ANTHROPIC_API_KEY` | extração documental por LLM | lida pelo `anthropic` SDK (`DocumentExtractor`/`AnthropicGenerator`) |
| `NVIDIA_NIM_API_KEY` | benchmark (judge/geração OSS grátis) | usada via `--api-key-env NVIDIA_NIM_API_KEY` |
| `GROQ_API_KEY` | benchmark (judge OSS, default) | default do `OpenAICompatJudge` |
| `GEMINI_API_KEY` | benchmark (judge independente) | via `--api-key-env GEMINI_API_KEY` + `--judge-no-seed` |

A escolha de backend de grafo é em código (`NetworkXStore` default; adapter Neo4j opcional), não por env var.

## Como rodar os testes

```bash
pytest                      # suíte completa
pytest --cov=glyph          # com cobertura
pytest tests/architecture   # invariantes de arquitetura
```

## Uso

```python
# 1. Construir um grafo de código (determinístico, sem LLM)
from glyph.extract.code import CodeExtractor
from glyph.store import NetworkXStore

nodes, edges = CodeExtractor().extract("path/to/src")     # Python + Java
store = NetworkXStore()
store.upsert_nodes(nodes)
store.upsert_edges(edges)
store.save("out/code.json")

# 2. Servir contexto graph-aware — a fronteira de produto (ADR-G6), que satisfaz o port Retriever
from glyph.integration import GraphContextSource
from glyph.embed.sentence_transformers_embedder import SentenceTransformerEmbedder

source = GraphContextSource.from_graph_file("out/code.json", SentenceTransformerEmbedder())
pack = source.retrieve("how is retry handled?", token_budget=1000)   # -> ContextPack
for segment in pack.segments:
    print(segment.score, segment.source, segment.text)
```

Para o grafo **documental**, troque `CodeExtractor` por `glyph.extract.document.extractor.DocumentExtractor` (requer `ANTHROPIC_API_KEY`). O eixo **global** por comunidades (ADR-G7) vive em `glyph.retrieval.community`.

## Decisões de design

ADRs em [docs/decisions/](docs/decisions/):

- **ADR-G1**: Extractor port + escolha de backend (NetworkX default, Neo4j adapter).
- **ADR-G2**: schema de extração documental (entidades/relações DeD).
- **ADR-G3**: baseline vetorial justo + contrato de saída unificado.
- **ADR-G4**: metodologia de eval (query set, judge OSS reference-free, bootstrap CI, custo).
- **ADR-G5**: resolução de símbolo no code extractor (por nome único, limitação declarada).
- **ADR-G6**: `GraphContextSource` é a fronteira de produto (satisfaz o port `Retriever`).
- **ADR-G7**: eixo global por comunidades (Louvain seeded; isolamento por projeção de traversal podada).

Resultados e reprodução: [METRICS.md](METRICS.md) (corrida real, n=25). Artigo técnico (validation-first) em [docs/article.md](docs/article.md); claims de portfólio em [docs/portfolio.md](docs/portfolio.md).

## Relação com AXON

O AXON consome o GLYPH como dependência: o `GraphContextSource` do AXON (ADR-102 no repo do AXON) delega para `glyph.integration.GraphContextSource` desta lib. Code-graph e document-graph viram dois usos do mesmo núcleo. Contrato e fronteira em [docs/axon-integration.md](docs/axon-integration.md).

## Contribuindo

Ver [CONTRIBUTING.md](CONTRIBUTING.md). TDD por sub-task, ADR por decisão arquitetural, invariantes verificados no CI.

## Licença

MIT. Ver [LICENSE](LICENSE).
