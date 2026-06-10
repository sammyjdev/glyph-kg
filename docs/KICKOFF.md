# KICKOFF — Fase 0: Fundação

Documento de handoff para execução agent-assisted. Define contexto, contratos e definição de done por sub-task. O agente executa sob direção arquitetural; não inventa escopo fora deste documento.

## Contexto mínimo

GLYPH é uma biblioteca de knowledge graph sobre documentos e código (ver SPEC.md, ARCHITECTURE.md). A Fase 0 entrega o esqueleto: estrutura, modelo de domínio, ports e quality gates. Sem extração, sem retrieval ainda. Objetivo: lib instalável com os contratos no lugar e o CI verde.

Leia antes de começar: `SPEC.md`, `docs/ARCHITECTURE.md`, `docs/decisions/dec-g1-extractor-port-and-backend.md`.

## Stack

- Python 3.11+
- Pydantic v2 (modelo de domínio)
- NetworkX (adapter de store default)
- pytest + pytest-cov (testes)
- ruff + mypy (lint, types)
- tree-sitter entra na Fase 4; não nesta fase

## Estrutura de arquivos alvo

```
glyph-kg/
  pyproject.toml
  glyph/
    __init__.py
    model/
      __init__.py
      node.py          # Node, NodeType
      edge.py          # Edge, EdgeType
    store/
      __init__.py
      port.py          # GraphStore Protocol
      networkx_store.py
    extract/
      __init__.py
      port.py          # Extractor Protocol
  tests/
    architecture/
      test_dependencies.py   # invariantes de import
    model/
    store/
```

## Sub-tasks

### P0.1 — Scaffold
- Criar `pyproject.toml` com build pip-installable, deps (pydantic, networkx) e dev-deps (pytest, pytest-cov, ruff, mypy).
- Estrutura de pacotes acima, com `__init__.py`.
- **Done:** `pip install -e ".[dev]"` funciona; `import glyph` funciona.

### P0.2 — Modelo de domínio
- `Node` (id, type, label, attrs), `NodeType` (enum cobrindo código e documento), `Edge` (src, dst, type, attrs), `EdgeType` (enum com tipos de código e de documento separados).
- Pydantic v2, imutável onde fizer sentido (frozen).
- **Done:** modelos validam; testes de serialização round-trip passam; `model` não importa nada fora de pydantic e stdlib.

### P0.3 — GraphStore port
- `Protocol` com `upsert_nodes`, `upsert_edges`, `neighbors(node, hops)`, `subgraph(seed, hops)`, `shortest_path(src, dst)`.
- Tipos de retorno definidos (`Subgraph`, `Path`).
- **Done:** o Protocol importa só de `model`; mypy valida.

### P0.4 — Adapter NetworkX
- Implementa `GraphStore` sobre um `nx.DiGraph`.
- Persistência: `save(path)` / `load(path)`.
- **Done:** suíte de contrato do store passa (upsert, neighbors com hops, subgraph, shortest_path, persistência round-trip).

### P0.5 — Extractor port
- `Protocol` com `extract(source) -> tuple[Sequence[Node], Sequence[Edge]]`.
- Definir o tipo `Source` (union de path/string conforme necessário).
- **Done:** Protocol importa só de `model`; nenhum adapter concreto ainda (vêm nas Fases 1 e 4).

### P0.6 — Quality gates
- CI (GitHub Actions): lint (ruff), types (mypy), test (pytest), coverage gate.
- `tests/architecture/test_dependencies.py`: verifica os invariantes do ARCHITECTURE.md (model não importa extract/store/retrieval; adapters não se importam entre si).
- **Done:** CI verde no primeiro PR; gate de cobertura ativo; teste de invariante falha se alguém violar a regra de import.

## Definição de done da Fase 0

- `pip install -e .` funciona; `import glyph` expõe model, store port, extract port.
- NetworkX store passa a suíte de contrato.
- CI verde com lint, types, test, coverage e invariantes de arquitetura.
- ADR-G1 commitado (já está).
- Nenhuma lógica de extração ou retrieval ainda; isso é Fase 1+.

## Fora de escopo nesta fase

DocumentExtractor, CodeExtractor, retrieval, baseline vetorial, GNOMON, integração AXON. Não antecipar. Cada um tem sua fase no GLYPH_PLAN.md.

## Convenções

- TDD: teste antes da implementação em cada sub-task.
- Commits pequenos por sub-task, mensagem no imperativo.
- Decisão arquitetural nova durante a execução: parar e abrir ADR, não decidir inline.
