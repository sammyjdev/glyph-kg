# Design — Fase 2: Retrieval graph-aware + baseline vetorial (P2.1–P2.4)

**Data:** 2026-06-10
**Status:** Aprovado (brainstorming)
**Escopo desta sessão:** P2.1–P2.4 (os três braços + contrato). A **medição** é Fase 3
(depende de destravar o GNOMON na Fase 2.5) e fica fora desta sessão.

## Objetivo

Construir a maquinaria de retrieval do GLYPH: três braços — graph-aware, baseline vetorial
justo e híbrido — produzindo um contrato de saída único (`ContextPack`) comparável token-a-token,
sobre o mesmo corpus (o grafo do Monster Manual e os chunks que o geraram). Sem benchmark nesta
fase; só os modos funcionando. Entregável: `retrieve(query, mode=graph|vector|hybrid) -> ContextPack`.

## Decisões fechadas (brainstorming)

1. **Embeddings:** sentence-transformers local multilingual + índice vetorial in-memory
   (numpy/cosine). Zero servidor, zero custo de API, lida com PT-BR. Extra opcional
   `glyph-kg[retrieval]`. Modelo default `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
   (configurável).
2. **Corpus justo:** o baseline vetorial indexa o texto dos **mesmos ~425 chunks de criatura**
   (saída do `chunk.by_creature` + `is_creature`). O braço-grafo é a extração estruturada desses
   mesmos chunks. Mesma fonte, representações diferentes — experimento controlado.
3. **Ancoragem do grafo:** embedding leve — embeda a query e os labels dos nós, ancora nos top-k
   labels mais próximos. Cobre query nomeando entidade ("o que o Deva resiste?") e query conceitual
   ("que criaturas resistem a fogo?") porque a expansão de vizinhança é não-direcionada.
4. **Budget de token:** os três braços cortam a saída no mesmo budget, medido por **estimativa
   por char** (limitação declarada; tokenizer real é assunto da Fase 3, alinhado à honestidade do AXON).
5. **Contrato:** `Segment`/`ContextPack` em `glyph/model/`, idêntico nos três braços.

## Arquitetura (hexagonal, preserva os invariantes do Phase 0)

```
glyph/
  embed/          # infra de embedding — não importa model/store/retrieval/baseline
    port.py         # Embedder Protocol; VectorIndex Protocol
    sentence_transformers_embedder.py   # adapter local (extra opcional)
    memory_index.py # InMemoryVectorIndex (numpy cosine)
  model/
    contract.py     # Segment, ContextPack (+ estimate_tokens helper)
  retrieval/        # depende de store(port) + embed + model; NÃO importa baseline
    port.py         # Retriever Protocol: retrieve(query, ...) -> ContextPack
    graph.py        # GraphRetriever (anchor -> subgraph(hops) -> ContextPack)
    hybrid.py       # HybridRetriever(graph, vector) — fusão via injeção, sob o Retriever Protocol
  baseline/         # depende de embed + model; NÃO importa retrieval
    vector.py       # VectorBaseline (indexa chunks -> top-k cosine -> ContextPack)
```

Regra de dependência: `embed` não conhece ninguém de dentro; `retrieval` e `baseline` dependem de
`embed`, `model` e (retrieval) do `GraphStore` port. **`retrieval` e `baseline` não se importam**:
o `HybridRetriever` recebe dois objetos que satisfazem o `Retriever` Protocol (injeção pelo
composition root), então funde sem acoplar as camadas. O teste de invariantes de arquitetura é
estendido para cobrir `embed`/`retrieval`/`baseline`.

## Contrato de saída

```python
class Segment(BaseModel, frozen=True):
    text: str
    source: str          # node id (graph) ou chunk label (vector)
    score: float

class ContextPack(BaseModel, frozen=True):
    mode: Literal["graph", "vector", "hybrid"]
    segments: list[Segment]
    token_estimate: int

def estimate_tokens(text: str) -> int   # estimativa por char (declarada)
```

O `ContextPack` é montado adicionando segmentos (ordenados por score) até o budget de token; o
`token_estimate` reflete o total incluído. Mesmo budget nos três braços = comparação justa.

## Fluxo de dados por braço

- **GraphRetriever(store, embedder, node_labels_index):** embeda a query → top-k âncoras (cosine vs
  labels) → `store.subgraph(anchor_ids, hops)` → para cada nó do subgrafo monta um `Segment`
  (label + atributos + suas arestas, ex.: "Deva — resists radiante; immune_to enfeitiçado") →
  ordena por proximidade da âncora → corta ao budget.
- **VectorBaseline(embedder):** `index(chunks)` embeda cada texto de chunk no `InMemoryVectorIndex`
  → `retrieve(query, budget)` embeda a query, busca top-k, monta `Segment` por chunk → corta ao budget.
- **HybridRetriever(graph, vector):** roda os dois, funde por **reciprocal rank fusion** sobre
  `Segment.source`, deduplica, corta ao budget compartilhado.

## ADR-G3 (a registrar antes da implementação)

Baseline justo: mesmos chunks, mesmo embedder, mesmo budget, implementação real (chunk + embedding
+ vector store + top-k), não espantalho. Documenta a igualdade de budget entre os braços e a
limitação da contagem de token por char nesta fase.

## Estratégia de teste (TDD)

- Toda a lógica — anchoring, montagem do `ContextPack`, corte por budget, fusão RRF, busca cosine
  do `InMemoryVectorIndex` — testada com um **fake embedder determinístico** (vetores fixos por
  texto). Sem download de modelo, sem rede.
- `InMemoryVectorIndex`: testes de cosine/top-k com vetores conhecidos.
- O adapter `SentenceTransformerEmbedder` real ganha um smoke test **opt-in** (`@pytest.mark.slow`,
  fora do CI default — baixa o modelo) que embeda duas frases PT-BR e checa que a mais similar
  rankeia primeiro. Espelha o padrão do fake-LLM/`@pytest.mark.live` da Fase 1.
- Validação manual (sem CI): um script de wiring carrega `out/monster-manual.json` num
  `NetworkXStore` + os chunks do MM, e roda algumas queries nos três modos para inspeção.

## Critérios de done da Fase 2

- ADR-G3 commitado antes da implementação.
- `embed` (port + ST adapter + InMemoryVectorIndex), `Segment`/`ContextPack`, `GraphRetriever`,
  `VectorBaseline`, `HybridRetriever` implementados com TDD; suíte verde; invariantes de arquitetura
  estendidos e verdes.
- Os três modos rodam sobre o grafo/chunks reais do MM via script de wiring (inspeção manual).
- CI verde (lint, types, test, coverage, invariantes). Testes que baixam modelo ficam fora do default.

## Fora de escopo

Benchmark/medição (Fase 3), GNOMON (Fase 2.5), tokenizer real, integração AXON, facade `Glyph`
completa do README (os três retrievers + script de wiring bastam; o facade entra quando agregar valor).
