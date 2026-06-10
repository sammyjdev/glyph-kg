# Arquitetura do GLYPH

## Visão de camadas

GLYPH segue arquitetura hexagonal: o domínio (grafo) no centro, extração e persistência como ports com adapters intercambiáveis.

```
glyph/
  model/        # domínio: Node, Edge, EdgeType, NodeType (Pydantic v2)
  extract/      # Extractor port + adapters (document, code)
  store/        # GraphStore port + adapters (networkx, neo4j)
  retrieval/    # graph-aware retrieval sobre o port
  baseline/     # vector baseline (controle do benchmark)
  eval/         # integração GNOMON, métricas, CIs
```

Regra de dependência: as setas apontam para dentro. `extract`, `store`, `retrieval`, `baseline` e `eval` dependem de `model`. `model` não depende de ninguém. Nenhum adapter conhece outro adapter.

## Núcleo de grafo (model)

`Node`, `Edge`, `EdgeType`, `NodeType` em Pydantic v2.

Tipos separados por domínio para não criar um schema genérico que não serve a nenhum:
- Código: `NodeType` em {File, Module, Class, Function}; `EdgeType` em {DEFINES, IMPORTS, CALLS, INHERITS, REFERENCES}.
- Documento: `NodeType` em {Entity, Concept, Section}; `EdgeType` em {RELATES_TO, MENTIONS, REQUIRES, RESISTS, ...} conforme o domínio do corpus.

O grafo é agnóstico de origem: depois de construído, retrieval e store tratam nós e arestas igual, independentemente de qual extractor os produziu.

## Extractor port

```python
class Extractor(Protocol):
    def extract(self, source: Source) -> tuple[Sequence[Node], Sequence[Edge]]: ...
```

Dois adapters:

- **DocumentExtractor** (probabilístico). Ingestão de PDF (pymupdf ou similar) → chunking estrutura-aware → prompt de extração entidade/relação a um LLM → parse para `Node`/`Edge`. A extração tem erro; é por isso que o benchmark mede qualidade em vez de assumir.
- **CodeExtractor** (determinístico). tree-sitter para Python e Java, alinhado ao `graph_extractor.py` do AXON. A relação `A CALLS B` é fato extraído da AST, não inferência. Limitação: resolução de símbolo por nome no import graph + intra-file, sem type inference cross-language completa.

A assimetria probabilístico/determinístico é a razão de existirem dois adapters em vez de um extractor genérico.

## GraphStore port

```python
class GraphStore(Protocol):
    def upsert_nodes(self, nodes: Sequence[Node]) -> None: ...
    def upsert_edges(self, edges: Sequence[Edge]) -> None: ...
    def neighbors(self, node: NodeId, hops: int) -> Subgraph: ...
    def subgraph(self, seed: Sequence[NodeId], hops: int) -> Subgraph: ...
    def shortest_path(self, src: NodeId, dst: NodeId) -> Path | None: ...
```

- **NetworkX** (default): in-memory com persistência (graphml/pickle). Zero servidor, pip-installable. O grafo de um corpus documental cabe em memória; para código também na escala dos repos-alvo.
- **Neo4j** (adapter): smoke-tested, openCypher. Existe pelo keyword de CV e pela história de produção, não como default. Não é serviço always-on do projeto.

Ambos passam o mesmo conjunto de testes de contrato. Trocar o backend não muda resultado, só desempenho/escala.

## Retrieval graph-aware

Dado uma query: ancorar entidades relevantes no grafo (por match de nó ou por embedding leve do label), expandir a vizinhança por `hops`, montar o contexto a partir do subgrafo. O contexto retornado é estrutural: as entidades conectadas à âncora, não os chunks mais parecidos.

Saída em contrato unificado (`Segment`/`ContextPack`) comparável token-a-token com o baseline.

## Baseline vetorial (controle)

Implementação real e justa em Python sobre o **mesmo** corpus: chunk + embedding + vector store + top-k por similaridade. Mesmo budget de token dos outros braços. Este é o controle do experimento; enfraquecê-lo invalida o benchmark. Existe um terceiro braço híbrido (fusão graph + vetor).

## Eval (GNOMON)

Mede os três braços (graph, vetor, híbrido) sobre um query set do corpus. Métricas: relevância de contexto, token efficiency, custo, latência. Todas com intervalo de confiança via percentile bootstrap. Resultado reproduzível de um fixture versionado.

## Integração AXON

O AXON consome o GLYPH como dependência. O `GraphContextSource` do AXON (ADR-102 no repo do AXON) delega para `glyph.retrieval`. O grafo de código consolidado do AXON (ADR-103) é uma das fontes que o `CodeExtractor` alimenta ou consome. Isso mantém uma única fonte canônica de lógica de grafo.

## Invariantes verificados no CI

1. `model` não importa nada de `extract`, `store`, `retrieval`, `baseline`, `eval`.
2. Nenhum adapter de extract importa adapter de store, e vice-versa.
3. `retrieval` depende do `GraphStore` port, não de um adapter concreto.
4. O baseline vetorial e o graph retrieval compartilham o mesmo contrato de saída e o mesmo budget de token.
