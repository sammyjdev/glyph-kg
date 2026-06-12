# Spec P7 — Eixo global por comunidades (GLYPH)

**Data:** 2026-06-11
**Status:** Implementado + **medido** (lado GLYPH) — ver dec-g7. Benchmark global rodado (n=8, dois judges): resumo de comunidade dá qualidade igual-ou-melhor a ~½ dos tokens. Orquestração AXON = spec-irmão pendente.
**ADR associada:** [dec-g7](dec-g7-global-community-axis.md)
**Fronteira:** lógica e detecção no GLYPH (lib pura, LLM injetado, espelha `DocumentExtractor`). Orquestração (hook post-commit, MCP `get_global_context`, provider/API key) é o spec-irmão no AXON — **fora deste spec**.

---

## 1. Objetivo

Adicionar um **eixo global** ao retrieval: detectar comunidades no grafo, resumi-las via LLM injetado, e recuperar esses resumos para perguntas temáticas/de sense-making ("quais são os grandes subsistemas deste repo?") — que a expansão local por vizinhança (`GraphRetriever`, 2 hops) não responde bem.

O local responde "o que depende de X?"; o global responde "como isto se organiza?". São eixos complementares atrás do mesmo port `Retriever`.

## 2. Decisões-núcleo (locked)

1. **Lógica no GLYPH, LLM injetado.** A lib detém detecção + sumarização; recebe um `llm` como dependência (sem provider/API key na lib), espelhando `DocumentExtractor`. O AXON injeta o modelo e dispara no hook.
2. **Comunidades vivem no mesmo grafo** como nós `COMMUNITY` + arestas `CONTAINS` (community→membro).
3. **Isolamento por projeção de traversal, NÃO por indexação disjunta.** ⬅️ *correção crítica vs. desenho original — ver §4.*

## 3. Componentes

### 3.1 Modelo (1 linha cada)

- `glyph/model/node.py` → `NodeType`: adicionar `COMMUNITY = "community"`.
- `glyph/model/edge.py` → `EdgeType`: adicionar `CONTAINS = "contains"`.
- `glyph/model/contract.py` → `Mode`: `Literal["graph", "vector", "hybrid", "community"]`.

### 3.2 `glyph/retrieval/community.py` (novo)

```python
# Allowlist default = subgrafo ESTRUTURAL de código. NÃO incluir arestas de
# documento/decisão (MENTIONS, RELATES_TO, etc.) senão clusters misturam ADR com função.
STRUCTURAL_EDGES = frozenset({
    EdgeType.DEFINES, EdgeType.IMPORTS, EdgeType.CALLS,
    EdgeType.INHERITS, EdgeType.REFERENCES,
})

def detect_communities(
    store, nodes, *, seed: int,
    edge_types: frozenset[EdgeType] = STRUCTURAL_EDGES,
) -> list[Community]:
    """louvain_communities(seed=seed) sobre o subgrafo estrutural.
    - EXCLUI nós COMMUNITY pré-existentes (idempotência — ver build §5).
    - Alimenta o grafo em ORDEM DETERMINÍSTICA (sorted by node.id); seed
      sozinho NÃO reproduz sem ordem fixa de iteração no networkx.
    - Filtra arestas por `edge_types`.
    """

def to_graph_elements(communities) -> tuple[list[Node], list[Edge]]:
    """Nós COMMUNITY + arestas CONTAINS (community→membro), para upsert no mesmo store.

    id da comunidade = "community:" + hash(tuple(sorted(member_ids)))   ⬅️ NÃO índice N.
    attrs={"members": k}. (summary/title preenchidos em summarize_communities.)

    Motivo do hash: id estável entre rebuilds → comunidades inalteradas mantêm o id
    → summarize PULA as não-mudadas → corta custo de LLM no hook (alavanca, não cosmético).
    """

def summarize_communities(communities, member_text, llm) -> list[Node]:
    """Monta prompt dos membros, chama o `llm` injetado, devolve nós COMMUNITY com
    attrs["summary"] E attrs["title"] (1 linha temática) preenchidos.
    Espelha DocumentExtractor: nenhum provider/API key aqui.

    title: hedge barato contra diluição de embedding (resumo multi-frase num único
    vetor dilui o centroide). v1 ancora no summary; title fica pronto p/ um retriever
    futuro trocar sem migração de schema.
    """
```

### 3.3 `CommunityRetriever(Retriever)` (em `community.py`)

```python
class CommunityRetriever:
    def __init__(self, community_nodes, embedder):
        # embeda attrs["summary"] de cada nó COMMUNITY (apenas COMMUNITY).
    def retrieve(self, query, token_budget=1000) -> ContextPack:
        # ancora a query nos resumos e os devolve como Segments.
        # mode="community". SEM expansão de subgrafo (sem _store.subgraph).
        return pack("community", segments, token_budget)
```

Satisfaz o port `glyph.retrieval.port.Retriever` (`retrieve(query, token_budget) -> ContextPack`), igual aos outros braços — teste de conformância `isinstance(r, Retriever)` como em dec-g6.

## 4. Isolamento — a correção crítica

**O furo do desenho original:** "indexação disjunta" isola só o *anchoring*. Mas `GraphRetriever.retrieve` chama `store.subgraph(anchors, hops)`, e `NetworkXStore.subgraph` (`networkx_store.py:49`) faz BFS sobre o **grafo inteiro**:

```python
reachable.update(nx.single_source_shortest_path_length(undirected, src, cutoff=hops))
```

Logo, mesmo ancorando só em nós não-COMMUNITY, a expansão de hops atravessa arestas `CONTAINS`. Dois efeitos:

1. **Vazamento:** nó COMMUNITY entra no ContextPack local.
2. **Distorção (pior):** cada COMMUNITY vira **super-hub** ligado a todos os membros → dois símbolos quaisquer da mesma comunidade ficam a 2 hops (membro→COMMUNITY→membro). Corrompe TODAS as distâncias do retrieval local.

**Fix (mesmo grafo, projeção podada) — decisão B, feita por inteiro:**

- Params **genéricos** `exclude_node_types` / `exclude_edge_types` (keyword, default vazio) nos **três** métodos de traversal do `GraphStore`: `subgraph`, `neighbors`, `shortest_path`. A store fica burra — só honra um filtro, **não sabe** o que é COMMUNITY (mantém o port trocável; o adapter Neo4j não precisa saber do overlay). A **política mora na camada de retrieval**.
- O `GraphRetriever` passa `{COMMUNITY}` / `{CONTAINS}` (constantes `_OVERLAY_*` no módulo). A camada global (`CommunityRetriever`) **não traversa** (anchoring direto em resumos), então não precisa da projeção.
- Os tools MCP do AXON (`get_graph_neighbors`/`get_graph_path`) passam o mesmo overlay — **no spec-irmão do AXON**, não aqui.
- Footgun (esquecer de excluir) coberto pelo **teste de invariância topológica** (§6.5), não por acoplar a store.

**No dec-g7 cravar:** "isolamento por **projeção de traversal podada**, não indexação disjunta; filtro genérico na store, política na camada de retrieval (opção B)."

## 5. Fluxo de dados

- **Build** (AXON dispara depois):
  `load → detect_communities(seed) → to_graph_elements → upsert → summarize_communities(llm) → upsert COMMUNITY c/ summary+title → persist`.
  **Idempotência:** antes de detectar, remover nós `COMMUNITY` + arestas `CONTAINS` anteriores (ou `detect` exclui COMMUNITY da entrada **e** o build limpa os antigos), senão acumula a cada commit.
- **Query:** `CommunityRetriever.retrieve(query_global) → resumos como ContextPack`.

## 6. Testes / validação (deste spec — gate 100%, custo zero)

LLM **fake injetado** + embedder fake:

1. **Detecção** particiona um grafo pequeno conhecido (assert partição esperada, reprodutível com seed fixo + ordem determinística).
2. **summarize** preenche `attrs["summary"]` e `attrs["title"]` via stub.
3. **CommunityRetriever** ancora na comunidade certa (query temática → resumo correto no topo).
4. **Isolamento — vazamento:** retrieval local NÃO contém nós COMMUNITY.
5. **Isolamento — invariância topológica** ⬅️ *novo, obrigatório*: em **topologia realista** (comunidades de tamanho >2, nós estruturalmente distantes), dois nós distantes que caem na mesma comunidade **continuam distantes** no subgrafo local depois do build. Pega o efeito super-hub que o teste 4 não pega. (Topologia não precisa de LLM real — só stub.)
6. **Conformância de port:** `isinstance(CommunityRetriever(...), Retriever)`.

## 7. Follow-ups (fora deste spec)

- Rodada **real** de sumarização (corpus Monster Manual / DeD — custa LLM).
- Braço de **benchmark global** (depende de um global query set que ainda não existe).
- Eventual troca de anchoring `summary` → `title`/title-weighted **se** o benchmark mostrar diluição.

## 8. Fronteira AXON (spec-irmão, não aqui)

- Hook post-commit dispara `detect + summarize + persist` com o LLM do AXON.
- Novo tool MCP `get_global_context` (separado de `get_graph_context`).
- Caller escolhe local vs global — **sem roteador automático no v1**.

## 9. dec-g7 (ADR a escrever junto)

Registra: eixo global; Louvain nativo + seed + ordem determinística; **isolamento in-graph por projeção de traversal podada** (não indexação disjunta); id de comunidade por hash de membros (alavanca de re-summarização); fronteira de sumarização (lógica GLYPH / LLM injetado); `title`+`summary` no nó COMMUNITY.
