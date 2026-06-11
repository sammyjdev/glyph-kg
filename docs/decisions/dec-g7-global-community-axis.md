# ADR-G7: Eixo global por comunidades

**Data:** 2026-06-11
**Status:** Aceito

## Contexto

O retrieval local (`GraphRetriever`) ancora a query e expande a vizinhança por `hops` —
ótimo para *"o que depende de X?"*, ruim para perguntas temáticas/de sense-making
(*"quais são os grandes subsistemas?"*). Falta um **eixo global**: detectar comunidades no
grafo, resumi-las, e recuperar esses resumos. Espelha o GraphRAG global, mas dentro do
perfil do GLYPH (lib pura, reprodutível, validation-first, mesmo contrato `ContextPack`).

## Decisão

- **Detecção: Louvain nativo do networkx** (`nx.community.louvain_communities`) — zero
  dependência nova (vs. Leiden, que traria `igraph`/`leidenalg`). Sobre a **projeção
  estrutural** do grafo (`STRUCTURAL_EDGES`: DEFINES/IMPORTS/CALLS/INHERITS/REFERENCES) —
  arestas de documento/decisão (MENTIONS, RELATES_TO, …) misturariam entidades não
  relacionadas, então são excluídas.

- **Reprodutibilidade: seed obrigatório + ordem determinística.** Louvain é estocástico;
  só o seed não basta (o resultado do networkx depende da ordem de iteração), então nós e
  arestas entram ordenados. Mesmo grafo + seed → mesmas comunidades.

- **Comunidades no mesmo grafo** como nós `COMMUNITY` + arestas `CONTAINS` (community→membro),
  reusando o modelo. Resumo e título em `Node.attrs` (`summary`/`title`) — sem mudança de
  schema (`attrs` já existe).

- **Isolamento por projeção de traversal podada, não indexação disjunta (opção B).** Um nó
  `COMMUNITY` é um super-hub ligado a todos os membros; traversar `CONTAINS` colapsaria toda
  distância intra-comunidade para 2 hops e vazaria o overlay no retrieval local. Fix: o
  `GraphStore` ganha filtro **genérico** `exclude_node_types`/`exclude_edge_types` em
  `subgraph`/`neighbors`/`shortest_path` (a store **não sabe** o que é COMMUNITY — mantém o
  port trocável p/ Neo4j); a **política** mora na camada de retrieval (o `GraphRetriever`
  passa `{COMMUNITY}`/`{CONTAINS}`). O `CommunityRetriever` não traversa (ancora direto nos
  resumos). Footgun de esquecer coberto por teste de invariância topológica, não por
  acoplar a store.

- **id de comunidade = hash estável (hashlib) dos membros ordenados**, não índice. Comunidades
  inalteradas mantêm o id entre rebuilds → a sumarização (paga) pula as não-mudadas.

- **Fronteira de sumarização: lógica no GLYPH, LLM injetado.** `summarize_communities` monta o
  prompt a partir dos membros e preenche `title`+`summary` via um `CommunitySummarizer`
  injetado (Protocol) — provider/API key ficam fora da lib (espelha `DocumentExtractor`). O
  AXON injeta o modelo e dispara no hook (spec-irmão).

- **`CommunityRetriever` satisfaz o port `Retriever`** (`retrieve(query, token_budget) ->
  ContextPack`, `mode="community"`), como todo braço — drop-in onde um `Retriever` é esperado.

## Consequências

Eixo global mensurável com o mesmo harness (braço global = follow-up, depende de um global
query set). O retrieval local fica provadamente imune ao overlay. A store permanece genérica;
a política de overlay é da camada de retrieval. Orquestração (hook, tool MCP `get_global_context`,
modelo real) é o spec-irmão no AXON — fora deste repo. Follow-ups: rodada real de sumarização
(custa LLM), braço de benchmark global, e troca de anchoring `summary`→`title` se o benchmark
mostrar diluição de embedding.
