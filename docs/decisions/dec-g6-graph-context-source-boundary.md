# ADR-G6: GraphContextSource é a fronteira de produto do GLYPH

**Data:** 2026-06-11
**Status:** Aceito

## Contexto

O GLYPH passa a ser usado como **biblioteca de produto** por consumidores externos (o AXON,
via dec-116, e clientes futuros), não só pelo próprio benchmark. Um consumidor precisa de uma
fronteira **nomeada e estável** para retrieval graph-aware, em vez de depender dos detalhes de
construção do `GraphRetriever` (embedder + lista de nós + `hops`/`anchors`), que são internos e
podem evoluir (fusão híbrida, anchoring diferente, mudança no node-listing).

O `glyph.integration.GraphContextSource` já existia como facade fino, mas expunha o método
`context()`, paralelo — e portanto **não satisfazia** o port canônico
`glyph.retrieval.port.Retriever` (`retrieve(query, token_budget) -> ContextPack`). Dois nomes
para o mesmo contrato é justamente a incoerência que uma biblioteca de produto não deve ter.

## Decisão

- `GraphContextSource` **satisfaz o port `Retriever`**: o método externo é `retrieve(query,
  token_budget) -> ContextPack`, idêntico ao contrato de todos os braços. Assim o facade é um
  `Retriever` estrutural e entra onde quer que um `Retriever` seja esperado. Um teste trava a
  conformância (`isinstance(source, Retriever)`).
- **Duas entradas, o mesmo objeto:**
  - `GraphContextSource(store, embedder, nodes)` — **in-memory**: o chamador já tem um
    `GraphStore` e a lista de nós (caso do AXON, que os monta a partir do grafo SQLite).
  - `GraphContextSource.from_graph_file(path, embedder)` — **persistido**: carrega um grafo
    NetworkX (documento ou código) do disco, dobrando load + node-listing + wiring numa chamada.
- A fronteira pode evoluir por trás de `retrieve()` sem quebrar consumidores. O `nodes` segue
  explícito no construtor (o `GraphStore` port não enumera nós, e o `GraphRetriever` já os exige);
  não inventamos um método de listagem no port só para esconder esse argumento.

## Consequências

- O GLYPH expõe **um** contrato de retrieval (`Retriever`), sem método paralelo.
- O AXON (dec-116) passa a delegar a `GraphContextSource(...).retrieve(...)` em vez de instanciar
  o `GraphRetriever` direto — depende da fronteira, não dos internals (mudança rastreada no repo
  do AXON).
- Renomear `context()` → `retrieve()` é uma quebra de API; como o único consumidor (AXON) ainda
  não usava o facade, não há compatibilidade a manter. Mudanças futuras na fronteira passam a ser
  decisões conscientes registradas aqui.
