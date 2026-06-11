# GLYPH — Plano de Execução

> Biblioteca de knowledge graph unificada: núcleo de grafo comum, dois extractors plugáveis (documento via LLM, código via tree-sitter), retrieval graph-aware, medido contra baseline vetorial com GNOMON. Document-first (cobre as vagas), code depois (on-brand AXON).

## Arquitetura alvo

```
                      Extractor port
              +-------------+-------------+
              |                           |
     DocumentExtractor             CodeExtractor
     (LLM, probabilístico)         (tree-sitter, determinístico)
              |                           |
              +----------> Graph <--------+
                          (Node/Edge model)
                              |
                       GraphStore port
                    +---------+---------+
                    |                   |
              NetworkX (default)   Neo4j (adapter)
                              |
                    graph-aware retrieval
                              |
              GNOMON benchmark  vs  vector baseline (Python, mesmo corpus)
```

Invariantes (ArchUnit-style):
- Extractors implementam o port; núcleo de grafo não conhece extractor concreto.
- Retrieval e store dependem só do modelo de domínio, não do extractor.
- O baseline vetorial é implementação justa e real, não espantalho.

## Decisões fechadas

- Versão forte: núcleo comum + Extractor port com dois adapters. Não um schema único servindo aos dois domínios.
- Backend: GraphStore port, NetworkX default + Neo4j adapter (keyword de CV, smoke-tested).
- Ordem: DocumentExtractor primeiro (driver = cobrir vagas), CodeExtractor depois.
- Extração documental: LLM-based (justificada para prosa).
- Consumido pelo AXON via `GraphContextSource` (ADR-102), que passa a delegar para GLYPH.

## Dependências e bloqueios

- **GNOMON (auditado): já é pip-installable** (hatchling, src-layout; `run_eval` e `aggregate_metric` importáveis, não presos em `__main__`). **Deixa de bloquear a Fase 3.** O único item é o GLYPH referenciá-lo por path/git no `pyproject` (não há PyPI/remote) — uma linha, sub-task da Fase 3, não fase própria.
- **Corpus:** 10-15 livros DeD (150-300 pág). Dado real, denso em entidades. Extração LLM será cara no volume; medir custo em 1 livro antes de escalar (gate na Fase 1).
- **APIs voláteis** (LLM extraction, NetworkX, Neo4j driver, lib de PDF): confirmar na doc atual antes de cada fase de implementação.

---

## Fase 0 — Fundação

Objetivo: esqueleto da lib, modelo de domínio, ports, quality gates. Sem extração ainda.

- **P0.1** Scaffold do repo: `pyproject.toml` (pip-installable), estrutura hexagonal (`glyph/model`, `glyph/store`, `glyph/extract`, `glyph/retrieval`, `glyph/eval`), licença, README stub.
- **P0.2** Modelo de domínio (Pydantic v2): `Node`, `Edge`, `EdgeType`, `NodeType`. Genérico o suficiente para código e documento, específico o suficiente para não ser sopa. Tipos de aresta separados por domínio (`CALLS`/`IMPORTS` para código; `RELATES_TO`/`MENTIONS`/etc. para documento).
- **P0.3** `GraphStore` port (Protocol): `upsert_nodes`, `upsert_edges`, `neighbors(node, hops)`, `subgraph(seed, hops)`, `shortest_path`.
- **P0.4** Adapter NetworkX (default): implementa o port in-memory com persistência (pickle/graphml).
- **P0.5** `Extractor` port (Protocol): `extract(source) -> tuple[Sequence[Node], Sequence[Edge]]`.
- **P0.6** Quality gates: pytest + coverage gate, teste de invariante de arquitetura (núcleo não importa extractor concreto), CI (lint, types, test).
- **ADR-G1**: Extractor port + escolha de backend (NetworkX default, Neo4j adapter, justificativa pip-installable).

Entregável: lib instalável vazia com modelo, ports e NetworkX. Conteúdo: nenhum ainda.

---

## Fase 1 — Document extraction (prioridade, cobre vagas)

Objetivo: construir KG documental real a partir do corpus DeD.

- **P1.1** Ingestão de PDF em Python (pymupdf ou similar): PDF → texto por seção/página, com metadados (livro, página).
- **P1.2** Chunking de documento (estrutura-aware: capítulo/seção, não corte cego).
- **P1.3** `DocumentExtractor` (LLM): prompt de extração entidade/relação, parse para `Node`/`Edge`. Schema de entidades do domínio DeD (criatura, magia, item, regra, local) e relações (resiste a, requer, pertence a, etc.).
- **P1.4** Gate de custo/qualidade em 1 livro: rodar P1.1-1.3 num único livro, medir custo de extração (tokens × preço), latência, e amostragem manual de qualidade das relações. **Não escalar antes de aprovar este gate.**
- **P1.5** Escalar para o corpus completo (10-15 livros), persistir o grafo via NetworkX.
- **ADR-G2**: estratégia de extração documental (schema de entidades, modelo de prompt, limitação probabilística declarada).

Entregável: KG documental persistido sobre corpus real. Conteúdo: post curto sobre extração de KG de documento.

---

## Fase 2 — Retrieval + baseline vetorial justo

Objetivo: retrieval graph-aware e o baseline contra o qual medir. Os dois lados da comparação.

- **P2.1** Retrieval graph-aware: dado uma query, ancorar entidades e expandir vizinhança por `hops`, retornar contexto estrutural.
- **P2.2** Baseline vetorial (Python, mesmo corpus): chunk + embedding + vector store (in-memory ou pgvector). Implementação real e justa, igual cuidado do graph path. Este é o controle do experimento.
- **P2.3** Modo híbrido: fusão graph + vetor (terceiro braço do benchmark).
- **P2.4** Contrato de saída unificado (`Segment`/`ContextPack`) para os três modos, comparável token-a-token.
- **ADR-G3**: desenho do baseline (por que é justo, parâmetros, igualdade de budget entre os braços).

Entregável: três modos de retrieval funcionais sobre o mesmo corpus. Conteúdo: nenhum até medir.

---

## Fase 2.5 — Destravar GNOMON (RESOLVIDO pela auditoria)

A auditoria do GNOMON dissolveu esta fase: o GNOMON **já** é pip-installable e `run_eval`/
`aggregate_metric` são importáveis de verdade. Não há empacotamento a fazer (cai a decisão A vs B).
O que sobrou é pequeno e migra para a Fase 3 como P3.0: referenciar o GNOMON por path/git no
`pyproject` do GLYPH e escrever o adapter `RagTarget` (ver abaixo). Esta fase deixa de ser um
bloqueio.

---

## Fase 3 — Benchmark + baseline reproduzível

Objetivo: o número honesto. Esta fase é o claim das vagas e o artigo.

- **P3.0** Consumir o GNOMON: referenciar por path/git no `pyproject` do GLYPH; escrever um adapter
  **`RagTarget`** (o GNOMON é pull-based — `run_eval` chama `target.query(question)`). O GLYPH
  pré-computa os resultados de cada braço (graph, vetor, híbrido) keyed por question e expõe um
  `RagTarget` por braço que devolve o resultado armazenado via `query(question) -> RagResponse`.
  Roda `run_eval` uma vez por braço. (Push-based `evaluate()` no GNOMON é melhoria futura — ROADMAP
  do GNOMON, não do GLYPH.) **Restrições de contrato do GNOMON v1, declaradas:**
  - `RagResponse` exige `total_tokens` e `latency_ms` (validados ≥ 0) → **instrumentar token e
    latência reais nos três braços é requisito**, não opcional (placeholder polui o relatório).
  - Métricas v1 = **`faithfulness` e `context_precision`** apenas (answer relevance e context recall
    são v2, não-construídos). O artigo reporta só essas duas, com CIs — não prometer recall.
  - **Custo em moeda não existe no GNOMON**, só tokens; o GLYPH calcula US$ a partir de `total_tokens`
    por fora.
- **P3.1** Harness de benchmark: roda os três modos (graph, vetor, híbrido) sobre um conjunto de queries do corpus DeD.
- **P3.2** Conjunto de queries: query set realista (perguntas que exigem relações entre entidades, onde grafo deveria ganhar; e factuais simples, onde vetor pode bastar). Reportar os dois.
- **P3.3** Métricas via GNOMON: `faithfulness` e `context_precision` (as duas do v1), mais token efficiency, custo (calculado pelo GLYPH a partir de tokens) e latência. Todas com CI (percentile bootstrap).
- **P3.4** Baseline reproduzível (aprendizado C1b+3c): dataset fixo versionado + `make benchmark` que regenera agregados; número de referência no `METRICS.md` com check de regressão (build falha se divergir além de tolerância). Corpus/queries em estado congelado para reprodutibilidade.
- **P3.5** Relatório honesto: tabela com CIs incluindo onde o grafo **não** ganhou; declarar n (queries), largura de CI, e que tokens são contagem real ou estimativa.
- **Backlog de qualidade da extração** (observado no gate P1.4): normalizar casing de labels e avaliar erros probabilísticos de relação (ex. ANKHEG `resists ácido`). Detalhe em [decisions/phase3-quality-backlog.md](decisions/phase3-quality-backlog.md).
- **ADR-G4**: metodologia de eval (query set, bootstrap, definição de relevância).

Entregável: tabela de resultados reproduzível. Conteúdo: artigo principal ("GraphRAG vs vector retrieval sobre corpus real, medido").

---

## Fase 4 — Code extraction (on-brand, AXON)

Objetivo: segundo extractor, segundo domínio, segundo claim de KG.

- **P4.1** `CodeExtractor` (tree-sitter): reusar/alinhar com a lógica já existente no AXON (`graph_extractor.py`) em vez de reimplementar do zero. Decisão: GLYPH passa a ser a fonte canônica e o AXON delega, ou GLYPH espelha o comportamento. Resolver na P4.1.
- **P4.2** Linguagens-alvo: Python + Java (o set indexado do AXON), TS opcional.
- **P4.3** Validar em repos reais num SHA fixo (AXON, GNOMON), gerar grafo de código.
- **P4.4** Rodar o mesmo harness de benchmark (Fase 3) no domínio código: contexto estrutural vs vetorial para tarefas de agente.
- **ADR-G5**: resolução de símbolo no code extractor (limitação por nome/import graph declarada).

Entregável: KG de código validado. Conteúdo: post comparando os dois domínios no mesmo framework.

---

## Fase 5 — Integração AXON

Objetivo: fechar o loop com o reescopo já especificado (ADR-102/103).

- **P5.1** `GraphContextSource` do AXON (ADR-102) passa a delegar para a lib GLYPH como dependência.
- **P5.2** Ajustar ADR-102/103: nota de uma linha apontando que o `GraphContextSource` é implementado pelo GLYPH.
- **P5.3** Consolidação do grafo do AXON (ADR-103) permanece; GLYPH consome o grafo SQLite consolidado.
- **P5.4** Teste de integração: AXON serve contexto graph-aware via GLYPH sobre MCP.

Entregável: AXON consumindo GLYPH. Conteúdo: post sobre a integração context-source.

---

## Fase 6 — Publicação / portfólio

Objetivo: converter o trabalho em evidência verificável.

- **P6.1** README do GLYPH com métricas verificadas, limitações declaradas, link reproduzível do benchmark.
- **P6.2** Artigo técnico (validation-first: claim checklist contra a codebase, número verificado antes de publicar).
- **P6.3** Claim de CV/LinkedIn: "built a knowledge-graph library spanning document and code domains; benchmarked graph-aware vs vector retrieval with confidence intervals." Cobre Near e Marlabs ao pé da letra.
- **P6.4** Endorsement/visibilidade: showcase nos Discords (Qdrant, MCP), link no primeiro comentário, não no corpo do post.

Entregável: portfólio publicável. Cobre o gap de KG das vagas com evidência medida.

---

## Sequência crítica

```
Fase 0  ->  Fase 1  ->  Fase 2  ->  Fase 3 (claim das vagas)
                                          |    (GNOMON pronto; P3.0 = referenciar + adapter RagTarget)
                                          |
                    Fase 4 (code)  -------+
                          |
                    Fase 5 (AXON)  ->  Fase 6 (publicação)
```

O caminho até a Fase 3 é o que cobre as vagas. Fases 4-5 fortalecem o moat e não bloqueiam o claim documental. Cada fase tem entregável publicável isolado: você acumula conteúdo sem esperar o fim.

## Gates globais

- TDD por sub-task; nada entra sem teste.
- Invariante de arquitetura verificado no CI.
- Cada fase fecha com seu ADR antes de avançar.
- Custo medido cedo (P1.4) antes de escalar extração.
- Baseline justo é condição de validade do benchmark, não opcional.
- Número publicado é reproduzível do repo (P3.4), senão não é publicado.

## Aberto antes da Fase 0

- Nome: GLYPH unificado confirmado. Manter.
- Numeração de ADR: ADR-G1..G5 são internos do GLYPH (repo próprio); independentes do dec-NNN do AXON.
- Lib de embedding e vector store do baseline (P2.2): definir na Fase 2 (opções: sentence-transformers + pgvector, ou o que o GNOMON/AXON já usam, para consistência).
