# SPEC: GLYPH

> Biblioteca de knowledge graph unificada sobre documentos e código. Núcleo de grafo comum, dois extractors plugáveis sob um port, retrieval graph-aware, medido contra baseline vetorial justo com intervalos de confiança. Document-first.

## Problema

Retrieval vetorial recupera por similaridade semântica e ignora relações explícitas entre entidades. Para corpora densos em relações (documentos técnicos, regras) e para código (calls, imports, inheritance), o contexto estrutural importa: a resposta certa muitas vezes está a uma ou duas arestas de distância, não no chunk mais parecido. GLYPH constrói o grafo desses dois domínios sob uma abstração única e mede o ganho do retrieval graph-aware sobre o vetorial.

## Tese mensurável

Contexto graph-aware supera vector-only em relevância e/ou eficiência de token para queries que dependem de relações entre entidades. Onde a query é factual simples, o vetor pode bastar. GLYPH reporta os dois casos, com custo e latência como métricas de primeira classe.

## Princípio de design

Os dois domínios compartilham núcleo de grafo, store, retrieval e medição. Diferem apenas na **extração**: documento é probabilístico (LLM lê prosa, infere relações com erro), código é determinístico (tree-sitter, a relação é fato). A fronteira certa é um `Extractor` port com dois adapters, não um schema único tentando servir aos dois.

## Escopo IN

1. **Núcleo de grafo**: modelo `Node`/`Edge`/`EdgeType` (Pydantic v2), genérico o suficiente para os dois domínios, com tipos de aresta separados por domínio.
2. **`GraphStore` port**: NetworkX como default embedded (pip-installable, zero servidor), Neo4j como adapter smoke-tested.
3. **`Extractor` port** com dois adapters:
   - `DocumentExtractor` (LLM): ingestão de PDF, chunking estrutura-aware, extração entidade/relação.
   - `CodeExtractor` (tree-sitter): Python + Java, alinhado ao `graph_extractor` do AXON.
4. **Retrieval graph-aware**: ancoragem de entidades + expansão de vizinhança por `hops`.
5. **Baseline vetorial justo** (Python, mesmo corpus): chunk + embedding + vector store. Controle real do experimento.
6. **Benchmark GNOMON**: graph vs vetor vs híbrido, com CIs (percentile bootstrap), custo e latência.
7. **Baseline reproduzível**: dataset/fixture versionado + regeneração + check de regressão no número publicado.

## Escopo OUT (honestidade de claim)

- Type inference cross-language completa no código. Resolução por nome no import graph + intra-file. Declarada.
- Tokenizer real onde herdar a contagem por char-estimate do AXON. Declarado no benchmark.
- Backend de grafo dedicado externo além do adapter Neo4j. Default fica embedded (local-first).
- Reimplementar parsing que o AXON já tem. O `CodeExtractor` alinha em vez de duplicar.

## Arquitetura

Ver [ARCHITECTURE.md](ARCHITECTURE.md). Resumo: Extractor port (2 adapters) → núcleo de grafo (Node/Edge) → GraphStore port (NetworkX/Neo4j) → retrieval graph-aware → benchmark GNOMON contra baseline vetorial.

Invariantes (verificados no CI):
- Extractors implementam o port; o núcleo de grafo não importa extractor concreto.
- Retrieval e store dependem do modelo de domínio, não do extractor.
- O baseline vetorial é implementação justa, não espantalho.

## Validação

| Camada | Critério |
|---|---|
| Técnica | Ports funcionam; NetworkX e Neo4j passam o mesmo smoke test; retrieval determinístico e reproduzível |
| Evidência | Relevância, token efficiency, custo, latência: graph vs vetor vs híbrido, com CIs, sobre corpus real |
| Honestidade | Baseline reproduzível do repo; limitações declaradas; benchmark não afirmado antes de rodar |

Corpus de validação documental: 10-15 livros de DeD (150-300 pág), dado real e denso em entidades. Gate de custo em 1 livro antes de escalar (extração LLM é cara no volume).

## Fasamento

Detalhe em [GLYPH_PLAN.md](GLYPH_PLAN.md). Resumo: Fase 0 fundação → Fase 1 document extraction → Fase 2 retrieval + baseline → Fase 2.5 destravar GNOMON → Fase 3 benchmark (claim das vagas) → Fase 4 code → Fase 5 integração AXON → Fase 6 publicação. Cada fase tem entregável publicável isolado.

## Quality gates

- TDD por sub-task.
- Cobertura com gate no CI.
- Invariante de arquitetura testado.
- ADR por decisão arquitetural (ADR-G1..G5).
- Custo medido antes de escalar extração.
- Número publicado reproduzível do repo, senão não publicado.

## Decisões abertas (não bloqueiam Fase 0)

- Lib de embedding e vector store do baseline (Fase 2): definir entre sentence-transformers + pgvector ou o que GNOMON/AXON já usam, por consistência.
- GNOMON empacotado (Fase 2.5): empacotar vs vendorizar a lógica de eval.
- Licença (sugerido MIT para portfólio público).
