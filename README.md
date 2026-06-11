# GLYPH

![status](https://img.shields.io/badge/status-in%20development-orange) ![phase](https://img.shields.io/badge/phase-3%20benchmark-blue) ![python](https://img.shields.io/badge/python-3.11%2B-blue)

> GLYPH builds a knowledge graph from documents and code, then serves graph-aware context for retrieval. Document entities come from LLM extraction, code structure from tree-sitter, both behind one extractor port. Retrieval is benchmarked against a vector baseline with confidence intervals (in progress).

## Por que existe

Retrieval por similaridade vetorial ignora a estrutura: quem cita quem, o que se relaciona com o quê. Para corpora com entidades e relações ricas (regras, documentos técnicos) e para código (calls, imports), um grafo entrega contexto que o vetor não vê. GLYPH constrói esse grafo dos dois domínios sob uma abstração só, e mede quando o grafo ganha do vetor e quando não vale o custo.

## O que faz

- Constrói knowledge graph de **documentos** (extração por LLM) e de **código** (tree-sitter, determinístico).
- Serve **retrieval graph-aware**: dado uma query, ancora entidades e expande vizinhança por `hops`.
- Compara contra um **baseline vetorial justo** sobre o mesmo corpus, medido com intervalos de confiança.

## O que NÃO faz (ainda)

- Não faz type inference cross-language completa no código. A resolução de símbolo é por nome no import graph e intra-file. Limitação declarada.
- Não usa tokenizer real nas métricas herdadas do AXON (contagem por estimativa de char). Onde aplicável, o benchmark declara isso.
- Não substitui o parsing do AXON; quando integrado, o GLYPH é a fonte canônica de grafo e o AXON delega.
- Benchmark ainda não publicado. Os números entram após a Fase 3. Até lá, nenhum resultado é afirmado aqui.

## Estado atual

Fases 0–2 completas: modelo de domínio + ports + adapter NetworkX (F0), extração documental por LLM com grafo persistido sobre o Monster Manual (F1), e retrieval graph-aware + baseline vetorial justo + híbrido sob um contrato único (F2).

Fase 3 (benchmark) em andamento: query set congelado ([eval/queries.json](eval/queries.json)), passo de geração instrumentado (tokens/latência reais), adapter `RagTarget` e judge OSS na nuvem para o GNOMON, harness que agrega com CIs de bootstrap, e o gate de regressão (`make benchmark`). Os **números ainda não foram publicados** — saem após uma corrida real com chaves; até lá [METRICS.md](METRICS.md) declara que está pendente. Metodologia em [ADR-G4](docs/decisions/dec-g4-eval-methodology.md). Plano em [docs/GLYPH_PLAN.md](docs/GLYPH_PLAN.md).

## Pré-requisitos

- Python 3.11+
- [TODO: confirmar na Fase 1] provedor de LLM para o `DocumentExtractor` (Claude ou OpenAI via env)
- Opcional: Neo4j 5+ apenas se usar o adapter (o default NetworkX não exige servidor)

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

| Variável | Obrigatório | Descrição | Exemplo |
|---|---|---|---|
| `GLYPH_LLM_PROVIDER` | sim (document) | Provedor de extração documental | `anthropic` |
| `GLYPH_LLM_API_KEY` | sim (document) | Chave do provedor | `sk-...` |
| `GLYPH_STORE_BACKEND` | não | Backend de grafo | `networkx` (default) / `neo4j` |
| `GLYPH_NEO4J_URI` | só com neo4j | Conexão Neo4j | `bolt://localhost:7687` |

[TODO: confirmar nomes finais quando a extração documental entrar (Fase 1)]

## Como rodar os testes

```bash
pytest                      # suíte completa
pytest --cov=glyph          # com cobertura
pytest tests/architecture   # invariantes de arquitetura
```

## Uso (API planejada)

```python
from glyph import Glyph
from glyph.extract import DocumentExtractor, CodeExtractor

g = Glyph(store="networkx")

# domínio documento
g.build(source="corpus/", extractor=DocumentExtractor())
ctx = g.retrieve("how does spell resistance interact with elemental damage", hops=2)

# domínio código
g.build(source="repo/", extractor=CodeExtractor(languages=["python", "java"]))
ctx = g.retrieve(symbol="module.func", hops=2)
```

[TODO: confirmar assinatura final após Fase 2]

## Decisões de design

ADRs em [docs/decisions/](docs/decisions/):

- **ADR-G1**: Extractor port + escolha de backend (NetworkX default, Neo4j adapter).
- **ADR-G2**: schema de extração documental (entidades/relações DeD).
- **ADR-G3**: baseline vetorial justo + contrato de saída unificado.
- **ADR-G4**: metodologia de eval (query set, judge OSS reference-free, bootstrap CI, custo).
- **ADR-G5**: resolução de símbolo no code extractor (por nome único, limitação declarada).

Resultados e reprodução: [METRICS.md](METRICS.md) (pendente da corrida real). Artigo técnico (validation-first) em [docs/article.md](docs/article.md); claims de portfólio em [docs/portfolio.md](docs/portfolio.md).

## Relação com AXON

O AXON consome o GLYPH como dependência: o `GraphContextSource` do AXON (ADR-102 no repo do AXON) delega para `glyph.integration.GraphContextSource` desta lib. Code-graph e document-graph viram dois usos do mesmo núcleo. Contrato e fronteira em [docs/axon-integration.md](docs/axon-integration.md).

## Contribuindo

Ver [CONTRIBUTING.md](CONTRIBUTING.md). TDD por sub-task, ADR por decisão arquitetural, invariantes verificados no CI.

## Licença

MIT. Ver [LICENSE](LICENSE).
