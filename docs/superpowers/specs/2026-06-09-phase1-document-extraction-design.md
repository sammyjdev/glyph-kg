# Design — Fase 1: Document Extraction (P1.1–P1.4)

**Data:** 2026-06-09
**Status:** Aprovado (brainstorming)
**Escopo desta sessão:** P1.1 → P1.4. P1.5 (escalar o corpus) fica para uma sessão futura.

## Objetivo

Construir um knowledge graph documental real a partir de um livro de Dungeons & Dragons,
sob o `Extractor` port já existente, e medir o custo da extração LLM em um livro antes de
escalar. Entregável: KG do Monster Manual persistido via NetworkX, com números reais de
custo e latência, ADR-G2 commitado, suíte TDD verde e CI passando.

## Corpus

Os três core books PT-BR de D&D 5e estão disponíveis localmente e são o corpus-alvo:

| Livro | Caminho | Papel na Fase 1 |
|---|---|---|
| Monster Manual | `~/Downloads/3-Monster Manual.pdf` | **Âncora.** Gate de custo (P1.4) roda sobre este. |
| Player's Handbook | `~/Downloads/1-Player's Handbook.pdf` | Registrado; entra em fase futura com schema próprio. |
| Dungeon Master Guide | `~/Downloads/2-Dungeon Master Guide.pdf` | Registrado; entra em fase futura com schema próprio. |

Esses caminhos não são committados no repo (PDFs são gitignored e não são nossos para
distribuir). O harness do benchmark referencia o caminho por env/argumento.

### Realidade do PDF (sondada antes do design)

- Monster Manual: 351 páginas, **texto extraível** (não é scan), ~500–700 palavras/página.
- Idioma: **português** ("Manual dos Monstros"). Schema e prompt em PT-BR; enums em inglês.
- **Sem TOC** no PDF. As criaturas aparecem como **cabeçalhos em CAIXA ALTA** no início de
  cada verbete ("ABOCANHADOR MATRAQUEANTE", "DUERGAR", "KUO-TOA"), com número de página.

## Decisões fechadas

1. **Provedor LLM:** Claude Haiku 4.5 (`claude-haiku-4-5`), $1/$5 por 1M tokens (input/output),
   contexto 200K. Default do README (`GLYPH_LLM_PROVIDER=anthropic`). Dá números de custo
   reais para o benchmark.
2. **Ingestão:** texto local via pymupdf (P1.1), não PDF-nativo/vision. Barato, determinístico,
   controle total do chunk — o que o GLYPH_PLAN P1.1 descreve.
3. **Chunking:** por verbete via detecção de cabeçalho (CAIXA ALTA / fonte), com fallback para
   janela por página onde a detecção tiver baixa confiança. Cada criatura ≈ um chunk ≈ uma
   chamada LLM.
4. **Schema:** MM-focado. Criatura = `NodeType.ENTITY`; tipo de dano / condição / habitat =
   `NodeType.CONCEPT`. Relações exercitadas pelo MM (ver ADR-G2).
5. **Saída estruturada:** `messages.parse` com schema Pydantic (Haiku suporta structured
   outputs), mapeado para `Node`/`Edge`.

## Encaixe arquitetural

Arquitetura hexagonal do Phase 0 preservada. Novo pacote `glyph/extract/document/` — um
adapter do `Extractor` port. Importa só de `glyph.model`, `glyph.extract.port` e libs externas
(pymupdf, anthropic). **Não importa `glyph.store`** — o teste de invariantes do Phase 0
(`tests/architecture/test_dependencies.py`) garante a regra.

```
glyph/extract/document/
  __init__.py
  pdf.py          # P1.1: PDF -> [Page(book, number, text, spans)]
  chunk.py        # P1.2: [Page] -> [Chunk(label, text, book, pages)] por verbete
  schema.py       # Pydantic: ExtractedEntity / ExtractedRelation (contrato de saída do LLM)
  prompt.py       # system prompt de extração (PT-BR) + few-shot
  llm.py          # thin adapter Anthropic (Haiku, structured output, captura de usage)
  extractor.py    # DocumentExtractor: implementa Extractor; orquestra pdf->chunk->LLM->Node/Edge
  cost.py         # P1.4: tokens x preço, latência, agregação
```

## Modelo — extensão de EdgeType (ADR-G2)

`EdgeType` (domínio documento) ganha os tipos que o MM exercita, somando aos existentes
(`RESISTS`, `RELATES_TO`, `MENTIONS`, `REQUIRES`):

- `IMMUNE_TO` — imune a (dano/condição)
- `VULNERABLE_TO` — vulnerável a (dano)
- `INHABITS` — habita (local/plano)
- `SUMMONS` — invoca (outra criatura)

`NodeType` **não muda**. Schema do grafo:

```
ENTITY(criatura) --RESISTS/IMMUNE_TO/VULNERABLE_TO--> CONCEPT(fogo, frio, veneno, atordoado, ...)
ENTITY(criatura) --INHABITS--> CONCEPT(Subterrâneo, plano, ...)
ENTITY(criatura) --SUMMONS--> ENTITY(criatura invocada)
```

ADR-G2 registra o schema, a justificativa do recorte MM-focado, e a **limitação probabilística
declarada**: a extração documental tem erro; é por isso que o benchmark mede qualidade em vez
de assumir. ADR commitado antes da implementação (regra do CONTRIBUTING).

## Fluxo de dados

`build(source=MM.pdf)`:
1. `pdf.load(source)` → páginas com texto e spans (fonte/tamanho por linha) + metadados (livro, página).
2. `chunk.by_creature(pages)` → chunks por verbete (cabeçalho CAIXA ALTA/fonte; fallback janela de página).
3. Para cada chunk: `llm.extract(chunk)` → resposta estruturada (Haiku, `messages.parse`,
   **system prompt cacheado** entre chunks para baratear), capturando `usage`.
4. `map(extracted)` → `Node`/`Edge`, com **dedup** (mesma criatura/conceito = um nó por id normalizado).
5. Persistir via `NetworkXStore.save(path)`.

## Contrato de saída do LLM (schema.py)

Pydantic, alinhado às restrições de structured outputs (sem recursão, `additionalProperties:false`):

```
ExtractedEntity:  name: str   kind: "creature" | "concept"   attrs: dict (CR, tipo, alinhamento ...)
ExtractedRelation: subject: str   predicate: RESISTS|IMMUNE_TO|VULNERABLE_TO|INHABITS|SUMMONS   object: str
ExtractionResult:  entities: list[ExtractedEntity]   relations: list[ExtractedRelation]
```

O mapeamento `ExtractionResult -> (Node[], Edge[])` é lógica pura, testável sem rede.

## Estratégia de teste (TDD)

Mock só na fronteira da API (não-determinístico e pago — exceção justificada do TDD skill).
Toda a lógica de domínio é testada com código real.

- **pdf.py** — fixture PDF mínimo gerado no teste (pymupdf) → asserta páginas, texto, metadados.
- **chunk.py** — fixtures de texto sintético com cabeçalhos → asserta limites por verbete e o fallback.
- **schema.py / mapeamento** — `ExtractionResult` canônico → `Node`/`Edge` esperados + dedup. Puro.
- **extractor.py** — injeta um *fake LLM* (devolve `ExtractionResult` fixo) → testa a orquestração
  pdf→chunk→map sem rede.
- **cost.py** — função pura (tokens, preço) → custo/latência.
- **llm.py (adapter real)** — um smoke test ao vivo marcado `@pytest.mark.live`, pulado por
  padrão (roda só com `ANTHROPIC_API_KEY`). O run do gate P1.4 é a validação ao vivo de verdade.

CI continua verde: os testes `live` ficam fora do default; cobertura/lint/types/invariantes mantidos.

## P1.4 — gate de custo (run pago)

Roda o pipeline no Monster Manual inteiro e mede:
- **Custo:** Σ(input_tokens × $1/1M + output_tokens × $5/1M), lido de `response.usage`.
- **Latência:** tempo total e por chunk.
- **Volume:** nº de entidades e relações extraídas.
- **Qualidade:** amostra ~10 criaturas para conferência manual das relações.

Persiste o grafo resultante. **Estimativa: ~US$1–3** (≈290K tokens de texto + overhead de prompt;
output estruturado por criatura). **Pré-condições do run:** `ANTHROPIC_API_KEY` exportada
(hoje não está) e OK explícito do usuário antes da chamada paga. Gate de disciplina: **não
escalar** para PHB/DMG nesta sessão (isso é P1.5).

## Dependências e empacotamento

Extra opcional `glyph-kg[document]` = `pymupdf`, `anthropic` — mantém a lib base leve (valor de
DX do ADR-G1). Env vars: `ANTHROPIC_API_KEY` (obrigatória para extração/gate),
`GLYPH_LLM_PROVIDER=anthropic` (já no README).

## Critérios de done da Fase 1 (esta sessão)

- ADR-G2 commitado antes da implementação.
- `EdgeType` estendido; modelo valida; round-trip mantido.
- `glyph/extract/document/` implementado com TDD; suíte verde; invariantes de arquitetura intactos.
- Smoke test `live` passa com a key (executado uma vez).
- Gate P1.4 rodado no MM com OK do usuário: custo/latência/volume reportados, ~10 criaturas
  amostradas, grafo persistido.
- CI verde (lint, types, test, coverage, invariantes).

## Fora de escopo

P1.5 (escalar para o corpus completo), PHB/DMG, retrieval (Fase 2), baseline vetorial, GNOMON,
integração AXON. Schema amplo (magia/item/regra/local) fica para quando PHB/DMG entrarem.
