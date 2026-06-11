# P1.4 — Resultados do gate de custo (Monster Manual)

**Data:** 2026-06-10
**Livro:** Monster Manual (PT-BR), 351 páginas
**Modelo:** Claude Haiku 4.5 (`claude-haiku-4-5`), extração estruturada via `messages.parse`
**Filtro:** `chunk.is_creature` — só chunks com bloco de atributos (FOR/DES/CON/INT/SAB/CAR);
front-matter, regras e índice descartados (767 chunks → 425 criaturas).

## Números medidos

| Métrica | Valor |
|---|---|
| Chunks (criaturas) | 425 |
| Nós | 693 (458 `ENTITY` / 235 `CONCEPT`) |
| Arestas | 1.305 |
| Tokens input | 739.649 |
| Tokens output | 94.258 |
| **Custo** | **US$ 1,2109** ($1/M input + $5/M output) |
| Latência | 1.048,4 s (~17,5 min), 2,47 s/chunk |

Distribuição de arestas: `immune_to` 768, `resists` 395, `vulnerable_to` 47, `summons` 49,
`inhabits` 46. Os cinco tipos de relação documentais do ADR-G2 são exercitados.

O custo veio **abaixo da estimativa** (~US$1,50): a saída estruturada é compacta (94K tokens
de output, não os ~255K estimados).

## Qualidade (amostragem manual)

Relações majoritariamente corretas:
- Deva/Planetário/Solar (anjos): `resists radiante/concussão/perfurante/cortante`,
  `immune_to enfeitiçado` — correto.
- Aparição/espectro (mortos-vivos): `resists ácido/elétrico/fogo/frio/trovejante` — correto.
- Aarakocra: sem relações — correto.

Erro probabilístico observado (esperado, é o que o benchmark mede):
- ANKHEG: `resists ácido` — provável confusão entre *ataque* de ácido e *resistência* a ácido.

Limitação cosmética: os labels dos nós vêm com caixa inconsistente (`ANKHEG`, `abolete`,
`Deva`) porque o LLM devolve nomes em caixas variadas. Os ids são normalizados (lowercase),
então a deduplicação funciona; só os labels não estão uniformes. Candidato a normalização de
label numa próxima iteração.

## Conclusão

Gate aprovado: US$ 1,21 para o bestiário completo, bem dentro do orçamento; grafo denso em
relações e com qualidade suficiente para a Fase 2 (retrieval + baseline vetorial). **Não
escalar** para PHB/DMG nesta fase (P1.5) — esses livros pedem schema próprio.

Reprodução: `python3 scripts/extract_book.py "<Monster Manual.pdf>" out/monster-manual.json`
com `ANTHROPIC_API_KEY` setada. Grafo persistido em `out/monster-manual.json`.
