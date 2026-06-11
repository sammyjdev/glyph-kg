# Backlog de qualidade da extração — revisitar na Fase 3

Itens de qualidade observados no gate P1.4 (Monster Manual, ver
`phase1-cost-gate-results.md`). Não bloqueiam a Fase 2; entram na avaliação da Fase 3,
onde a qualidade da extração documental é medida (não assumida).

## 1. Normalização de casing dos labels de nó

Os labels vêm do nome devolvido pelo LLM, em caixas inconsistentes (`ANKHEG`, `abolete`,
`Deva`). Os **ids já são normalizados** (lowercase + espaços colapsados), então a
deduplicação funciona — o problema é só cosmético no `label`.

**Sugestão:** normalizar o `label` na exibição/persistência (ex.: title-case com exceções
para artigos/preposições, ou preservar a forma mais frequente vista). Avaliar se afeta a
relevância medida ou só a apresentação.

## 2. Erros probabilísticos de relação

A extração é probabilística e tem erro — é por isso que o benchmark mede. Exemplo do gate:

- **ANKHEG: `resists ácido`** — provável confusão entre o *ataque* de ácido da criatura e
  uma *resistência* a ácido.

**Sugestão (Fase 3):** incluir no query set casos onde o texto distingue ataque vs
resistência/imunidade, e reportar a taxa de erro desse tipo como parte do número honesto
(P3.5). Considerar um passo de verificação/few-shot mais forte no prompt se a taxa pesar.

## 3. (a observar) Cabeçalhos de seção em fonte de criatura

O filtro `is_creature` (bloco de atributos) já removeu seções de regras/índice. Se um livro
futuro (PHB/DMG) não tiver bloco de atributos, esse sinal não serve — schema e filtro
próprios serão necessários (P1.5/fase futura).
