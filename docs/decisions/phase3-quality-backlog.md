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

## 4. Retrieval (Fase 2) — declarar/observar no benchmark

Do review final da Fase 2. Os itens de correção mais baratos já foram resolvidos no commit de
hardening (unificação de identidade no híbrido por chave case-insensitive; desempate
determinístico por `source` no grafo). Restam para a Fase 3:

- **Superfície de embedding assimétrica:** o braço-grafo embeda só os labels de nó (nomes); o
  braço-vetor embeda o texto inteiro do chunk (bloco de stats). É por design (o valor do grafo
  está na expansão de vizinhança, não no recall da âncora), mas precisa ser **declarado no
  artigo** para a comparação não ser lida como "só representação".
- **Cobertura por budget difere entre braços:** os segmentos do grafo (`label — relações`) são
  bem mais curtos que os chunks do vetor e ignoram `Node.attrs` (CR/tipo/alinhamento extraídos
  mas não usados). Sob o mesmo budget de char, o grafo empacota mais segmentos. Não é injusto,
  mas afeta a leitura de recall@budget — declarar, e considerar incluir attrs no segmento do grafo.
- **Tokenizer real:** o budget é estimativa por char (ADR-G3). Trocar por contagem real onde pesar.
- **Demo sem cache de embeddings:** `scripts/retrieve_demo.py` reembeda todos os labels de nó a
  cada execução; lento no grafo completo com o modelo ST real. Cachear se virar uso recorrente.
