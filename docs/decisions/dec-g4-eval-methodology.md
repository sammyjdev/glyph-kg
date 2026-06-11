# ADR-G4: Metodologia de avaliação (benchmark GraphRAG vs vetor)

**Data:** 2026-06-11
**Status:** Aceito

## Contexto

A Fase 3 produz o número honesto: retrieval graph-aware vs baseline vetorial vs híbrido,
medido sobre o mesmo corpus, com intervalos de confiança. O GNOMON foi auditado e é
pip-installable; este ADR fixa **como** medimos.

## Decisões

**Query set (P3.2).** 25 perguntas autorais sobre o grafo do Monster Manual, congeladas em
`eval/queries.json` e versionadas. Balanceadas por hipótese: relacionais (single/multi/
entity-relation, onde o grafo deveria ganhar) e factuais (atributo/descrição, onde o vetor
deveria bastar). Cada query traz um oráculo de relevância **derivado do KG** — candidato, não
gold verificado: herda erros de extração (ex. `ankheg → resists ácido`), mantidos de propósito
para o relatório expor ruído (P3.5). `n=25` é declarado; ampliável.

**Passo de geração.** O `faithfulness` do GNOMON nota o quão ancorada a resposta está nos
contextos, então cada braço gera uma resposta sobre o seu contexto recuperado (prompt grounded:
"responda só pelo contexto"). Sem geração não há `faithfulness`. Tokens e latência são reais.

**Igualdade de budget.** Os três braços cortam o contexto no mesmo `token_budget` e usam o mesmo
embedder local (ADR-G3). O diferencial de custo é o tamanho do contexto que cada braço entrega ao
gerador (segmentos de grafo vs chunks de vetor) — exatamente a eficiência de token que medimos.

**Judge (P3.0).** Métricas v1 do GNOMON: **`faithfulness` e `context_precision` apenas** (answer
relevance e context recall são v2, não construídas — não prometemos recall). O judge v1 é
**reference-free na prática**: o schema do `EvalCase` exige `expected_answer`/`expected_contexts`,
mas o prompt do judge os ignora; preenchemos a partir do query set para validar o schema. Usamos um
**`OpenAICompatJudge` (Groq, OSS na nuvem)** que reusa o prompt e o parse do próprio GNOMON — só o
transporte muda, então o score significa o mesmo que o do `OllamaJudge`. `seed + run` dá a sequência
determinística por seed declarada.

**Agregação e CI (P3.3).** Bypass do `run_eval`: ele descarta scores por-caso e nós os queremos
(para reportar onde o grafo **perdeu**, P3.5). Dirigimos o judge por caso, colapsamos os
`judge_runs` por média (um score por caso), e reusamos o `aggregate_metric` do GNOMON para o
**bootstrap percentil semeado** (2000 resamples, `n ≥ 2`). Mesma máquina de CI, com detalhe por-caso.

**Custo.** O GNOMON só conta tokens; o GLYPH calcula US$ a partir dos tokens **de geração**, às
taxas assimétricas do Haiku 4.5 ($1/M input, $5/M output). O judge OSS é precificado à parte e
excluído da tabela (declarado no `METRICS.md`).

**Baseline reproduzível (P3.4).** `make benchmark` regenera `eval/benchmark-baseline.json`
(commitado) + `METRICS.md`. `make benchmark-check` falha se uma corrida nova divergir além da
tolerância (default 0.05) das médias commitadas. Corpus e query set congelados.

**Relatório honesto (P3.5).** A tabela mostra cada métrica com CI para **todos** os braços —
inclusive onde o grafo não ganhou — mais tokens, custo e latência. Declara `n`, que tokens são
contagem real, e o caveat do oráculo KG-derivado.

## Consequências

Comparação controlada e reproduzível do repo. O número não é publicado sem rodar `make benchmark`
com chaves reais (`GROQ_API_KEY` + `ANTHROPIC_API_KEY`); até lá o `METRICS.md` declara que está
pendente. Limitações declaradas: oráculo KG-derivado não-gold, só duas métricas v1, judge OSS
não-determinístico dentro da variância medida.
