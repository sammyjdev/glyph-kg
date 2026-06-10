# ADR-G2: Schema de extração documental (Monster Manual)

**Data:** 2026-06-09
**Status:** Aceito

## Contexto

A Fase 1 constrói o primeiro knowledge graph documental do GLYPH a partir do
Monster Manual (PT-BR, 351 páginas, texto extraível, sem TOC). A extração é
probabilística: um LLM lê prosa em português e infere entidades e relações, com
erro. O `EdgeType` do Phase 0 cobre apenas `{RELATES_TO, MENTIONS, REQUIRES,
RESISTS}` para documento — insuficiente para as relações que o Monster Manual
exercita de fato.

## Decisão

**Schema MM-focado.** Entidade = criatura (`NodeType.ENTITY`); tipo de dano,
condição e local/plano = conceito (`NodeType.CONCEPT`). `EdgeType` ganha quatro
membros de domínio documento:

- `IMMUNE_TO` — imunidade a dano/condição
- `VULNERABLE_TO` — vulnerabilidade a dano
- `INHABITS` — habita local/plano
- `SUMMONS` — invoca/conjura outra criatura

somando aos existentes `RESISTS`, `RELATES_TO`, `MENTIONS`, `REQUIRES`.
`NodeType` não muda.

Schema do grafo:

```
ENTITY(criatura) --RESISTS/IMMUNE_TO/VULNERABLE_TO--> CONCEPT(fogo, frio, atordoado, ...)
ENTITY(criatura) --INHABITS--> CONCEPT(Subterrâneo, plano, ...)
ENTITY(criatura) --SUMMONS--> ENTITY(criatura invocada)
```

## Consequências

**Positivas:** schema casa com a estrutura do MM, alto sinal para o benchmark
grafo-vs-vetor. Adicionar um domínio novo (PHB/DMG) é estender o enum e o prompt,
sem tocar o núcleo.

**Trade-offs / a observar:** a extração documental tem erro — é por isso que o
benchmark mede qualidade em vez de assumir. O schema não cobre magia/item/regra
(conteúdo de PHB/DMG); isso é declarado e fica para fase futura.

## Alternativas consideradas

| Alternativa | Por que foi descartada |
|---|---|
| Reusar só os EdgeType existentes | Funde imune/resiste/vulnerável numa aresta só; perde fidelidade |
| Schema amplo (magia/item/regra/local) já | MM não exercita esses tipos; extração ruidosa, mais cara, sem ganho agora |
| Dict de atributos livre na saída do LLM | Structured outputs exigem `additionalProperties:false`; campos fixos opcionais no lugar |
