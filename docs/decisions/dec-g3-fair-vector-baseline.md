# ADR-G3: Baseline vetorial justo e contrato de saída

**Data:** 2026-06-10
**Status:** Aceito

## Contexto

A tese do GLYPH é que retrieval graph-aware supera vector-only em queries que dependem de
relações entre entidades. O experimento só é válido se o baseline vetorial for forte e justo
— enfraquecê-lo invalida o número.

## Decisão

**Mesmo corpus:** o baseline vetorial indexa o texto dos mesmos chunks por criatura
(`chunk.by_creature` + `is_creature`) que geraram o grafo. O braço-grafo é a extração
estruturada desses chunks; o vetor é o texto cru deles embedado.

**Mesmo embedder e mesmo budget:** os dois braços usam o mesmo embedder local
(sentence-transformers multilingual) e cortam a saída no mesmo budget de token. O braço híbrido
funde os dois sob o mesmo budget.

**Contrato único:** `Segment`/`ContextPack` idêntico nos três modos, comparável token-a-token.

**Limitação declarada:** o budget é medido por estimativa de char nesta fase (não tokenizer
real). Declarado aqui e no benchmark (Fase 3).

## Consequências

**Positivas:** comparação controlada (mesma fonte, mesmo embedder, mesmo budget). O baseline é
implementação real (chunk + embedding + vector store + top-k), não espantalho.

**Trade-offs / a observar:** a estimativa de token por char é aproximada; a Fase 3 troca por
contagem real onde pesar. A fusão híbrida (reciprocal rank fusion) trata segmentos de grafo
(source = id de nó) e de vetor (source = label de chunk) como fontes distintas; unificar a
identidade entre representações fica para depois se o benchmark pedir.
