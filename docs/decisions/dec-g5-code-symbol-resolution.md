# ADR-G5: Resolução de símbolo no code extractor

**Data:** 2026-06-11
**Status:** Aceito

## Contexto

A Fase 4 adiciona o segundo extractor: código, via tree-sitter (determinístico), atrás do mesmo
`Extractor` port do documental. O parsing do AST é exato; o que é heurístico é **ligar um uso ao
símbolo que ele referencia** (chamada → função, `extends` → classe) através de arquivos.

## Decisão

**Linguagens (P4.2):** Python e Java (o set indexado do AXON). TS fica como extensão futura — o
`Grammar` é plugável (node-types + extratores de nome por linguagem), então adicionar uma é
localizado.

**Nós e arestas.** `FILE` (id = path relativo posix), `CLASS`/`FUNCTION` (id qualificado por escopo,
ex. `glyph/eval/cost.py::ArmResponse.total`), `MODULE` (alvo de import). Arestas: `DEFINES`
(escopo → símbolo), `IMPORTS` (file → module), `CALLS` (função → função), `INHERITS` (classe →
classe). Tudo deduplicado.

**Resolução por nome não-qualificado, único (a limitação declarada).** O extractor constrói um índice
`nome simples → ids definidos no corpus` e só emite `CALLS`/`INHERITS` quando o nome do alvo é
definido **exatamente uma vez**. Consequências:
- Nome ambíguo (definido em 2+ lugares) → aresta **omitida** (under-approximation, não inventamos a
  mais provável).
- Nome não definido no corpus (stdlib, lib externa, método herdado de fora) → aresta omitida.
- Chamadas via subscript/expressão (`fns[0]()`) → sem nome simples → omitidas.
- Sem type inference: `obj.metodo()` resolve pelo **nome do método** (`metodo`), não pelo tipo de
  `obj`. Em corpora com um nome de método único isso acerta; com colisão, omite.

**GLYPH como fonte canônica (P4.1).** O AXON não está no escopo deste repo; o GLYPH implementa o code
extractor de forma standalone e passa a ser a fonte canônica de code-graph. Quando o AXON integrar
(Fase 5), ele delega para cá em vez de reimplementar — alinhado ao ADR-102/103 do AXON.

**Benchmark de código (P4.4).** O harness da Fase 3, os retrievers (graph/vector) e o contrato
`ContextPack` são **agnósticos de domínio**: operam sobre qualquer grafo + quaisquer textos de chunk.
Rodar o mesmo benchmark no domínio código é apontar o harness para um code-graph + um query set de
código — nenhum código novo de avaliação é necessário, só o corpus e as queries.

## Consequências

Determinístico e reproduzível (mesmo SHA → mesmo grafo). Precisão alta, recall limitado por design na
resolução cross-símbolo — declarado, não assumido, e o benchmark mede onde isso pesa. Caminhos de
melhoria futura (não nesta fase): resolução por import graph + escopo, qualificação por tipo.
