# ADR-G1: Extractor port com dois adapters e backend NetworkX/Neo4j

**Data:** 2026-06-09
**Status:** Aceito

## Contexto

GLYPH precisa construir knowledge graph de dois domínios com naturezas de extração opostas:

- **Documento**: extração probabilística. Um LLM lê prosa e infere entidades e relações, com erro. Ex: "Goblin resiste a fogo" é uma relação inferida, não garantida.
- **Código**: extração determinística. tree-sitter produz a AST e a relação `A CALLS B` é fato, não inferência.

Forçar os dois num único extractor genérico, ou num schema único de grafo que sirva aos dois, produz uma abstração que não é boa em nenhum domínio. A diferença não é cosmética: é a confiabilidade da aresta.

Em paralelo, a lib precisa ser pip-installable e ter bom DX. Um backend de grafo que exija servidor (Neo4j) é fricção alta para uma biblioteca. Mas "Neo4j" é keyword que recruiter e busca de vaga escaneiam, e a história de produção tem valor.

## Decisão

**Extractor port com dois adapters.** Definimos um `Extractor` protocol com `extract(source) -> (nodes, edges)`. `DocumentExtractor` (LLM) e `CodeExtractor` (tree-sitter) o implementam. O núcleo de grafo, o store e o retrieval são compartilhados; só a extração é específica do domínio.

**GraphStore port com NetworkX default e Neo4j adapter.** NetworkX é o backend default: embedded, pip-installable, zero servidor, suficiente para a escala dos corpora-alvo (documentos cabem em memória; repos também). Neo4j existe como adapter smoke-tested, pelo keyword e pela história de produção, não como serviço always-on.

## Consequências

**Positivas:**
- A fronteira certa fica explícita: o que é comum (grafo, store, retrieval, medição) vs o que é específico de domínio (extração). Demonstra abstração arquitetural, não só uso de lib.
- Adicionar um terceiro domínio futuro é um novo adapter de extractor, sem tocar o núcleo.
- DX de biblioteca preservado pelo NetworkX default; keyword Neo4j obtido honestamente porque o adapter roda de verdade.

**Negativas / trade-offs:**
- Manter dois adapters de extração e dois de store é mais superfície de teste.
- NetworkX não escala para grafos muito grandes; se um corpus futuro estourar memória, o Neo4j adapter (ou outro) absorve, mas isso é trabalho adicional.

**Neutras / a observar:**
- A assimetria probabilístico/determinístico fica visível no benchmark: a qualidade da extração documental é medida, a do código é assumida correta dentro da limitação de resolução de símbolo.

## Alternativas consideradas

| Alternativa | Por que foi descartada |
|---|---|
| Extractor único genérico | Os domínios têm confiabilidade de aresta oposta; um extractor só não serve a nenhum bem |
| Schema de grafo único para os dois | Produz grafo medíocre nos dois domínios; reviewer técnico detecta |
| Neo4j como default | Exige servidor, péssimo DX para lib pip-installable |
| kuzu como default | openCypher embedded é atraente, mas NetworkX tem DX e maturidade melhores para a escala-alvo; kuzu fica como opção futura se a escala exigir |
| Projetos separados (um doc, um code) | Perde o claim de "biblioteca de KG"; duplica núcleo de grafo, store e medição |
