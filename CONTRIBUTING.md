# Contribuindo com o GLYPH

## Metodologia

GLYPH segue Spec-Driven Development: a spec e os quality gates são definidos antes da implementação. Agentes executam sob direção arquitetural; decisões arquiteturais novas viram ADR antes de virar código.

## Fluxo

1. Pegue uma sub-task do `docs/GLYPH_PLAN.md` (ou do KICKOFF da fase atual).
2. Escreva o teste antes da implementação (TDD).
3. Implemente até o teste passar.
4. Garanta lint, types, cobertura e invariantes de arquitetura verdes localmente.
5. Abra PR referenciando a sub-task.

## Quality gates (enforçados no CI)

- **Testes:** `pytest` verde. Nada entra sem teste.
- **Cobertura:** gate ativo; PR que baixa cobertura falha.
- **Tipos:** `mypy` sem erro.
- **Lint:** `ruff` limpo.
- **Arquitetura:** `tests/architecture/` verifica os invariantes do `docs/ARCHITECTURE.md`. Violação de regra de import falha o build.

## Regra de ADR

Toda decisão arquitetural é registrada em `docs/decisions/` antes da implementação. Se durante a execução surgir uma decisão não prevista (escolha de lib, mudança de contrato, novo backend), pare e abra a ADR. Não decida inline no código.

Formato: ver os ADRs existentes (`dec-g1-...`). Status, Contexto, Decisão, Consequências (positivas / trade-offs / a observar), Alternativas consideradas.

## Honestidade de claim

- README e docs são fonte de verdade. Não afirme capacidade ou número que o código não entrega.
- Métrica publicada é reproduzível do repo. Sem reprodução, não publica.
- Limitações conhecidas ficam documentadas, não escondidas.

## Commits

- Mensagem no imperativo, escopo pequeno por sub-task.
- Um PR resolve uma sub-task ou um conjunto coeso.

## Merge / integração

- Branches integram na `main` por **rebase + fast-forward** — histórico linear, sem commits de merge.
  Fluxo: `git rebase main` na branch, depois `git checkout main && git merge --ff-only <branch>`.
- No GitHub, o único método habilitado é **Rebase and merge** (merge commit e squash desabilitados).
- `git config pull.rebase true` mantém os pulls lineares também.

## Estilo

- Voz ativa, prosa densa, sem fluff.
- Sem `robusto`, `escalável`, `poderoso` sem evidência concreta.
- Métrica vaga ("melhora performance") sai ou ganha número.
