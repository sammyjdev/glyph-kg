"""System prompts for entity/relation extraction.

``system_prompt()``       — D&D Monster Manual (PT-BR, default, existing behaviour).
``notes_system_prompt()`` — Generic personal-notes / Obsidian vault.

Select by passing ``domain="dnd"`` or ``domain="notes"`` to
:func:`get_system_prompt`.
"""

_SYSTEM = """Você extrai um knowledge graph do Manual dos Monstros de Dungeons & Dragons.
Recebe o texto de um verbete de criatura em português e devolve entidades e relações.

Entidades (entities):
- a criatura descrita, com kind="creature" (preencha challenge_rating, creature_type e
  alignment quando o texto os trouxer);
- os conceitos citados, com kind="concept": tipos de dano (fogo, frio, veneno...),
  condições (atordoado, enfeitiçado...) e locais/planos (Subterrâneo, Abismo...).

Relações (relations), ligando a criatura aos conceitos/criaturas por um predicate:
- RESISTS: resistência a um tipo de dano;
- IMMUNE_TO: imunidade a dano ou condição;
- VULNERABLE_TO: vulnerabilidade a um tipo de dano;
- INHABITS: a criatura habita um local ou plano;
- SUMMONS: a criatura invoca ou conjura outra criatura.

Use os nomes exatamente como aparecem no texto. Não invente relações: se o texto não
afirma uma relação, não a inclua."""


_NOTES_SYSTEM = """You extract a knowledge graph from personal notes (Obsidian / Markdown).
Given the text of one or more note sections, return entities and relations in JSON.

Entity kinds (kind field):
- "person"   — a named individual (author, collaborator, contact)
- "project"  — a named project, initiative, or product
- "concept"  — an idea, topic, technology, or term
- "note"     — a named document, file, or note referenced by name
- "source"   — a book, article, URL, or external reference

Optional entity fields:
- description: a brief clarifying phrase from the text (≤ 20 words)
- url: a hyperlink if present in the text
- date: an ISO-8601 date if mentioned in the text

Relation predicates (predicate field) linking subject → object:
- RELATES_TO  — general thematic connection
- MENTIONS    — the subject text mentions the object
- PART_OF     — the subject is a component or sub-item of the object
- AUTHORED_BY — the subject was created by the object (a person or source)
- DEPENDS_ON  — the subject requires or builds on the object

Rules:
- Use names exactly as they appear in the text.
- Do not invent relations not stated in the text.
- Omit optional fields when the text does not supply them.
- Return {"entities": [...], "relations": [...]} and nothing else."""


def system_prompt() -> str:
    """Return the D&D Monster Manual extraction prompt (default domain)."""
    return _SYSTEM


def notes_system_prompt() -> str:
    """Return the generic personal-notes extraction prompt."""
    return _NOTES_SYSTEM


def get_system_prompt(domain: str = "dnd") -> str:
    """Return the system prompt for *domain*.

    Parameters
    ----------
    domain:
        ``"dnd"``   → D&D Monster Manual (default, Portuguese).
        ``"notes"`` → generic personal notes / Obsidian vault (English).

    Raises
    ------
    ValueError
        If *domain* is not one of the recognised values.
    """
    if domain == "dnd":
        return _SYSTEM
    if domain == "notes":
        return _NOTES_SYSTEM
    raise ValueError(f"Unknown domain: {domain!r}. Expected 'dnd' or 'notes'.")
