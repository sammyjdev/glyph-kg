import pytest

from glyph.extract.document.prompt import get_system_prompt, notes_system_prompt, system_prompt


def test_system_prompt_names_every_predicate() -> None:
    text = system_prompt()
    for predicate in ("RESISTS", "IMMUNE_TO", "VULNERABLE_TO", "INHABITS", "SUMMONS"):
        assert predicate in text


def test_system_prompt_forbids_inventing_relations() -> None:
    text = system_prompt().lower()
    assert "não invente" in text


# ---------------------------------------------------------------------------
# notes_system_prompt
# ---------------------------------------------------------------------------


def test_notes_system_prompt_names_every_predicate() -> None:
    text = notes_system_prompt()
    for predicate in ("RELATES_TO", "MENTIONS", "PART_OF", "AUTHORED_BY", "DEPENDS_ON"):
        assert predicate in text


def test_notes_system_prompt_names_every_entity_kind() -> None:
    text = notes_system_prompt()
    for kind in ("person", "project", "concept", "note", "source"):
        assert kind in text


def test_notes_system_prompt_instructs_json_output() -> None:
    text = notes_system_prompt()
    assert "entities" in text
    assert "relations" in text


# ---------------------------------------------------------------------------
# get_system_prompt dispatch
# ---------------------------------------------------------------------------


def test_get_system_prompt_dnd_returns_dnd_prompt() -> None:
    assert get_system_prompt("dnd") == system_prompt()


def test_get_system_prompt_notes_returns_notes_prompt() -> None:
    assert get_system_prompt("notes") == notes_system_prompt()


def test_get_system_prompt_default_is_dnd() -> None:
    assert get_system_prompt() == system_prompt()


def test_get_system_prompt_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown domain"):
        get_system_prompt("unknown")
