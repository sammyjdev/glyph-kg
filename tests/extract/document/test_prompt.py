from glyph.extract.document.prompt import system_prompt


def test_system_prompt_names_every_predicate() -> None:
    text = system_prompt()
    for predicate in ("RESISTS", "IMMUNE_TO", "VULNERABLE_TO", "INHABITS", "SUMMONS"):
        assert predicate in text


def test_system_prompt_forbids_inventing_relations() -> None:
    text = system_prompt().lower()
    assert "não invente" in text
