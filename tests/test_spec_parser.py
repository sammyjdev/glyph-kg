"""Tests for spec_parser: Story ID extraction and TCM stub formatting."""

from glyph.extract.spec_parser import format_tcm_stub, parse_stories


def test_parse_stories_from_headings() -> None:
    spec = """\
## It must validate input
Some detail.
### It should log errors
More detail.
## Architecture overview
No requirement keyword here.
"""
    stories = parse_stories(spec)
    assert len(stories) == 2
    assert stories[0]["id"] == "S-01"
    assert "must validate input" in stories[0]["requirement"].lower()
    assert stories[1]["id"] == "S-02"
    assert "should log errors" in stories[1]["requirement"].lower()


def test_parse_stories_from_bullets() -> None:
    spec = """\
## Overview

- The system must handle retries.
- The system shall expose a health endpoint.
- Plain bullet without keyword.
- When the user logs in, a session is created.
- Another plain item.
"""
    stories = parse_stories(spec)
    # "must", "shall", "when" match; plain bullets skip
    assert len(stories) == 3
    assert stories[0]["id"] == "S-01"
    assert "must" in stories[0]["requirement"].lower()
    assert stories[1]["id"] == "S-02"
    assert "shall" in stories[1]["requirement"].lower()
    assert stories[2]["id"] == "S-03"
    assert "when" in stories[2]["requirement"].lower()


def test_format_tcm_stub() -> None:
    stories = [
        {"id": "S-01", "requirement": "System must validate input"},
        {"id": "S-02", "requirement": "System should log errors"},
    ]
    table = format_tcm_stub(stories)

    assert "| Story ID |" in table
    assert "| S-01" in table
    assert "| S-02" in table
    assert "✗ unmet" in table
    # Both rows present
    lines = [line for line in table.splitlines() if line.startswith("| S-")]
    assert len(lines) == 2
