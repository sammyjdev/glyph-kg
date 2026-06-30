"""Stdlib-only spec parser: extracts Story IDs from a markdown spec file.

Usage:
    python -m glyph.extract.spec_parser docs/spec.md
"""

import os
import re
import sys

# Requirement keywords — case-insensitive
_KEYWORDS = re.compile(r"\b(must|should|shall|when|given)\b", re.IGNORECASE)
# h2 or h3 headings only
_HEADING = re.compile(r"^#{2,3}\s+(.+)")
# bullet items (-, *, +)
_BULLET = re.compile(r"^\s*[-*+]\s+(.+)")


def _extract_stories(text: str) -> list[dict[str, str]]:
    requirements: list[str] = []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            content = m.group(1).strip()
            if _KEYWORDS.search(content):
                requirements.append(content)
            continue
        m = _BULLET.match(line)
        if m:
            content = m.group(1).strip()
            if _KEYWORDS.search(content):
                requirements.append(content)
    return [{"id": f"S-{i + 1:02d}", "requirement": req} for i, req in enumerate(requirements)]


def parse_stories(spec_path: str) -> list[dict[str, str]]:
    """Extract stories from a markdown spec file (or raw markdown text).

    Reads h2/h3 headings and bullet items that contain requirement keywords
    (must, should, shall, when, given), numbers them sequentially as S-01,
    S-02, ... and returns a list of dicts with keys 'id' and 'requirement'.

    Args:
        spec_path: Path to a markdown file, or raw markdown text (if the
                   string is not an existing file path it is treated as text).
    """
    if os.path.exists(spec_path):
        with open(spec_path) as f:
            text = f.read()
    else:
        text = spec_path
    return _extract_stories(text)


def format_tcm_stub(stories: list[dict[str, str]]) -> str:
    """Format a list of stories as a markdown TCM table stub."""
    lines = [
        "| Story ID | Requirement | Test IDs | Status |",
        "|----------|-------------|----------|--------|",
    ]
    for s in stories:
        lines.append(f"| {s['id']}     | {s['requirement']} |          | ✗ unmet |")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m glyph.extract.spec_parser <spec.md>", file=sys.stderr)
        sys.exit(1)
    _stories = parse_stories(sys.argv[1])
    print(format_tcm_stub(_stories))
