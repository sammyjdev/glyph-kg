"""P3.1: load the frozen query set as eval cases."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Query:
    id: str
    question: str
    reference: str
    reference_contexts: list[str]


def _expected_answer(query: dict[str, Any]) -> str:
    answer_key = query.get("answer_key")
    if isinstance(answer_key, list) and answer_key:
        return "; ".join(str(x) for x in answer_key)
    if isinstance(answer_key, str) and answer_key.strip():
        return answer_key
    return "; ".join(query["relevant_labels"])


def load_eval_cases(path: str | Path) -> list[Query]:
    """Build one Query per query in ``eval/queries.json``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Query(
            id=query["id"],
            question=query["question"],
            reference=_expected_answer(query),
            reference_contexts=query["relevant_labels"],
        )
        for query in payload["queries"]
    ]
