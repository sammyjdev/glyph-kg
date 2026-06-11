"""P3.1: load the frozen query set as GNOMON EvalCases.

GNOMON's EvalCase requires non-empty ``expected_answer`` and ``expected_contexts``.
The v1 judge is reference-free (it ignores both), but the schema still demands them,
so we fill them from the KG-joined query set: contexts from the relevant labels, and
the answer from the answer_key (relation targets / attribute value) or, when the query
has none, the relevant labels themselves. The GNOMON import stays lazy.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gnomon.domain.models import EvalCase


def _expected_answer(query: dict[str, Any]) -> str:
    answer_key = query.get("answer_key")
    if isinstance(answer_key, list) and answer_key:
        return "; ".join(str(x) for x in answer_key)
    if isinstance(answer_key, str) and answer_key.strip():
        return answer_key
    # relational queries (answer_key null) and open descriptions: the relevant
    # labels are the meaningful reference answer.
    return "; ".join(query["relevant_labels"])


def load_eval_cases(path: str | Path) -> list["EvalCase"]:
    """Build one EvalCase per query in ``eval/queries.json``."""
    from gnomon.domain.models import EvalCase

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=query["id"],
            question=query["question"],
            expected_answer=_expected_answer(query),
            expected_contexts=query["relevant_labels"],
        )
        for query in payload["queries"]
    ]
