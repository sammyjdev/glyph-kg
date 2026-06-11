"""P3.2: build the frozen benchmark query set from authored questions + the KG.

The questions are hand-authored against the Monster Manual domain. For each one we
join the GLYPH knowledge graph (``out/monster-manual.json``) to attach a *candidate*
relevance oracle (`relevant_sources`) and an answer key. The oracle is KG-derived,
so it inherits the extraction's known probabilistic errors (see
``docs/decisions/phase1-cost-gate-results.md`` — e.g. ANKHEG ``resists ácido``).
It is a starting point for ``context_precision``, NOT verified gold; the honest
report (P3.5) declares this. Verify against the source text before publishing.

Usage:
    python3 scripts/build_query_set.py            # writes eval/queries.json
    python3 scripts/build_query_set.py --check    # fail if the committed file is stale
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "out" / "monster-manual.json"
OUT_PATH = ROOT / "eval" / "queries.json"


class Graph:
    """Thin read view over the persisted KG for query-set joins."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._label = {n["id"]: n["label"] for n in payload["nodes"]}
        self._attrs = {n["id"]: n.get("attrs", {}) for n in payload["nodes"]}
        self._edges = payload["edges"]

    def label(self, node_id: str) -> str:
        return self._label.get(node_id, node_id)

    def attr(self, src: str, key: str) -> str | None:
        return self._attrs.get(src, {}).get(key)

    def sources_with(self, edge_type: str, targets: Sequence[str]) -> set[str]:
        """Entities that have an edge of ``edge_type`` to any of ``targets``."""
        wanted = set(targets)
        return {e["src"] for e in self._edges if e["type"] == edge_type and e["dst"] in wanted}

    def targets_of(self, src: str, edge_type: str) -> list[str]:
        """Targets reached from ``src`` via ``edge_type`` (entity-relation lookups)."""
        return [e["dst"] for e in self._edges if e["src"] == src and e["type"] == edge_type]


# Predicate = (graph) -> set of relevant node ids. Authored per question.
def relation(edge_type: str, *targets: str) -> Callable[[Graph], set[str]]:
    return lambda g: g.sources_with(edge_type, targets)


def conjunction(*clauses: Callable[[Graph], set[str]]) -> Callable[[Graph], set[str]]:
    def predicate(g: Graph) -> set[str]:
        sets = [clause(g) for clause in clauses]
        out = sets[0]
        for s in sets[1:]:
            out = out & s
        return out

    return predicate


def entity(node_id: str) -> Callable[[Graph], set[str]]:
    return lambda g: {node_id}


# Each spec: id, question (PT-BR), category, graph_favored hypothesis, predicate,
# and an answer_key resolver (human-readable expected answer).
_SPECS: list[dict[str, Any]] = [
    # --- relational_single: one relation hop; graph favored on precision ---
    {
        "id": "rel-fogo-resist",
        "category": "relational_single",
        "graph_favored": True,
        "question": "Quais criaturas resistem a dano de fogo?",
        "predicate": relation("resists", "fogo"),
        "answer": "labels",
    },
    {
        "id": "rel-acido-resist",
        "category": "relational_single",
        "graph_favored": True,
        "question": "Quais criaturas resistem a dano de ácido?",
        "predicate": relation("resists", "ácido"),
        "answer": "labels",
    },
    {
        "id": "rel-eletrico-resist",
        "category": "relational_single",
        "graph_favored": True,
        "question": "Quais criaturas resistem a dano elétrico?",
        "predicate": relation("resists", "elétrico"),
        "answer": "labels",
    },
    {
        "id": "rel-fogo-immune",
        "category": "relational_single",
        "graph_favored": True,
        "question": "Quais criaturas são imunes a dano de fogo?",
        "predicate": relation("immune_to", "fogo"),
        "answer": "labels",
    },
    {
        "id": "rel-veneno-immune",
        "category": "relational_single",
        "graph_favored": True,
        "question": "Quais criaturas são imunes a dano de veneno?",
        "predicate": relation("immune_to", "veneno"),
        "answer": "labels",
    },
    {
        "id": "rel-petrificado-immune",
        "category": "relational_single",
        "graph_favored": True,
        "question": "Quais criaturas são imunes à condição petrificado?",
        "predicate": relation("immune_to", "petrificado"),
        "answer": "labels",
    },
    {
        "id": "rel-fogo-vuln",
        "category": "relational_single",
        "graph_favored": True,
        "question": "Quais criaturas são vulneráveis a dano de fogo?",
        "predicate": relation("vulnerable_to", "fogo"),
        "answer": "labels",
    },
    {
        "id": "rel-luzsolar-vuln",
        "category": "relational_single",
        "graph_favored": True,
        "question": "Quais criaturas são vulneráveis à luz solar?",
        "predicate": relation("vulnerable_to", "luz solar"),
        "answer": "labels",
    },
    {
        "id": "rel-subterraneo-inhabit",
        "category": "relational_single",
        "graph_favored": True,
        "question": "Quais criaturas habitam o subterrâneo?",
        "predicate": relation("inhabits", "subterrâneo"),
        "answer": "labels",
    },
    # --- relational_multi: intersections; graph strongly favored ---
    {
        "id": "mul-fogo-frio-resist",
        "category": "relational_multi",
        "graph_favored": True,
        "question": "Quais criaturas resistem tanto a dano de fogo quanto a dano de frio?",
        "predicate": conjunction(relation("resists", "fogo"), relation("resists", "frio")),
        "answer": "labels",
    },
    {
        "id": "mul-fisico-resist",
        "category": "relational_multi",
        "graph_favored": True,
        "question": "Quais criaturas resistem a dano perfurante, cortante e de concussão?",
        "predicate": conjunction(
            relation("resists", "perfurante"),
            relation("resists", "cortante"),
            relation("resists", "concussão"),
        ),
        "answer": "labels",
    },
    {
        "id": "mul-fogovuln-venenoimmune",
        "category": "relational_multi",
        "graph_favored": True,
        "question": "Quais criaturas são vulneráveis a fogo e ao mesmo tempo imunes a veneno?",
        "predicate": conjunction(
            relation("vulnerable_to", "fogo"), relation("immune_to", "veneno", "envenenado")
        ),
        "answer": "labels",
    },
    # --- entity_relation: single entity, relation lookup; graph favored ---
    {
        "id": "ent-dragaofada-summons",
        "category": "entity_relation",
        "graph_favored": True,
        "question": "Quais magias o dragão-fada consegue conjurar?",
        "predicate": entity("dragão-fada"),
        "answer": ("targets", "dragão-fada", "summons"),
    },
    {
        "id": "ent-efreeti-summons",
        "category": "entity_relation",
        "graph_favored": True,
        "question": "Que elemental o efreeti consegue invocar?",
        "predicate": entity("efreeti"),
        "answer": ("targets", "efreeti", "summons"),
    },
    {
        "id": "ent-drow-inhabit",
        "category": "entity_relation",
        "graph_favored": True,
        "question": "Onde o drow habita?",
        "predicate": entity("drow"),
        "answer": ("targets", "drow", "inhabits"),
    },
    {
        "id": "ent-ente-summons",
        "category": "entity_relation",
        "graph_favored": True,
        "question": "O que o ente consegue invocar?",
        "predicate": entity("ente"),
        "answer": ("targets", "ente", "summons"),
    },
    {
        "id": "ent-mumia-defesas",
        "category": "entity_relation",
        "graph_favored": True,
        "question": "Quais são as resistências e imunidades da múmia?",
        "predicate": entity("múmia"),
        "answer": ("targets", "múmia", "resists"),
    },
    # Known probabilistic error (P1.4): ANKHEG resists ácido is likely wrong. Kept on
    # purpose — the benchmark must expose where the graph carries extraction noise.
    {
        "id": "ent-ankheg-resist",
        "category": "entity_relation",
        "graph_favored": True,
        "question": "A quais tipos de dano o ankheg resiste?",
        "predicate": entity("ankheg"),
        "answer": ("targets", "ankheg", "resists"),
        "note": "KG carries a likely-wrong 'resists ácido' edge (see P1.4 cost-gate notes).",
    },
    # --- factual_attribute: attrs live in node.attrs, not in graph segments; vector favored ---
    {
        "id": "fac-aarakocra-cr",
        "category": "factual_attribute",
        "graph_favored": False,
        "question": "Qual é o nível de desafio (CR) do aarakocra?",
        "predicate": entity("aarakocra"),
        "answer": ("attr", "aarakocra", "challenge_rating"),
    },
    {
        "id": "fac-balor-tipo",
        "category": "factual_attribute",
        "graph_favored": False,
        "question": "Que tipo de criatura é o balor?",
        "predicate": entity("balor"),
        "answer": ("attr", "balor", "creature_type"),
    },
    {
        "id": "fac-aarakocra-alinhamento",
        "category": "factual_attribute",
        "graph_favored": False,
        "question": "Qual é o alinhamento do aarakocra?",
        "predicate": entity("aarakocra"),
        "answer": ("attr", "aarakocra", "alignment"),
    },
    {
        "id": "fac-ente-cr",
        "category": "factual_attribute",
        "graph_favored": False,
        "question": "Qual é o nível de desafio do ente?",
        "predicate": entity("ente"),
        "answer": ("attr", "ente", "challenge_rating"),
    },
    # --- factual_description: open prose; vector favored (raw chunk text) ---
    {
        "id": "des-mephit-poeira",
        "category": "factual_description",
        "graph_favored": False,
        "question": "O que é um mephit da poeira?",
        "predicate": entity("mephit da poeira"),
        "answer": "none",
    },
    {
        "id": "des-balor",
        "category": "factual_description",
        "graph_favored": False,
        "question": "Descreva as características gerais do balor.",
        "predicate": entity("balor"),
        "answer": "none",
    },
    {
        "id": "des-ankheg",
        "category": "factual_description",
        "graph_favored": False,
        "question": "Como é a aparência e o comportamento do ankheg?",
        "predicate": entity("ankheg"),
        "answer": "none",
    },
]


def _resolve_answer(spec: dict[str, Any], g: Graph) -> Any:
    answer = spec["answer"]
    if answer == "labels":
        return None  # the relevant_sources labels already carry the answer
    if answer == "none":
        return None
    kind = answer[0]
    if kind == "targets":
        _, src, edge_type = answer
        return [g.label(t) for t in g.targets_of(src, edge_type)]
    if kind == "attr":
        _, src, key = answer
        return g.attr(src, key)
    raise ValueError(f"unknown answer kind: {answer!r}")


def build(g: Graph) -> dict[str, Any]:
    queries = []
    for spec in _SPECS:
        relevant = sorted(spec["predicate"](g))
        item: dict[str, Any] = {
            "id": spec["id"],
            "question": spec["question"],
            "category": spec["category"],
            "graph_favored": spec["graph_favored"],
            "relevant_sources": relevant,
            "relevant_labels": [g.label(r) for r in relevant],
            "answer_key": _resolve_answer(spec, g),
        }
        if "note" in spec:
            item["note"] = spec["note"]
        queries.append(item)
    return {
        "_meta": {
            "description": "Frozen benchmark query set for GLYPH Phase 3 (P3.2).",
            "corpus": "Monster Manual (PT-BR), graph at out/monster-manual.json",
            "n": len(queries),
            "categories": sorted({q["category"] for q in queries}),
            "oracle_caveat": (
                "relevant_sources is KG-derived (candidate oracle for context_precision), "
                "NOT verified gold. It inherits extraction errors (see P1.4). The P3.5 "
                "report declares this; verify against source text before publishing numbers."
            ),
            "generator": "scripts/build_query_set.py",
        },
        "queries": queries,
    }


def main(argv: list[str]) -> int:
    if not GRAPH_PATH.exists():
        print(f"missing graph: {GRAPH_PATH}", file=sys.stderr)
        return 2
    g = Graph(json.loads(GRAPH_PATH.read_text(encoding="utf-8")))
    payload = build(g)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    if "--check" in argv:
        current = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.exists() else ""
        if current != rendered:
            print(f"{OUT_PATH} is stale; re-run scripts/build_query_set.py", file=sys.stderr)
            return 1
        print(f"{OUT_PATH} up to date ({payload['_meta']['n']} queries)")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({payload['_meta']['n']} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
