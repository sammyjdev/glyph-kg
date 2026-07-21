"""P3.5: build_confusion_probes ground-truth-exists guard.

scripts/build_query_set.py has no dedicated test file otherwise - it is exercised
only indirectly via tests/eval/test_query_set.py (which asserts the committed
eval/queries.json matches the generator, but never calls build_confusion_probes
directly). This closes a mutation-testing gap: removing the
`if not g.targets_of(...): raise ValueError(...)` guard survived undetected.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
GENERATOR_PATH = ROOT / "scripts" / "build_query_set.py"
GRAPH_PATH = ROOT / "out" / "monster-manual.json"


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("build_query_set", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_ground_truth_edge_raises(generator) -> None:
    # Minimal synthetic graph: no edges at all, so neither confusion-probe
    # contrast (espectro/resists, deva/immune_to) can be found.
    graph = generator.Graph({"nodes": [], "edges": []})
    with pytest.raises(ValueError, match="ent-espectro-resist"):
        generator.build_confusion_probes(graph)


def test_real_graph_produces_expected_probes(generator) -> None:
    graph = generator.Graph(json.loads(GRAPH_PATH.read_text(encoding="utf-8")))
    probes = generator.build_confusion_probes(graph)
    by_id = {p["id"]: p for p in probes}
    assert set(by_id) == {"ent-espectro-resist", "ent-deva-immune"}
    assert by_id["ent-espectro-resist"]["attack_vs_resistance_confusion"] is False
    assert by_id["ent-deva-immune"]["attack_vs_resistance_confusion"] is False
