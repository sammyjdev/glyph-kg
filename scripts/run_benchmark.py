"""P3.1: run the three retrieval arms over the query set and score them with GNOMON.

Pre-computes each arm's grounded answer per question (real tokens + latency), then
judges every case with an OpenAI-compatible OSS judge (Groq by default) and aggregates
with seeded bootstrap CIs. Writes a committed JSON baseline + a human METRICS.md.

Usage:
    GROQ_API_KEY=... ANTHROPIC_API_KEY=... \\
      python3 scripts/run_benchmark.py out/monster-manual.json "<Monster Manual.pdf>"

    # regression gate: fail if a fresh run drifts past tolerance from the baseline
    python3 scripts/run_benchmark.py out/monster-manual.json "<book.pdf>" --check

Requires: pip install -e ".[document,retrieval,embeddings,eval]"
"""

import argparse
import json
import os
import sys
from pathlib import Path

from glyph.baseline.vector import VectorBaseline
from glyph.embed.sentence_transformers_embedder import SentenceTransformerEmbedder
from glyph.eval.dataset import load_eval_cases
from glyph.eval.generate import AnswerGenerator, AnthropicGenerator
from glyph.eval.harness import run_benchmark
from glyph.eval.judge import OpenAICompatJudge
from glyph.eval.report import DEFAULT_TOLERANCE, regression_check, render_markdown, to_dict
from glyph.eval.response import ArmResponse
from glyph.extract.document import chunk, pdf
from glyph.retrieval.graph import GraphRetriever
from glyph.retrieval.hybrid import HybridRetriever
from glyph.retrieval.port import Retriever
from glyph.store.networkx_store import NetworkXStore

DEFAULT_BASELINE = "eval/benchmark-baseline.json"
DEFAULT_METRICS = "METRICS.md"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _precompute(retriever: Retriever, cases, generator) -> dict[str, ArmResponse]:  # type: ignore[no-untyped-def]
    answerer = AnswerGenerator(retriever, generator)
    out: dict[str, ArmResponse] = {}
    for case in cases:
        out[case.id] = answerer.answer(case.question)
    return out


def _build_arms(graph_path: str, book_path: str):  # type: ignore[no-untyped-def]
    store = NetworkXStore.load(graph_path)
    payload = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    nodes = store.subgraph([n["id"] for n in payload["nodes"]], hops=0).nodes
    documents = [
        (piece.label, piece.text)
        for piece in chunk.by_creature(pdf.load(book_path))
        if chunk.is_creature(piece)
    ]
    embedder = SentenceTransformerEmbedder()
    graph = GraphRetriever(store=store, embedder=embedder, nodes=nodes)
    vector = VectorBaseline(embedder=embedder)
    vector.index(documents)
    hybrid = HybridRetriever(graph, vector)
    return {"graph": graph, "vector": vector, "hybrid": hybrid}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="GLYPH GraphRAG vs vector benchmark")
    parser.add_argument("graph", help="persisted KG json (e.g. out/monster-manual.json)")
    parser.add_argument("book", help="source PDF for the vector arm's chunk texts")
    parser.add_argument("--queries", default="eval/queries.json")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="judge model (OpenAI-compatible)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge-runs", type=int, default=3)
    parser.add_argument("--out", default=DEFAULT_BASELINE)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--check", action="store_true", help="fail if drift exceeds tolerance")
    args = parser.parse_args(argv)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set", file=sys.stderr)
        return 2
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set (needed for generation)", file=sys.stderr)
        return 2

    cases = load_eval_cases(args.queries)
    arms = _build_arms(args.graph, args.book)
    generator = AnthropicGenerator()
    responses_by_arm = {arm: _precompute(r, cases, generator) for arm, r in arms.items()}

    judge = OpenAICompatJudge(model=args.model, api_key=api_key)
    report = run_benchmark(
        cases,
        responses_by_arm,
        judge,
        judge_model=args.model,
        seed=args.seed,
        judge_runs=args.judge_runs,
    )
    fresh = to_dict(report)

    if args.check:
        baseline_path = Path(args.out)
        if not baseline_path.exists():
            print(f"no committed baseline at {baseline_path}; run without --check first")
            return 2
        committed = json.loads(baseline_path.read_text(encoding="utf-8"))
        violations = regression_check(committed, fresh, tolerance=args.tolerance)
        if violations:
            print("REGRESSION:\n  " + "\n  ".join(violations), file=sys.stderr)
            return 1
        print("benchmark within tolerance of the committed baseline")
        return 0

    Path(args.out).write_text(json.dumps(fresh, ensure_ascii=False, indent=2) + "\n", "utf-8")
    Path(args.metrics).write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {args.metrics} (n={report.n_cases}, judge={args.model})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
