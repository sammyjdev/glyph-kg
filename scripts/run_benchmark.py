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
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from glyph.baseline.vector import VectorBaseline
from glyph.embed.sentence_transformers_embedder import SentenceTransformerEmbedder
from glyph.eval.code_corpus import code_documents
from glyph.eval.dataset import load_eval_cases
from glyph.eval.generate import (
    AnswerGenerator,
    AnthropicGenerator,
    Generator,
    OpenAICompatGenerator,
)
from glyph.eval.harness import ArmReport, run_benchmark
from glyph.eval.judge import GROQ_BASE_URL, OpenAICompatJudge
from glyph.eval.report import DEFAULT_TOLERANCE, regression_check, render_markdown, to_dict
from glyph.eval.response import ArmResponse
from glyph.extract.document import chunk, pdf
from glyph.model.node import NodeType
from glyph.retrieval.community import CommunityRetriever
from glyph.retrieval.graph import GraphRetriever
from glyph.retrieval.hybrid import HybridRetriever
from glyph.retrieval.port import Retriever
from glyph.store.networkx_store import NetworkXStore

DEFAULT_BASELINE = "eval/benchmark-baseline.json"
DEFAULT_METRICS = "METRICS.md"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

_CODE_SYSTEM = (
    "You answer questions about the AXON codebase using ONLY the provided context "
    "(source files and graph relations). If the context lacks the answer, say so. Be "
    "concise and do not invent code or facts beyond the context."
)


def _precompute(
    retriever: Retriever, cases, generator: Generator, system: str | None = None
) -> dict[str, ArmResponse]:  # type: ignore[no-untyped-def]
    answerer = (
        AnswerGenerator(retriever, generator, system=system)
        if system is not None
        else AnswerGenerator(retriever, generator)
    )
    out: dict[str, ArmResponse] = {}
    for case in cases:
        out[case.id] = answerer.answer(case.question)
    return out


def _build_arms(graph_path: str, source: str, domain: str = "document"):  # type: ignore[no-untyped-def]
    store = NetworkXStore.load(graph_path)
    payload = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    nodes = store.subgraph([n["id"] for n in payload["nodes"]], hops=0).nodes
    embedder = SentenceTransformerEmbedder()

    if domain == "code-global":
        # global axis (ADR-G7): community summaries vs a fair vector baseline vs local graph
        community_nodes = [n for n in nodes if n.type is NodeType.COMMUNITY]
        local_nodes = [n for n in nodes if n.type is not NodeType.COMMUNITY]
        community = CommunityRetriever(community_nodes, embedder)
        vector = VectorBaseline(embedder=embedder)
        vector.index(code_documents(source))
        graph = GraphRetriever(store=store, embedder=embedder, nodes=local_nodes)
        return {"community": community, "vector": vector, "graph": graph}

    if domain == "code":
        documents = code_documents(source)
    else:
        documents = [
            (piece.label, piece.text)
            for piece in chunk.by_creature(pdf.load(source))
            if chunk.is_creature(piece)
        ]
    graph = GraphRetriever(store=store, embedder=embedder, nodes=nodes)
    vector = VectorBaseline(embedder=embedder)
    vector.index(documents)
    hybrid = HybridRetriever(graph, vector)
    return {"graph": graph, "vector": vector, "hybrid": hybrid}


def _fingerprint(queries_path: Path, graph_path: str, book_path: str) -> str:
    """Hash the inputs that determine the generated answers: query set, graph, book.

    The judge model/runs are NOT included — they affect scoring, not generation, so a
    cached answer set stays valid across judges. If any generation input changes, the
    fingerprint changes and the stale cache is refused instead of silently reused.
    """
    h = hashlib.sha256()
    h.update(queries_path.read_bytes())
    h.update(Path(graph_path).read_bytes())
    book = Path(book_path)
    h.update(f"{book.name}:{book.stat().st_size}".encode())
    return h.hexdigest()


def _save_answers(
    path: Path, responses_by_arm: dict[str, dict[str, ArmResponse]], fingerprint: str
) -> None:
    data = {
        "_fingerprint": fingerprint,
        "answers": {
            arm: {cid: r.model_dump() for cid, r in responses.items()}
            for arm, responses in responses_by_arm.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_answers(path: Path, fingerprint: str) -> dict[str, dict[str, ArmResponse]] | None:
    """Return the cached answers, or None if the cache is absent, stale or pre-fingerprint."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("_fingerprint") != fingerprint:
        return None
    return {
        arm: {cid: ArmResponse.model_validate(r) for cid, r in responses.items()}
        for arm, responses in data["answers"].items()
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="GLYPH GraphRAG vs vector benchmark")
    parser.add_argument("graph", help="persisted KG json (e.g. out/monster-manual.json)")
    parser.add_argument("source", help="document: source PDF; code: repo source dir")
    parser.add_argument(
        "--domain",
        choices=["document", "code", "code-global"],
        default="document",
        help="benchmark domain (code-global = community summaries vs vector vs local graph)",
    )
    parser.add_argument("--queries", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="judge model (OpenAI-compatible)")
    parser.add_argument(
        "--base-url", default=GROQ_BASE_URL, help="OpenAI-compatible judge endpoint (default Groq)"
    )
    parser.add_argument(
        "--api-key-env", default="GROQ_API_KEY", help="env var holding the judge API key"
    )
    parser.add_argument(
        "--gen-base-url",
        default=None,
        help="OpenAI-compatible generation endpoint; if set, generate via it (free OSS tier) "
        "instead of paid Anthropic",
    )
    parser.add_argument("--gen-api-key-env", default=None, help="env var for the generation key")
    parser.add_argument(
        "--gen-model",
        default="meta/llama-3.3-70b-instruct",
        help="generation model with --gen-base-url",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge-runs", type=int, default=3)
    parser.add_argument(
        "--judge-no-seed",
        action="store_true",
        help="omit the seed field from judge calls (some providers, e.g. Gemini, reject it)",
    )
    parser.add_argument(
        "--answers",
        default=None,
        help="cache of generated arm answers; reused if present so generation (paid) runs once",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--check", action="store_true", help="fail if drift exceeds tolerance")
    args = parser.parse_args(argv)

    is_code = args.domain in ("code", "code-global")
    _defaults = {
        "document": (
            "eval/queries.json",
            "out/bench-answers.json",
            DEFAULT_BASELINE,
            DEFAULT_METRICS,
        ),
        "code": (
            "eval/code-queries.json",
            "out/code-bench-answers.json",
            "eval/code-benchmark-baseline.json",
            "METRICS-code.md",
        ),
        "code-global": (
            "eval/code-global-queries.json",
            "out/code-global-answers.json",
            "eval/code-global-baseline.json",
            "METRICS-code-global.md",
        ),
    }
    d_queries, d_answers, d_out, d_metrics = _defaults[args.domain]
    args.queries = args.queries or d_queries
    args.answers = args.answers or d_answers
    args.out = args.out or d_out
    args.metrics = args.metrics or d_metrics

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"{args.api_key_env} not set", file=sys.stderr)
        return 2

    cases = load_eval_cases(args.queries)
    answers_path = Path(args.answers)
    fingerprint = _fingerprint(Path(args.queries), args.graph, args.source)
    cached = _load_answers(answers_path, fingerprint) if answers_path.exists() else None
    if cached is not None:
        responses_by_arm = cached
        print(f"reusing cached answers from {answers_path} (skipping generation)")
    else:
        if answers_path.exists():
            print(f"cache {answers_path} is stale (query set/graph/source changed); regenerating")
        generator: Generator
        if args.gen_base_url:
            gen_key = os.environ.get(args.gen_api_key_env or "")
            if not gen_key:
                print(f"{args.gen_api_key_env} not set (needed for generation)", file=sys.stderr)
                return 2
            generator = OpenAICompatGenerator(
                model=args.gen_model, api_key=gen_key, base_url=args.gen_base_url
            )
        else:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("ANTHROPIC_API_KEY not set (needed for generation)", file=sys.stderr)
                return 2
            generator = AnthropicGenerator()
        system = _CODE_SYSTEM if is_code else None
        arms = _build_arms(args.graph, args.source, args.domain)
        responses_by_arm = {
            arm: _precompute(r, cases, generator, system) for arm, r in arms.items()
        }
        _save_answers(answers_path, responses_by_arm, fingerprint)
        print(f"generated and cached answers to {answers_path}")

    def _on_case(
        arm: str, done: int, total: int, case_id: str, scores: Mapping[str, float]
    ) -> None:
        cells = " ".join(f"{m}={v:.2f}" for m, v in sorted(scores.items()))
        print(f"[{arm}] {done}/{total} {case_id} {cells}", flush=True)

    def _on_arm(report: ArmReport) -> None:
        cells = " · ".join(
            f"{m.metric} {m.mean:.3f} [{m.ci_low:.3f},{m.ci_high:.3f}]" for m in report.metrics
        )
        print(
            f"=== {report.arm} done — {cells} | tokens={report.total_tokens} "
            f"latency={report.mean_latency_ms:.0f}ms cost=${report.cost_usd:.4f}",
            flush=True,
        )

    judge = OpenAICompatJudge(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        send_seed=not args.judge_no_seed,
    )
    report = run_benchmark(
        cases,
        responses_by_arm,
        judge,
        judge_model=args.model,
        seed=args.seed,
        judge_runs=args.judge_runs,
        on_case=_on_case,
        on_arm=_on_arm,
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
