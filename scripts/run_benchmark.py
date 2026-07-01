"""P3.1: run the three retrieval arms over the query set and score them with a judge.

Pre-computes each arm's grounded answer per question (real tokens + latency), then
judges every case for faithfulness + context_precision via an OpenAI-compatible
endpoint (Groq by default) and aggregates with seeded numpy bootstrap CIs. Writes a
committed JSON baseline + METRICS.md.

Usage:
    GROQ_API_KEY=... ANTHROPIC_API_KEY=... \\
      python3 scripts/run_benchmark.py out/monster-manual.json "<Monster Manual.pdf>"

    # regression gate: fail if a fresh run drifts past tolerance from the baseline
    python3 scripts/run_benchmark.py out/monster-manual.json "<book.pdf>" --check

    # reuse cached answers (skip generation entirely):
    python3 scripts/run_benchmark.py out/monster-manual.json dummy \\
      --answers out/bench-answers.json --skip-fingerprint

    # multi-model parallel generation (MODEL@BASE_URL@KEY_ENV, space-separated):
    DEEPSEEK_API_KEY=... OPENAI_API_KEY=... \\
      python3 scripts/run_benchmark.py out/monster-manual.json "<book.pdf>" \\
      --gen-models deepseek-chat@https://api.deepseek.com@DEEPSEEK_API_KEY \\
                   gpt-4o-mini@https://api.openai.com/v1@OPENAI_API_KEY
    # Produces out/bench-answers-{slug}.json and METRICS-{slug}.md per model.

Requires: pip install -e ".[document,retrieval,embeddings]"
"""

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from glyph.eval.harness import ArmReport, run_benchmark, score_arm
from glyph.eval.judge import GROQ_BASE_URL, OpenAICompatJudge
from glyph.eval.report import DEFAULT_TOLERANCE, regression_check, render_markdown, to_dict
from glyph.eval.response import ArmResponse
from glyph.extract.document import chunk, pdf
from glyph.model.contract import ContextPack
from glyph.model.node import NodeType
from glyph.retrieval.community import CommunityRetriever
from glyph.retrieval.graph import GraphRetriever
from glyph.retrieval.hybrid import HybridRetriever
from glyph.retrieval.multi_anchor import MultiAnchorRetriever
from glyph.retrieval.path import PathRetriever
from glyph.retrieval.port import Retriever
from glyph.retrieval.reranked import RerankedRetriever
from glyph.retrieval.reranker import CrossEncoderReranker
from glyph.store.networkx_store import NetworkXStore

DEFAULT_BASELINE = "eval/benchmark-baseline.json"
DEFAULT_METRICS = "METRICS.md"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-").lower()


def _parse_gen_spec(spec: str) -> tuple[str, str, str]:
    """Parse 'model@base_url@key_env' into (model, base_url, key_env)."""
    parts = spec.split("@", 2)
    if len(parts) != 3:
        raise ValueError(f"invalid --gen-models spec {spec!r}; expected model@base_url@key_env")
    return parts[0], parts[1], parts[2]


def _run_one_model(
    slug: str,
    generator: Generator,  # type: ignore[no-untyped-def]
    arms: dict,  # type: ignore[type-arg]
    cases,  # type: ignore[no-untyped-def]
    system: str | None,
    fingerprint: str,
    answers_path: Path,
    skip_fingerprint: bool,
) -> dict:  # type: ignore[type-arg]
    """Generate and cache answers for one generator; returns responses_by_arm."""
    if skip_fingerprint and answers_path.exists():
        raw = json.loads(answers_path.read_text(encoding="utf-8"))
        return {
            arm: {cid: ArmResponse.model_validate(r) for cid, r in rs.items()}
            for arm, rs in raw["answers"].items()
        }
    cached = _load_answers(answers_path, fingerprint) if answers_path.exists() else None
    if cached is not None:
        print(f"[{slug}] reusing cached answers from {answers_path}")
        return cached
    print(f"[{slug}] generating answers...", flush=True)
    responses = {arm: _precompute(r, cases, generator, system) for arm, r in arms.items()}
    _save_answers(answers_path, responses, fingerprint)
    print(f"[{slug}] cached to {answers_path}", flush=True)
    return responses

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
    graph_pagerank = GraphRetriever(
        store=store, embedder=embedder, nodes=nodes, pagerank_weight=0.5
    )
    vector = VectorBaseline(embedder=embedder)
    vector.index(documents)
    hybrid = HybridRetriever(graph, vector)
    multi_anchor = MultiAnchorRetriever(store=store, embedder=embedder, nodes=nodes)
    path = PathRetriever(store=store, embedder=embedder, nodes=nodes)
    reranker = CrossEncoderReranker()
    # Reranks `vector` (0.460 context_precision, current best arm in METRICS.md) — not
    # `hybrid` (0.347, currently the worst of the three original arms). A reranker can
    # only reorder/filter what its underlying retriever already found; it can't recover
    # recall `hybrid` already lost. VectorBaseline governs result size via `k`, not
    # token_budget, so wrap it in `_KVectorRetriever(vector, k=50)` to actually widen the
    # candidate pool to 50 before reranking down to the final top-5.
    reranked_vector = RerankedRetriever(_KVectorRetriever(vector, k=50), reranker, k=5)
    return {
        "graph": graph,
        "graph_pagerank": graph_pagerank,
        "vector": vector,
        "hybrid": hybrid,
        "multi_anchor": multi_anchor,
        "path": path,
        "reranked_vector": reranked_vector,
    }


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


class _KVectorRetriever:
    """Adapts VectorBaseline to the Retriever port with a fixed k, for --k-sweep."""

    def __init__(self, vector: VectorBaseline, k: int) -> None:
        self._vector = vector
        self._k = k

    def retrieve(self, query: str, token_budget: int = 1000) -> ContextPack:
        return self._vector.retrieve(query, token_budget=token_budget, k=self._k)


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
        "--base-url", default=GROQ_BASE_URL, help="OpenAI-compat judge endpoint (default: Groq)"
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
    parser.add_argument(
        "--gen-models",
        nargs="+",
        metavar="MODEL@BASE_URL@KEY_ENV",
        default=None,
        help=(
            "run generation with multiple models in parallel; "
            "each spec: model@base_url@key_env "
            "(e.g. deepseek-chat@https://api.deepseek.com@DEEPSEEK_API_KEY)"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-fingerprint",
        action="store_true",
        help="skip cache fingerprint validation (use with --answers when source unavailable)",
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
    parser.add_argument(
        "--k-sweep",
        action="store_true",
        help="run vector arm with k=2,3,5,8 and print context_precision table; skip full benchmark",
    )
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

    if args.k_sweep:
        embedder = SentenceTransformerEmbedder()
        documents = [
            (piece.label, piece.text)
            for piece in chunk.by_creature(pdf.load(args.source))
            if chunk.is_creature(piece)
        ]
        sweep_generator: Generator
        if args.gen_base_url:
            gen_key = os.environ.get(args.gen_api_key_env or "")
            if not gen_key:
                print(f"{args.gen_api_key_env} not set (needed for generation)", file=sys.stderr)
                return 2
            sweep_generator = OpenAICompatGenerator(
                model=args.gen_model, api_key=gen_key, base_url=args.gen_base_url
            )
        else:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("ANTHROPIC_API_KEY not set (needed for generation)", file=sys.stderr)
                return 2
            sweep_generator = AnthropicGenerator()
        judge = OpenAICompatJudge(model=args.model, api_key=api_key, base_url=args.base_url)
        print(f"{'k':>4}  context_precision")
        for k in (2, 3, 5, 8):
            vector = VectorBaseline(embedder=embedder)
            vector.index(documents)
            retriever = _KVectorRetriever(vector, k)
            responses = {
                case.id: _precompute(retriever, [case], sweep_generator, None)[case.id]
                for case in cases
            }
            arm_report = score_arm("vector", cases, responses, judge, seed=args.seed)
            cp = next(m.mean for m in arm_report.metrics if m.metric == "context_precision")
            print(f"{k:>4}  {cp:.3f}")
        return 0

    # --- multi-model parallel mode ---
    if args.gen_models:
        specs: list[tuple[str, Generator]] = []
        for spec_str in args.gen_models:
            model_name, base_url, key_env = _parse_gen_spec(spec_str)
            key = os.environ.get(key_env)
            if not key:
                print(f"{key_env} not set for model {model_name}", file=sys.stderr)
                return 2
            specs.append((
                _slug(model_name),
                OpenAICompatGenerator(model=model_name, api_key=key, base_url=base_url),
            ))

        fingerprint = _fingerprint(Path(args.queries), args.graph, args.source)
        arms = _build_arms(args.graph, args.source, args.domain)
        system = _CODE_SYSTEM if is_code else None
        out_dir = Path(args.answers).parent

        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = {
                slug: pool.submit(
                    _run_one_model,
                    slug, gen, arms, cases, system, fingerprint,
                    out_dir / f"bench-answers-{slug}.json",
                    args.skip_fingerprint,
                )
                for slug, gen in specs
            }

        judge = OpenAICompatJudge(model=args.model, api_key=api_key, base_url=args.base_url)
        for slug, future in futures.items():
            responses_by_arm = future.result()

            def _on_arm_multi(report: ArmReport, _slug: str = slug) -> None:
                cells = " · ".join(
                    f"{m.metric} {m.mean:.3f} [{m.ci_low:.3f},{m.ci_high:.3f}]"
                    for m in report.metrics
                )
                print(
                    f"[{_slug}] {report.arm} — {cells} | tokens={report.total_tokens} "
                    f"cost=${report.cost_usd:.4f}",
                    flush=True,
                )

            report = run_benchmark(
                cases, responses_by_arm, judge,
                judge_model=args.model, seed=args.seed, on_arm=_on_arm_multi,
            )
            fresh = to_dict(report)
            stem = Path(args.out).stem
            baseline_path = Path(args.out).with_name(f"{stem}-{slug}.json")
            metrics_path = Path(args.metrics).with_suffix("").with_name(
                Path(args.metrics).stem + f"-{slug}.md"
            )
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(fresh, ensure_ascii=False, indent=2) + "\n", "utf-8")
            metrics_path.write_text(render_markdown(report) + "\n", encoding="utf-8")
            print(f"[{slug}] wrote {baseline_path} and {metrics_path}")
        return 0

    # --- single-model mode (existing behaviour) ---
    answers_path = Path(args.answers)
    if args.skip_fingerprint and answers_path.exists():
        cached_raw = json.loads(answers_path.read_text(encoding="utf-8"))
        cached = {
            arm: {cid: ArmResponse.model_validate(r) for cid, r in responses.items()}
            for arm, responses in cached_raw["answers"].items()
        }
    else:
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

    def _on_arm(report: ArmReport) -> None:
        cells = " · ".join(
            f"{m.metric} {m.mean:.3f} [{m.ci_low:.3f},{m.ci_high:.3f}]" for m in report.metrics
        )
        print(
            f"=== {report.arm} done — {cells} | tokens={report.total_tokens} "
            f"latency={report.mean_latency_ms:.0f}ms cost=${report.cost_usd:.4f}",
            flush=True,
        )

    judge = OpenAICompatJudge(model=args.model, api_key=api_key, base_url=args.base_url)
    report = run_benchmark(
        cases,
        responses_by_arm,
        judge,
        judge_model=args.model,
        seed=args.seed,
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
