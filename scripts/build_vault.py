"""Build a knowledge graph from an Obsidian vault (recursive .md files).

Usage
-----
    python scripts/build_vault.py <vault_dir> <out_graph.json> [options]

Options
-------
    --provider   litellm | anthropic          (default: litellm)
    --model      any litellm model string     (default: ollama/llama3)
    --base-url   override base URL for litellm (default: http://localhost:11434)
    --api-key    API key (default: none; omit for local Ollama)

Examples
--------
# Local Ollama (no cost):
    python scripts/build_vault.py ~/notes out/vault.json

# OpenRouter with DeepSeek:
    python scripts/build_vault.py ~/notes out/vault.json \\
        --provider litellm \\
        --model openrouter/deepseek/deepseek-chat \\
        --api-key sk-or-...

# Anthropic (requires ANTHROPIC_API_KEY env var):
    python scripts/build_vault.py ~/notes out/vault.json \\
        --provider anthropic \\
        --model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from glyph.vault import build_vault


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from an Obsidian markdown vault.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("vault", help="Path to the Obsidian vault directory")
    parser.add_argument("output", help="Path for the output graph JSON file")
    parser.add_argument(
        "--provider",
        default="litellm",
        choices=["litellm", "anthropic"],
        help="LLM provider (default: litellm)",
    )
    parser.add_argument(
        "--model",
        default="ollama/llama3",
        help="Model string understood by the provider (default: ollama/llama3)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Base URL for litellm (default: http://localhost:11434 for Ollama)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (omit for local Ollama)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    vault_path = Path(args.vault).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not vault_path.is_dir():
        print(f"error: {vault_path} is not a directory", file=sys.stderr)
        return 2

    build_vault(
        vault_path=vault_path,
        output_path=output_path,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
