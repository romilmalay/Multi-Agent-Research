"""`research "a question"` — the pipeline with a person at the other end.

This is an edge, and edges have three jobs the library deliberately does not do:
configure logging, validate the query before anything is spent on it, and own the
checkpointer's connection for the length of the run. Everything after that is
presentation — the report, what each agent did, and what the run cost.

The cost line is why the trace carries input and output tokens separately. They
are billed at different rates, so a total alone cannot be priced, and a number
this file made up would be worse than no number.
"""

import argparse
import asyncio
import sys
import time
from collections.abc import Iterable, Sequence
from typing import Any

from research_system.domain.state import ResearchState
from research_system.errors import InvalidQueryError, UnknownRunError
from research_system.graph.checkpointer import checkpointer
from research_system.guardrails.injection import validate_query
from research_system.llm.usage import Usage
from research_system.logging import configure_logging
from research_system.pipeline import resume_pipeline, run_pipeline
from research_system.settings import Settings, get_settings

HEADER = ("agent", "duration", "tokens", "prompt", "summary")
NUMERIC = (1, 2)
"""Columns to right-align, so digits line up under digits and sizes can be compared."""

GAP = "  "

EXIT_OK = 0
EXIT_NO_REPORT = 1
EXIT_BAD_REQUEST = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research",
        description="Research a question with the multi-agent pipeline.",
    )
    parser.add_argument("query", nargs="?", help="the question to research")
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="carry on the saved run with this id, from the step it stopped at",
    )
    return parser


async def _research(query: str | None, resume: str | None, settings: Settings) -> ResearchState:
    """One run, with the checkpointer open for exactly as long as it takes."""
    async with checkpointer(settings) as saver:
        if resume:
            return await resume_pipeline(resume, settings=settings, checkpointer=saver)
        # Validated here, at the edge, so a rejected query costs nothing at all.
        clean = validate_query(query or "", settings.guardrails)
        return await run_pipeline(clean, settings=settings, checkpointer=saver)


def main(argv: Sequence[str] | None = None) -> int:
    """The `research` entry point. Returns the process exit code."""
    args = _parser().parse_args(argv)
    if not args.query and not args.resume:
        _parser().error("give a question to research, or --resume a saved run")

    settings = get_settings()
    configure_logging(settings.logging.level, json_logs=settings.logging.format == "json")

    started = time.perf_counter()
    try:
        final = asyncio.run(_research(args.query, args.resume, settings))
    except (InvalidQueryError, UnknownRunError) as refused:
        print(f"error: {refused}", file=sys.stderr)
        return EXIT_BAD_REQUEST

    _report(final, settings, seconds=time.perf_counter() - started)
    return EXIT_OK if final["final_report"] else EXIT_NO_REPORT


def _report(final: ResearchState, settings: Settings, *, seconds: float) -> None:
    """The run, printed: the report, then how it was produced, then what it cost."""
    print(final["final_report"] or "The run produced no report.")
    print()
    print(_table(HEADER, list(_rows(final["pipeline_trace"]))))
    print()
    print(_summary(final, settings, seconds=seconds))
    errors = final["errors"]
    if errors:
        print()
        print(f"{len(errors)} non-fatal error" + ("s:" if len(errors) != 1 else ":"))
        for error in errors:
            print(f"  - {error}")


def _rows(trace: list[dict[str, Any]]) -> Iterable[tuple[str, ...]]:
    """One line per agent, in the order the run went through them."""
    for entry in trace:
        yield (
            entry["agent"],
            f"{entry['duration_ms']:,.0f}ms",
            f"{entry['tokens']:,}",
            entry["prompt_hash"][:8] or "-",
            entry["summary"],
        )


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """A fixed-width table, numbers right-aligned. Trailing padding is trimmed."""
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]
    lines = [
        GAP.join(
            cell.rjust(width) if column in NUMERIC else cell.ljust(width)
            for column, (cell, width) in enumerate(zip(line, widths, strict=True))
        ).rstrip()
        for line in (header, *rows)
    ]
    lines.insert(1, GAP.join("-" * width for width in widths))
    return "\n".join(lines)


def _summary(final: ResearchState, settings: Settings, *, seconds: float) -> str:
    """The one line that says what the answer cost."""
    usage = _spent(final["pipeline_trace"])
    return (
        f"run {final['run_id']} | {len(final['pipeline_trace'])} steps | {seconds:,.1f}s | "
        f"{usage.total_tokens:,} tokens "
        f"({usage.input_tokens:,} in / {usage.output_tokens:,} out) "
        f"of {settings.budget.max_tokens_per_run:,} | "
        f"${usage.cost_usd(settings.llm.price):.4f}"
    )


def _spent(trace: list[dict[str, Any]]) -> Usage:
    """What the whole run used, from the per-agent counts the trace already holds."""
    return Usage(
        input_tokens=sum(entry["input_tokens"] for entry in trace),
        output_tokens=sum(entry["output_tokens"] for entry in trace),
    )
