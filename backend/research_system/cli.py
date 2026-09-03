"""`research "a question"` — the pipeline with a person at the other end.

This is an edge, and edges have four jobs the library deliberately does not do:
configure logging, validate the query before anything is spent on it, own the
checkpointer's connection for the length of the run, and write the run down when
it is over. Everything else here is presentation — the report, what each agent
did, and what the run cost.

The cost line is why the trace carries input and output tokens separately. They
are billed at different rates, so a total alone cannot be priced, and a number
this file made up would be worse than no number.
"""

import argparse
import asyncio
import sys
import time
from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Any

from research_system.agents.trace import spent
from research_system.artifacts import write_artifact
from research_system.domain.state import ResearchState
from research_system.errors import InvalidQueryError, UnknownRunError
from research_system.graph.checkpointer import checkpointer
from research_system.guardrails.injection import validate_query
from research_system.llm.usage import Usage
from research_system.logging import configure_logging
from research_system.pipeline import Event, Finished, Progress, stream_pipeline, stream_resume
from research_system.settings import Settings, get_settings

STEPS = ("agent", "duration", "tokens", "prompt", "summary")
STEPS_NUMERIC = (1, 2)

TOTALS = ("agent", "calls", "time", "share", "tokens", "cost")
TOTALS_NUMERIC = (1, 2, 3, 4, 5)

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
            return await _watch(stream_resume(resume, settings=settings, checkpointer=saver))
        # Validated here, at the edge, so a rejected query costs nothing at all.
        clean = validate_query(query or "", settings.guardrails)
        return await _watch(stream_pipeline(clean, settings=settings, checkpointer=saver))


async def _watch(events: AsyncIterator[Event]) -> ResearchState:
    """Print each step as it happens, and return the run once it is over.

    Progress goes to stderr and the report goes to stdout, so redirecting the
    report to a file still shows the run happening in the terminal.
    """
    final: ResearchState | None = None
    for_now = time.perf_counter()

    async for event in events:
        match event:
            case Finished(state=state):
                final = state
            case Progress(agent=agent, detail=detail):
                elapsed = time.perf_counter() - for_now
                print(f"[{elapsed:6,.1f}s] {agent:<13} {detail}", file=sys.stderr)

    if final is None:
        raise RuntimeError("the run ended without producing a state")
    return final


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
    """The run, printed: the report, what happened, where it went, what it cost."""
    trace = final["pipeline_trace"]
    print(final["final_report"] or "The run produced no report.")
    print()
    print(_table(STEPS, list(_steps(trace)), numeric=STEPS_NUMERIC))
    print()
    print(_table(TOTALS, _totals(trace, settings), numeric=TOTALS_NUMERIC))
    print()
    print(_summary(final, settings, seconds=seconds))
    errors = final["errors"]
    if errors:
        print()
        print(f"{len(errors)} non-fatal error" + ("s:" if len(errors) != 1 else ":"))
        for error in errors:
            print(f"  - {error}")

    # Written last and printed, because everything above scrolls away and this does not.
    print()
    print(f"artifact: {write_artifact(final, seconds=seconds, settings=settings)}")


def _steps(trace: list[dict[str, Any]]) -> Iterable[tuple[str, ...]]:
    """What happened: one line per step, in the order the run went through them."""
    for entry in trace:
        yield (
            entry["agent"],
            f"{entry['duration_ms']:,.0f}ms",
            f"{entry['tokens']:,}",
            entry["prompt_hash"][:8] or "-",
            entry["summary"],
        )


def _totals(trace: list[dict[str, Any]], settings: Settings) -> list[tuple[str, ...]]:
    """Where it went: one line per agent, slowest first, and a total.

    The step table above is chronological, which is the wrong shape for the
    question this table exists to answer. A run with three researchers and two
    writer passes spreads one agent's cost over five rows; the bottleneck is
    whichever agent this table puts at the top.

    `share` is of agent time, not of the clock. Agents that ran in parallel each
    count their own duration in full, so the total here exceeds the elapsed time
    of the run — and by how much is exactly what the fan-out bought.
    """
    per_agent: dict[str, tuple[int, float, Usage]] = {}
    for entry in trace:
        calls, elapsed, usage = per_agent.get(entry["agent"], (0, 0.0, Usage()))
        per_agent[entry["agent"]] = (
            calls + 1,
            elapsed + entry["duration_ms"],
            Usage(
                input_tokens=usage.input_tokens + entry["input_tokens"],
                output_tokens=usage.output_tokens + entry["output_tokens"],
            ),
        )

    agent_ms = sum(elapsed for _, elapsed, _ in per_agent.values())
    slowest = sorted(per_agent.items(), key=lambda item: item[1][1], reverse=True)
    rows = [
        _total_row(agent, calls, elapsed, usage, agent_ms, settings)
        for agent, (calls, elapsed, usage) in slowest
    ]
    if rows:
        rows.append(_total_row("total", len(trace), agent_ms, spent(trace), agent_ms, settings))
    return rows


def _total_row(
    agent: str, calls: int, elapsed: float, usage: Usage, agent_ms: float, settings: Settings
) -> tuple[str, ...]:
    """One rollup line. `agent_ms` is the run's agent time, which `share` is of."""
    return (
        agent,
        f"{calls:,}",
        f"{elapsed:,.0f}ms",
        # `or 1` only ever divides zero by one: a real step takes measurable time.
        f"{elapsed / (agent_ms or 1):.1%}",
        f"{usage.total_tokens:,}",
        f"${usage.cost_usd(settings.llm.price):.4f}",
    )


def _table(
    header: tuple[str, ...], rows: list[tuple[str, ...]], *, numeric: tuple[int, ...]
) -> str:
    """A fixed-width table. `numeric` columns are right-aligned so digits line up
    under digits; trailing padding is trimmed."""
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]
    lines = [
        GAP.join(
            cell.rjust(width) if column in numeric else cell.ljust(width)
            for column, (cell, width) in enumerate(zip(line, widths, strict=True))
        ).rstrip()
        for line in (header, *rows)
    ]
    lines.insert(1, GAP.join("-" * width for width in widths))
    return "\n".join(lines)


def _summary(final: ResearchState, settings: Settings, *, seconds: float) -> str:
    """The one line that says what the answer cost."""
    usage = spent(final["pipeline_trace"])
    return (
        f"run {final['run_id']} | {len(final['pipeline_trace'])} steps | {seconds:,.1f}s | "
        f"{usage.total_tokens:,} tokens "
        f"({usage.input_tokens:,} in / {usage.output_tokens:,} out) "
        f"of {settings.budget.max_tokens_per_run:,} | "
        f"${usage.cost_usd(settings.llm.price):.4f}"
    )
