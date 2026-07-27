"""Search result cache.

The quality-gate retry re-issues queries the first research pass already ran, and
researchers working on related sub-topics search overlapping terms. Each repeat is
a paid API call for an answer already on disk.

SQLite because the pipeline runs as one process. A multi-worker deployment would
replace this with Redis, which implements the same four methods; nothing outside
this file would change.

The methods are synchronous although the callers are async. A local SQLite read
takes microseconds, so the pause is shorter than the scheduling it would take to
avoid it, and `aiosqlite` would add a dependency and a thread pool for nothing.
"""

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass

from research_system.domain.state import SearchResult
from research_system.settings import CacheSettings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    tool       TEXT NOT NULL,
    query      TEXT NOT NULL,
    results    TEXT NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (tool, query)
)
"""


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Cache activity for one run, plus what is currently stored."""

    hits: int
    misses: int
    entries: int


def _normalize(query: str) -> str:
    """Case and spacing must not decide whether a search is paid for twice."""
    return " ".join(query.lower().split())


class SearchCache:
    """Tool results keyed on the tool and the query it was given."""

    def __init__(self, settings: CacheSettings) -> None:
        self._settings = settings
        self._hits = 0
        self._misses = 0
        if settings.enabled:
            settings.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connection() as conn:
                conn.execute(_SCHEMA)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """`with conn` commits but does not close, so both wrappers are needed."""
        with closing(sqlite3.connect(self._settings.path)) as conn, conn:
            yield conn

    def get(self, tool: str, query: str) -> list[SearchResult] | None:
        """Stored results, or None when there are none worth using."""
        if not self._settings.enabled:
            return None
        with self._connection() as conn:
            row = conn.execute(
                "SELECT results FROM entries WHERE tool = ? AND query = ? AND expires_at > ?",
                (tool, _normalize(query), time.time()),
            ).fetchone()
        if row is None:
            self._misses += 1
            return None
        self._hits += 1
        results: list[SearchResult] = json.loads(row[0])
        return results

    def set(self, tool: str, query: str, results: list[SearchResult]) -> None:
        """Store results until the TTL runs out, replacing any earlier answer."""
        if not self._settings.enabled:
            return
        with self._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO entries (tool, query, results, expires_at)"
                " VALUES (?, ?, ?, ?)",
                (
                    tool,
                    _normalize(query),
                    json.dumps(results),
                    time.time() + self._settings.ttl_seconds,
                ),
            )

    def stats(self) -> CacheStats:
        """Hits and misses since this cache was built; entries still live."""
        entries = 0
        if self._settings.enabled:
            with self._connection() as conn:
                entries = conn.execute(
                    "SELECT count(*) FROM entries WHERE expires_at > ?", (time.time(),)
                ).fetchone()[0]
        return CacheStats(hits=self._hits, misses=self._misses, entries=entries)

    def clear(self) -> None:
        """Drop every stored result. Hit and miss counts are run stats, and stay."""
        if not self._settings.enabled:
            return
        with self._connection() as conn:
            conn.execute("DELETE FROM entries")
