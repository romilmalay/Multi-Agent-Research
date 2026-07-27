import asyncio
import time
from itertools import pairwise

from research_system.guardrails.rate_limit import RateLimiter
from research_system.settings import RateLimitSettings, get_settings

# 10 permits a second, so a wait is 0.1s and the whole file runs in about a second.
# Real 30 RPM pacing would make every timing assertion cost two seconds.
FAST = RateLimitSettings(requests_per_minute=600, burst=1)
INTERVAL = 0.1
IMMEDIATE = 0.05


def test_the_configured_limit_is_30_rpm() -> None:
    limit = get_settings().guardrails.rate_limit
    assert (limit.requests_per_minute, limit.burst) == (30, 5)


async def test_a_single_call_is_never_delayed() -> None:
    limiter = RateLimiter(get_settings().guardrails.rate_limit)
    started = time.monotonic()
    await limiter.acquire()
    assert time.monotonic() - started < IMMEDIATE


async def test_the_burst_goes_through_at_once() -> None:
    limiter = RateLimiter(RateLimitSettings(requests_per_minute=600, burst=3))
    started = time.monotonic()
    for _ in range(3):
        await limiter.acquire()
    assert time.monotonic() - started < IMMEDIATE


async def test_the_call_after_the_burst_waits_one_interval() -> None:
    limiter = RateLimiter(RateLimitSettings(requests_per_minute=600, burst=3))
    for _ in range(3):
        await limiter.acquire()
    started = time.monotonic()
    await limiter.acquire()
    assert time.monotonic() - started >= INTERVAL * 0.9


async def test_concurrent_callers_are_paced() -> None:
    """Five callers, burst of one: the last is due four intervals in.

    The upper bound catches an implementation that accumulates waits — one that
    slept for the sum rather than the maximum would land at 1.0s.
    """
    limiter = RateLimiter(FAST)
    started = time.monotonic()
    await asyncio.gather(*(limiter.acquire() for _ in range(5)))
    elapsed = time.monotonic() - started
    assert elapsed >= INTERVAL * 4 * 0.9, "not paced: the limiter let them all through"
    assert elapsed < 0.7, "waits accumulated instead of overlapping"


async def test_the_lock_is_free_while_callers_wait() -> None:
    """No `await` inside the critical section.

    This reads a private attribute because the rule is itself about internal
    discipline, and it is the one property that separates this design from sleeping
    with the lock held — that version paces this workload identically, so no timing
    assertion can tell them apart. Verified against both.
    """
    limiter = RateLimiter(FAST)
    waiters = [asyncio.create_task(limiter.acquire()) for _ in range(5)]
    await asyncio.sleep(INTERVAL / 2)
    held = limiter._lock.locked()
    await asyncio.gather(*waiters)
    assert not held, "the limiter slept holding its lock"


async def test_acquisitions_come_out_spaced_and_in_arrival_order() -> None:
    limiter = RateLimiter(FAST)
    taken: list[float] = []

    async def take() -> None:
        await limiter.acquire()
        taken.append(time.monotonic())

    await asyncio.gather(*(take() for _ in range(4)))

    assert taken == sorted(taken), "a booked slot must not overtake an earlier one"
    gaps = [later - earlier for earlier, later in pairwise(taken)]
    assert all(gap >= INTERVAL * 0.9 for gap in gaps), gaps


async def test_tokens_refill_while_nobody_is_asking() -> None:
    limiter = RateLimiter(FAST)
    await limiter.acquire()
    await asyncio.sleep(INTERVAL * 1.5)
    started = time.monotonic()
    await limiter.acquire()
    assert time.monotonic() - started < IMMEDIATE


async def test_idling_banks_no_more_than_the_burst() -> None:
    """Otherwise a slow run could store up an unlimited burst and spend it at once."""
    limiter = RateLimiter(RateLimitSettings(requests_per_minute=600, burst=2))
    await asyncio.sleep(INTERVAL * 5)

    started = time.monotonic()
    await limiter.acquire()
    await limiter.acquire()
    assert time.monotonic() - started < IMMEDIATE

    await limiter.acquire()
    assert time.monotonic() - started >= INTERVAL * 0.9


async def test_the_acquired_count_tracks_every_permit() -> None:
    """Phase 10 asserts a 3-researcher run records at least 3 acquisitions."""
    limiter = RateLimiter(FAST)
    assert limiter.acquired == 0
    await asyncio.gather(*(limiter.acquire() for _ in range(3)))
    assert limiter.acquired == 3
