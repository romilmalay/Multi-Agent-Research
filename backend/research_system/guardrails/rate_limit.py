"""Pacing outbound calls: a token bucket the researchers share.

Three researchers run concurrently, each firing several searches, and a search API
answers a burst of that with 429s. The bucket smooths it: `burst` calls may go at
once, the rest are spaced at `requests_per_minute`.

A waiter books a slot rather than queueing for one. Under the lock it takes its
token, letting the balance go negative, and works out when that borrowed token comes
due; then it releases the lock and sleeps alone until its own time. N callers
arriving together get N distinct wake times and one brief lock acquisition each, and
there is no `await` inside the critical section.

Sleeping with the lock held instead would pace *this* workload identically — each
waiter wakes owing exactly one token and hands the lock on, so the throughput works
out the same. Measured, not assumed. What it costs is robustness. The pacing would
depend on the sleep being exactly right, because tokens accruing during an over-long
sleep are tokens no other caller can reach, so the rate quietly drops below the
configured one. And it freezes the entire limiter for the length of a wait, which is
what stops a second method ever being added to this class.
"""

import asyncio
import time

from research_system.settings import RateLimitSettings


class RateLimiter:
    """A shared token bucket. One instance per run, awaited by every caller."""

    def __init__(self, settings: RateLimitSettings) -> None:
        self._capacity = float(settings.burst)
        self._rate = settings.requests_per_minute / 60.0
        self._tokens = float(settings.burst)
        self._updated = time.monotonic()
        self._acquired = 0
        self._lock = asyncio.Lock()

    @property
    def acquired(self) -> int:
        """Permits issued so far. The graph's wire-through test reads this."""
        return self._acquired

    async def acquire(self) -> None:
        """Wait until this caller's slot comes due, then take it."""
        async with self._lock:
            now = time.monotonic()
            # Refill for elapsed time. The cap is what stops a long idle period
            # banking an unlimited burst, and it can only bite once that same
            # elapsed time has covered every outstanding booking.
            self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
            self._updated = now
            # A negative balance is a booking: the token is spent now, and the wait
            # is how long until the bucket would have produced it.
            wait = 0.0 if self._tokens >= 1.0 else (1.0 - self._tokens) / self._rate
            self._tokens -= 1.0
            self._acquired += 1
        if wait:
            await asyncio.sleep(wait)
