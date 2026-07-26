from collections.abc import Iterator

import pytest

from research_system.logging import clear_context


@pytest.fixture(autouse=True)
def _clean_logging_context() -> Iterator[None]:
    """No log context leaks between tests."""
    clear_context()
    yield
    clear_context()
