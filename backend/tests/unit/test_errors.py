import pytest

from research_system.errors import PermanentToolError, ToolError, TransientToolError


def test_both_kinds_are_tool_errors() -> None:
    """One `except ToolError` catches every tool failure."""
    assert issubclass(TransientToolError, ToolError)
    assert issubclass(PermanentToolError, ToolError)


def test_error_names_the_tool_that_failed() -> None:
    error = TransientToolError("upstream is down", tool="tavily")
    assert error.tool == "tavily"
    assert str(error) == "upstream is down"


def test_subclasses_inherit_the_constructor() -> None:
    """The subclasses add no fields; the type alone carries the retry decision."""
    assert PermanentToolError("gone", tool="wikipedia").tool == "wikipedia"


def test_a_failing_tool_can_be_caught_by_the_base_class() -> None:
    with pytest.raises(ToolError):
        raise TransientToolError("timed out", tool="scraper")
