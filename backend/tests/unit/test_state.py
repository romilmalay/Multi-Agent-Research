import operator
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from research_system.domain.state import ResearchState, default_state

REDUCED_FIELDS = {
    "token_count",
    "errors",
    "pipeline_trace",
    "sources",
    "search_queries_used",
}

EXPECTED_FIELDS: dict[str, Any] = {
    "query": str,
    "token_count": int,
    "errors": list[str],
    "pipeline_trace": list[dict[str, Any]],
    "sub_topics": list[str],
    "research_plan": str,
    "sources": list[dict[str, Any]],
    "search_queries_used": list[str],
    "quality_score": float,
    "quality_passed": bool,
    "source_ranking": list[dict[str, Any]],
    "key_claims": list[dict[str, Any]],
    "conflicts": list[str],
    "synthesis": str,
    "drafts": list[str],
    "current_draft": str,
    "revision_count": int,
    "review": dict[str, Any],
    "final_report": str,
    "retry_count": int,
    "run_id": str,
}


def test_field_names_match_the_spec_exactly() -> None:
    assert set(ResearchState.__annotations__) == set(EXPECTED_FIELDS)


def test_spec_field_count() -> None:
    """19 spec fields plus the 2 internal control fields."""
    assert len(EXPECTED_FIELDS) == 21


def test_every_field_has_the_expected_type() -> None:
    hints = get_type_hints(ResearchState)
    assert hints == EXPECTED_FIELDS


def test_every_key_is_optional() -> None:
    """Nodes return only what they changed, so no key may be required."""
    assert ResearchState.__required_keys__ == frozenset()


def _reducer(field: str) -> Any:
    """The reducer attached to a field, or None if it is last-write-wins."""
    hint = get_type_hints(ResearchState, include_extras=True)[field]
    return get_args(hint)[1] if get_origin(hint) is Annotated else None


def test_exactly_the_parallel_written_fields_have_reducers() -> None:
    reduced = {name for name in EXPECTED_FIELDS if _reducer(name) is not None}
    assert reduced == REDUCED_FIELDS


def test_every_reducer_is_add() -> None:
    assert all(_reducer(name) is operator.add for name in REDUCED_FIELDS)


def test_list_reducer_concatenates_parallel_updates() -> None:
    """Three researchers each returning one source produce three sources, not one."""
    add = _reducer("sources")
    assert add(add([{"url": "a"}], [{"url": "b"}]), [{"url": "c"}]) == [
        {"url": "a"},
        {"url": "b"},
        {"url": "c"},
    ]


def test_token_count_reducer_sums() -> None:
    add = _reducer("token_count")
    assert add(add(100, 250), 50) == 400


def test_unreduced_field_would_overwrite() -> None:
    """`quality_score` has no reducer: a single writer, last write wins."""
    assert _reducer("quality_score") is None


def test_default_state_initializes_every_field() -> None:
    """No node should ever need a `.get()` guard."""
    assert set(default_state("q")) == set(EXPECTED_FIELDS)


def test_default_state_values_have_the_declared_types() -> None:
    state = default_state("q")
    for name, expected in EXPECTED_FIELDS.items():
        assert isinstance(state[name], get_origin(expected) or expected)  # type: ignore[literal-required]


def test_default_state_keeps_the_query() -> None:
    assert default_state("why is the sky blue")["query"] == "why is the sky blue"


def test_default_state_generates_a_run_id() -> None:
    assert default_state("q")["run_id"] != default_state("q")["run_id"]


def test_default_state_accepts_a_caller_run_id() -> None:
    """The API supplies the id it already wrote to the runs table."""
    assert default_state("q", run_id="run-123")["run_id"] == "run-123"


def test_default_state_returns_fresh_containers() -> None:
    """Two runs must not share a list, or one run's sources leak into the other."""
    first, second = default_state("a"), default_state("b")
    first["sources"].append({"url": "x"})
    assert second["sources"] == []
