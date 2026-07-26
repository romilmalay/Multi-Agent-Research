from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from research_system.prompts import registry

FIXTURE = """
system: |
  You plan research for {{query}}.
human: |
  {{#claims}}
  [{{idx}}] {{text}}
  {{/claims}}
"""


@pytest.fixture
def prompt_dir(tmp_path: Path) -> Path:
    (tmp_path / "demo.yaml").write_text(FIXTURE)
    return tmp_path


def test_load_builds_a_system_and_human_pair(prompt_dir: Path) -> None:
    prompt = registry.load("demo", directory=prompt_dir)
    messages = prompt.render(query="llm agents", claims=[])

    assert [type(m) for m in messages] == [SystemMessage, HumanMessage]
    assert prompt.name == "demo"


def test_mustache_loops_render_sequential_numbering(prompt_dir: Path) -> None:
    claims = [{"idx": 1, "text": "first"}, {"idx": 2, "text": "second"}]
    _, human = registry.load("demo", directory=prompt_dir).render(query="q", claims=claims)

    assert "[1] first" in human.content
    assert "[2] second" in human.content


def test_declared_variables_are_the_templates_own(prompt_dir: Path) -> None:
    assert registry.load("demo", directory=prompt_dir).variables == {"query", "claims"}


def test_hash_is_stable_and_tracks_file_content(prompt_dir: Path) -> None:
    first = registry.load("demo", directory=prompt_dir).hash
    assert first == registry.load("demo", directory=prompt_dir).hash

    (prompt_dir / "demo.yaml").write_text(FIXTURE + "\n")
    assert registry.load("demo", directory=prompt_dir).hash != first
    assert len(first) == registry.HASH_LENGTH


@pytest.mark.parametrize(
    "body",
    [
        "system: only one role",
        "system: a\nhuman: b\nextra: c",
        "system: a\nhuman: '  '",
        "- not a mapping",
    ],
)
def test_off_contract_files_are_rejected(tmp_path: Path, body: str) -> None:
    (tmp_path / "bad.yaml").write_text(body)

    with pytest.raises(ValueError):
        registry.load("bad", directory=tmp_path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        registry.load("nope", directory=tmp_path)
