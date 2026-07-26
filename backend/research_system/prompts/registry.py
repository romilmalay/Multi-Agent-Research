"""Loads an agent's prompt from YAML.

Each file holds exactly two message templates, `system` and `human`, written in
mustache so that loops and conditionals live in the template instead of in
Python string building. `load()` also returns a short hash of the file, which
travels with the run in `pipeline_trace` and in the Langfuse trace: a report can
then be tied back to the exact prompt text that produced it.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = Path(__file__).parent
HASH_LENGTH = 8
ROLES = ("system", "human")


@dataclass(frozen=True)
class Prompt:
    """A loaded prompt: the template, plus the identity of its source text."""

    name: str
    template: ChatPromptTemplate
    hash: str

    @property
    def variables(self) -> set[str]:
        """Names the template expects to be given."""
        return set(self.template.input_variables)

    def render(self, **values: Any) -> list[BaseMessage]:
        """The two messages, filled in."""
        return self.template.format_messages(**values)


def load(name: str, *, directory: Path = PROMPTS_DIR) -> Prompt:
    """Load the prompt named `name` from `directory`."""
    path = directory / f"{name}.yaml"
    raw = path.read_bytes()
    messages = _parse(raw, path)

    template = ChatPromptTemplate.from_messages(
        [(role, messages[role]) for role in ROLES],
        template_format="mustache",
    )
    return Prompt(name=name, template=template, hash=sha256(raw).hexdigest()[:HASH_LENGTH])


def _parse(raw: bytes, path: Path) -> dict[str, str]:
    """The file as `{role: template}`, rejecting anything off-contract."""
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or set(data) != set(ROLES):
        raise ValueError(f"{path} must define exactly {list(ROLES)}")
    for role, text in data.items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}: {role} is empty")
    return {role: str(data[role]) for role in ROLES}
