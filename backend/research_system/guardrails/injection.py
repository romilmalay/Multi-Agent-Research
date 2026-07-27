"""The first guardrail: what a query has to look like to enter the pipeline.

Size, because an empty query still costs a planner call and an oversized one eats
the token budget in its first prompt. Then injection, because the query is
interpolated into every prompt downstream, so one query that reads as an
instruction can rewrite seven agents' behaviour.

Regex catches the copied payload, not a novel rewording — the ceiling of a pattern
list, and why three more layers sit behind it. Where a pattern had to choose it
chose the false positive: a rejected question is a retry, an accepted injection is
a hijacked run.
"""

import re

from research_system.errors import InvalidQueryError
from research_system.settings import GuardrailsSettings

# Six families, in the order they are tried. The names are part of the interface:
# they appear in the rejection message and in the logs.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        # The classic: cancel what came before. Either branch is enough — naming
        # an instruction noun, or telling the model to drop what it was told.
        re.compile(
            r"\b(?:ignore|disregard|forget|override|discard)\b[^.\n]{0,40}?"
            r"\b(?:instructions?|prompts?|rules?|directives?|guidelines?|constraints?)\b"
            r"|\b(?:ignore|disregard|forget|override|discard)\s+(?:all\s+|any\s+)?(?:of\s+)?"
            r"(?:the\s+)?(?:previous|prior|preceding|earlier|above|foregoing|everything)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_hijack",
        # Reassign who the model is. "act as" alone is far too common in real
        # questions ("how do enzymes act as catalysts"), so the verbs only count
        # when what follows is a persona with the rules removed.
        re.compile(
            r"\byou\s+(?:are|'re)\s+(?:now|no\s+longer)\b"
            r"|\b(?:from\s+now\s+on|starting\s+now)\b[,\s]+you\b"
            r"|\b(?:act|behave|respond|roleplay)\s+(?:as|like)\b[^.\n]{0,20}?"
            r"\b(?:unrestricted|unfiltered|uncensored|jailbroken|dan|evil|rogue)\b"
            r"|\bpretend\s+(?:that\s+)?you\s+(?:are|were|have|can|do)\b"
            r"|\byou\s+are\s+(?:an?\s+)?(?:unrestricted|unfiltered|uncensored|jailbroken)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_extraction",
        # Make the model quote its own configuration back. The possessive is what
        # separates this from "how are system prompts written", which is research.
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|display|disclose|reproduce|list|"
            r"tell\s+me|what\s+(?:is|are|was|were))\b[^.\n]{0,30}?"
            r"\b(?:your|the)\s+(?:system|initial|original|hidden|developer|above)\s+"
            r"(?:prompt|instructions?|message|rules?)"
            r"|\brepeat\s+(?:everything|all|the\s+text|the\s+words)\s+(?:above|before)\b"
            r"|\bwhat\s+(?:were|was)\s+you\s+(?:told|instructed|programmed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_escape",
        # Forge the framing rather than argue with it: chat control tokens, role
        # tags, or a fake turn header that makes the rest read as a new message.
        re.compile(
            r"<\|[^|>]*\|>"
            r"|\[/?(?:INST|SYS|SYSTEM)\]"
            r"|</?(?:system|assistant|user|human)>"
            r"|(?:^|\n)\s*(?:system|assistant|ai|human)\s*:"
            r"|#{2,}\s*(?:system|new\s+)?instructions?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "obfuscation",
        # Smuggle the payload past a pattern list by encoding it. The long-token
        # branch is deliberate: an opaque 40-character blob is not a question, and
        # a real query that happens to carry one can be reworded.
        re.compile(
            r"\b(?:base64|rot13|hex|url)[\s-]*(?:encoded?|encoding|decoded?)\b"
            r"|\bdecode\s+(?:the\s+|this\s+|these\s+|following)\b"
            r"|\b[A-Za-z0-9+/]{40,}={0,2}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "restriction_bypass",
        # Ask for the version of the model with the limits switched off.
        re.compile(
            r"\b(?:bypass|circumvent|disable|remove|lift|turn\s+off)\b[^.\n]{0,30}?"
            r"\b(?:safety|filters?|guardrails?|restrictions?|limitations?|"
            r"moderation|polic(?:y|ies))\b"
            r"|\b(?:developer|debug|god|admin|sudo|dan)\s+mode\b"
            r"|\b(?:with\s+)?no\s+(?:restrictions?|filters?|limits?|rules?)\b"
            r"|\b(?:without|ignoring)\s+(?:any\s+)?"
            r"(?:restrictions?|filters?|limitations?|safety)\b",
            re.IGNORECASE,
        ),
    ),
)


def detect_injection(text: str) -> str | None:
    """The name of the first injection family `text` matches, else `None`."""
    for name, pattern in _PATTERNS:
        if pattern.search(text):
            return name
    return None


def validate_query(query: str, settings: GuardrailsSettings) -> str:
    """Return `query` with its whitespace collapsed, or raise `InvalidQueryError`.

    Size is measured on the collapsed form so padding cannot fake length; injection
    is scanned on the raw text, because collapsing newlines is what hides a forged
    turn header.
    """
    cleaned = " ".join(query.split())
    if len(cleaned) < settings.query_min_chars:
        raise InvalidQueryError(
            f"query is {len(cleaned)} characters, "
            f"below the {settings.query_min_chars} character minimum"
        )
    if len(cleaned) > settings.query_max_chars:
        raise InvalidQueryError(
            f"query is {len(cleaned)} characters, "
            f"above the {settings.query_max_chars} character maximum"
        )
    if settings.injection_enabled:
        family = detect_injection(query)
        if family is not None:
            raise InvalidQueryError(f"query matches a prompt-injection pattern: {family}")
    return cleaned
