"""Scrubbing personal data out of text on its way to the user.

The last layer, and the one that assumes the others leaked. Sources are scraped
web pages: an author's email, a contact number in a footer, an identifier pasted
into a forum post all arrive as ordinary snippet text, get quoted by the analyst,
and land in the report. Nothing upstream is in a position to notice.

Four types, ordered by how explicit their signal is, because an unseparated
12-digit run is genuinely ambiguous between an identifier and a phone number. Each
becomes a labelled placeholder, not a blanket mark, so a reader of the report can
tell what kind of thing was removed.

The pressure here runs opposite to `injection`. A research report is full of
numbers — years, sample sizes, arXiv ids, ISBNs — and a greedy phone pattern
quietly corrupts data. So the number patterns require real formatting: a bare
10-digit run is left alone.
"""

import re

from research_system.settings import GuardrailsSettings

EMAIL = "[EMAIL]"
PHONE = "[PHONE]"
SSN = "[SSN]"
AADHAAR = "[AADHAAR]"

# Order matters: each pattern runs on what the ones above it left behind, and it
# is strongest signal first, not narrowest pattern first. `+442071234567` is a `+`
# followed by exactly twelve digits, so Aadhaar would claim it and mislabel it —
# but a country code is explicit where a digit count is only suggestive. Phones
# therefore go first, which is free in reverse: every phone branch demands a marker
# (`+`, parens, 3-3-4 separators) that no Aadhaar grouping has.
_SUBSTITUTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), EMAIL),
    # US SSN: 3-2-4, a grouping nothing else here uses.
    (re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"), SSN),
    (
        re.compile(
            # A country code makes the intent explicit, so the grouping after it
            # can be loose enough for international formats.
            r"\+\d{1,3}[-\s]?(?:\(\d{1,4}\)[-\s]?)?\d{2,5}[-\s]?\d{3,5}(?:[-\s]?\d{2,4})?"
            # Otherwise the formatting has to carry it: (555) 123-4567, 555-123-4567.
            r"|\(\d{3}\)[-\s]?\d{3}[-\s]?\d{4}"
            r"|\b\d{3}[-\s]\d{3}[-\s]\d{4}\b"
        ),
        PHONE,
    ),
    # Aadhaar: 12 digits, grouped 4-4-4 or run together. A real one never starts
    # with 0 or 1, and that rule is what keeps this off large plain numbers.
    (re.compile(r"\b[2-9]\d{3}[-\s]?\d{4}[-\s]?\d{4}\b"), AADHAAR),
)


def scrub_pii(text: str, settings: GuardrailsSettings) -> str:
    """Replace every recognised identifier in `text` with a labelled placeholder."""
    if not settings.pii_enabled:
        return text
    for pattern, placeholder in _SUBSTITUTIONS:
        text = pattern.sub(placeholder, text)
    return text
