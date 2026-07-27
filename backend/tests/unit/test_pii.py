import pytest

from research_system.guardrails.pii import AADHAAR, EMAIL, PHONE, SSN, scrub_pii
from research_system.settings import GuardrailsSettings, get_settings


@pytest.fixture
def guardrails() -> GuardrailsSettings:
    return get_settings().guardrails


@pytest.mark.parametrize(
    "text",
    [
        "write to alice@example.com for the dataset",
        "contact: first.last+tag@sub.domain.co.uk",
        "a.b@c.io",
    ],
)
def test_emails_are_scrubbed(text: str, guardrails: GuardrailsSettings) -> None:
    assert "@" not in scrub_pii(text, guardrails)
    assert EMAIL in scrub_pii(text, guardrails)


@pytest.mark.parametrize(
    "text",
    [
        "call +1 555 123 4567 for details",
        "reachable on +91 98765 43210",
        "the number is (555) 123-4567",
        "dial 555-123-4567 now",
        "office: +442071234567",
    ],
)
def test_phone_numbers_are_scrubbed(text: str, guardrails: GuardrailsSettings) -> None:
    assert PHONE in scrub_pii(text, guardrails)


def test_ssn_is_scrubbed(guardrails: GuardrailsSettings) -> None:
    assert scrub_pii("ssn 123-45-6789 on file", guardrails) == f"ssn {SSN} on file"


@pytest.mark.parametrize(
    "text",
    [
        "aadhaar 2345 6789 0123 verified",
        "aadhaar 2345-6789-0123 verified",
        "aadhaar 234567890123 verified",
    ],
)
def test_aadhaar_is_scrubbed_in_every_grouping(text: str, guardrails: GuardrailsSettings) -> None:
    assert scrub_pii(text, guardrails) == f"aadhaar {AADHAAR} verified"


def test_an_aadhaar_never_starts_with_zero_or_one(guardrails: GuardrailsSettings) -> None:
    """The leading-digit rule is what keeps this pattern off large plain numbers."""
    assert scrub_pii("a count of 123456789012 items", guardrails) == (
        "a count of 123456789012 items"
    )


def test_an_unseparated_international_number_is_a_phone_not_an_aadhaar(
    guardrails: GuardrailsSettings,
) -> None:
    """`+44` plus 10 digits is twelve digits starting with 4 — Aadhaar's shape too.

    The country code decides it. Ordering Aadhaar first mislabels this as
    `+[AADHAAR]`, which both leaks the `+` and names the wrong thing.
    """
    assert scrub_pii("office: +442071234567", guardrails) == f"office: {PHONE}"


def test_all_four_types_combined(guardrails: GuardrailsSettings) -> None:
    text = (
        "Contact Dr. Rao at rao@lab.example.org or +91 98765 43210. "
        "SSN 123-45-6789, aadhaar 2345 6789 0123."
    )
    scrubbed = scrub_pii(text, guardrails)
    assert scrubbed == (f"Contact Dr. Rao at {EMAIL} or {PHONE}. SSN {SSN}, aadhaar {AADHAAR}.")


@pytest.mark.parametrize(
    "text",
    [
        "transformer models reached 92.4% accuracy on the 2024 benchmark",
        "see arXiv:2501.12345 and the ISBN 978-3-16-148410-0 edition",
        "the study sampled 1500 participants across 12 countries",
        "revenue grew from 4.2 billion to 5.8 billion between 2019 and 2024",
        "figures 3-5 and tables 10-12 report the ablation",
        "",
    ],
)
def test_clean_research_text_is_returned_unchanged(
    text: str, guardrails: GuardrailsSettings
) -> None:
    assert scrub_pii(text, guardrails) == text


def test_a_bare_ten_digit_run_is_left_alone(guardrails: GuardrailsSettings) -> None:
    """Unformatted digits are as likely to be a population count as a phone."""
    assert scrub_pii("a market of 5551234567 users", guardrails) == ("a market of 5551234567 users")


def test_repeated_occurrences_are_all_scrubbed(guardrails: GuardrailsSettings) -> None:
    scrubbed = scrub_pii("a@x.com, b@y.com, c@z.com", guardrails)
    assert scrubbed == f"{EMAIL}, {EMAIL}, {EMAIL}"


def test_scrubbing_can_be_disabled(guardrails: GuardrailsSettings) -> None:
    off = guardrails.model_copy(update={"pii_enabled": False})
    assert scrub_pii("alice@example.com", off) == "alice@example.com"


def test_scrubbing_is_idempotent(guardrails: GuardrailsSettings) -> None:
    """Placeholders must not themselves look like PII to a second pass."""
    once = scrub_pii("mail alice@example.com or call 555-123-4567", guardrails)
    assert scrub_pii(once, guardrails) == once
