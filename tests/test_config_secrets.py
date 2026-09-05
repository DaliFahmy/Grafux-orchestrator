"""Settings sanitizes provider keys before anything tries to authenticate with them.

A key reaches this service through a Render dashboard field or a .env line, and
both of those add characters the SDKs then send verbatim. The failure that
follows is a 401 indistinguishable from a revoked key -- which is how a working
key can look broken for a day.
"""

from __future__ import annotations

from app.config import Settings


def _key(value: str) -> str:
    return Settings(anthropic_api_key=value).anthropic_api_key


def test_a_pasted_key_keeps_neither_whitespace_nor_quotes():
    assert _key("  sk-ant-abc123  ") == "sk-ant-abc123"
    assert _key('"sk-ant-abc123"') == "sk-ant-abc123"
    assert _key("'sk-ant-abc123'") == "sk-ant-abc123"
    assert _key("sk-ant-abc123\n") == "sk-ant-abc123"


def test_every_provider_key_is_cleaned_not_just_anthropic():
    """Fixing one vendor is the asymmetry that comes back next quarter."""
    settings = Settings(
        openai_api_key=' "sk-openai" ',
        gemini_api_key="gem\n",
        tavily_api_key="'tvly'",
    )
    assert settings.openai_api_key == "sk-openai"
    assert settings.gemini_api_key == "gem"
    assert settings.tavily_api_key == "tvly"


def test_a_key_that_could_never_be_an_http_header_reads_as_absent():
    """An embedded newline raises deep in httpx; empty takes the clean fallback."""
    assert _key("sk-ant-\nabc") == ""


def test_an_unfamiliar_key_shape_is_still_used():
    """Pins the decision NOT to gate on format.

    Key shapes are the vendor's to change. Rejecting a legitimate future one
    would be a silent, permanent downgrade to another provider -- strictly worse
    than a 401, which at least names itself and now falls back with a note.
    """
    assert _key("some-future-token-format") == "some-future-token-format"


def test_an_unset_key_stays_empty_so_the_missing_key_fallback_still_fires():
    assert _key("") == ""
    assert _key("   ") == ""
