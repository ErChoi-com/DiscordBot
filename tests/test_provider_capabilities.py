"""Per-model capability overrides: swapping a provider slot's model must
re-tune limits (timeouts especially) without touching other providers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.resumes.resume import (
    PROVIDER_CAPABILITIES,
    resolve_provider_capabilities,
)


def test_nemotron_override_applies_by_model_prefix() -> None:
    caps = resolve_provider_capabilities(
        "openrouter", "nvidia/nemotron-3-super-120b-a12b:free"
    )
    assert caps is not None
    # Observed live generations run 75-250s; the 45s slot default would time
    # out nearly every call.
    assert caps.request_timeout_seconds >= 200


def test_non_override_model_keeps_provider_default() -> None:
    caps = resolve_provider_capabilities("openrouter", "openai/gpt-oss-120b")
    assert caps == PROVIDER_CAPABILITIES["openrouter"]


def test_no_model_falls_back_to_provider_default() -> None:
    assert resolve_provider_capabilities("groq") == PROVIDER_CAPABILITIES["groq"]
    assert resolve_provider_capabilities("gemini", "gemini-3.5-flash") == (
        PROVIDER_CAPABILITIES["gemini"]
    )


def test_unknown_provider_returns_none() -> None:
    assert resolve_provider_capabilities("nonexistent", "some/model") is None
