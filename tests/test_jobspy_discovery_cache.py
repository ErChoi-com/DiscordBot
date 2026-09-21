"""Finding the JobSpy interpreter must cost child processes once, not per scrape.

Discovery probes `py -3.11` through `py -3.14` on Windows and then runs an
`import jobspy` check per candidate -- up to a dozen children. It ran on every
call of scrape_job_postings, so every channel re-derived the same answer on
every refresh, and on Windows each probe opened a console window.

Counted rather than asserted structurally: an lru_cache decorator is easy to
add and just as easy to drop, and the symptom of dropping it is a slow drizzle
of processes that no other test notices.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import job_service


@pytest.fixture(autouse=True)
def _clear_caches():
    job_service.jobspy_python_executable.cache_clear()
    job_service.jobspy_runtime_metadata.cache_clear()
    yield
    job_service.jobspy_python_executable.cache_clear()
    job_service.jobspy_runtime_metadata.cache_clear()


def _count_spawns(monkeypatch) -> list[int]:
    calls = [0]
    real = subprocess.run

    def counting(*args, **kwargs):
        calls[0] += 1
        return real(
            [sys.executable, "-c", "raise SystemExit(1)"],
            capture_output=True, check=False,
        )

    monkeypatch.setattr(subprocess, "run", counting)
    return calls


def test_interpreter_discovery_spawns_nothing_after_the_first_call(monkeypatch):
    calls = _count_spawns(monkeypatch)

    job_service.jobspy_python_executable(None)
    first = calls[0]
    assert first > 0, "discovery spawned nothing at all; the probe never ran"

    for _ in range(9):
        job_service.jobspy_python_executable(None)

    assert calls[0] == first, (
        f"discovery re-probed: {calls[0] - first} extra child processes across "
        "9 repeat calls, so every scrape pays the cost again"
    )


def test_a_different_configured_interpreter_is_probed_separately(monkeypatch):
    """The cache is keyed on the argument, so it must not answer for another exe."""
    calls = _count_spawns(monkeypatch)

    job_service.jobspy_python_executable(None)
    after_default = calls[0]
    job_service.jobspy_python_executable("/some/other/python")

    assert calls[0] > after_default, "a different interpreter reused the cached answer"
