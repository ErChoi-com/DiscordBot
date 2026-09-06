"""Capacity scaling where it actually reaches behaviour: the semantic-matching
memory gate, and the launcher's refusal to act on a recycled pid."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from services import capacity, job_service

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402  (repo-root launcher, needs the path insert above)


@pytest.fixture(autouse=True)
def _reset_semantic_caches():
    """semantic_plugin_available is lru_cached, so a verdict from one test would
    otherwise leak into the next."""
    job_service.semantic_plugin_available.cache_clear()
    yield
    job_service.semantic_plugin_available.cache_clear()


def _fake_memory(monkeypatch, gb: float | None):
    monkeypatch.setattr(
        capacity, "memory_limit_bytes",
        lambda: None if gb is None else int(gb * 1024 ** 3),
    )


# ---------------------------------------------------------------------------
# Semantic model memory gate
# ---------------------------------------------------------------------------

def test_semantic_matching_is_disabled_on_a_low_memory_host(monkeypatch, capsys):
    """The torch runtime is the largest fixed allocation in the process; on a
    1 GB box loading it is the difference between running and being OOM-killed."""
    monkeypatch.setattr(job_service, "SEMANTIC_PLUGIN_ENABLED", True)
    _fake_memory(monkeypatch, 1.0)

    assert job_service.semantic_plugin_available() is False
    assert "Semantic matching disabled" in capsys.readouterr().out


def test_semantic_gate_does_not_fire_on_a_host_with_headroom(monkeypatch, capsys):
    """With enough memory the gate must be transparent: the decision passes
    through to the package-import check instead of being short-circuited.

    Deliberately does not import sentence_transformers -- pulling in torch costs
    ~50s and would make this suite painful to run.
    """
    monkeypatch.setattr(job_service, "SEMANTIC_PLUGIN_ENABLED", True)
    _fake_memory(monkeypatch, 32.0)

    probed: list[float] = []
    real_can_afford = capacity.can_afford
    monkeypatch.setattr(
        capacity, "can_afford",
        lambda gb: probed.append(gb) or real_can_afford(gb),
    )
    # Stand in for the package check so the verdict is attributable to the gate.
    monkeypatch.setitem(sys.modules, "sentence_transformers", object())

    assert job_service.semantic_plugin_available() is True
    assert probed == [job_service.SEMANTIC_MIN_MEMORY_GB]
    assert "Semantic matching disabled" not in capsys.readouterr().out


def test_semantic_gate_is_not_consulted_when_the_feature_is_off(monkeypatch):
    """An explicit disable short-circuits before any hardware probing."""
    monkeypatch.setattr(job_service, "SEMANTIC_PLUGIN_ENABLED", False)

    def _boom():
        raise AssertionError("memory must not be probed when disabled")

    monkeypatch.setattr(capacity, "memory_limit_bytes", _boom)
    assert job_service.semantic_plugin_available() is False


def test_undetectable_memory_does_not_disable_semantic_matching(monkeypatch):
    monkeypatch.setattr(job_service, "SEMANTIC_PLUGIN_ENABLED", True)
    _fake_memory(monkeypatch, None)

    # Must not be blocked by the gate; only a missing package may block it.
    assert capacity.can_afford(job_service.SEMANTIC_MIN_MEMORY_GB) is True


# ---------------------------------------------------------------------------
# Launcher: pid reuse
# ---------------------------------------------------------------------------

def test_launcher_ignores_a_lock_whose_pid_was_recycled(monkeypatch, tmp_path, capsys):
    """Linux recycles pids within hours; --forcerun must not kill whatever now
    holds a pid recorded in a stale lock file."""
    lock = tmp_path / ".bot.lock"
    lock.write_text('{"pid": 4242}', encoding="utf-8")
    monkeypatch.setattr(run, "OWNERSHIP_FILES", (lock,))
    monkeypatch.setattr(run.platform_support, "is_process_alive", lambda pid, system=None: True)
    monkeypatch.setattr(
        run.platform_support, "process_cmdline",
        lambda pid, system=None: "/usr/sbin/nginx -g daemon off;",
    )

    assert run._running_pid() is None
    assert "not this bot" in capsys.readouterr().out


def test_launcher_accepts_a_pid_whose_cmdline_is_the_bot(monkeypatch, tmp_path):
    lock = tmp_path / ".bot.lock"
    lock.write_text('{"pid": 4242}', encoding="utf-8")
    monkeypatch.setattr(run, "OWNERSHIP_FILES", (lock,))
    monkeypatch.setattr(run.platform_support, "is_process_alive", lambda pid, system=None: True)
    monkeypatch.setattr(
        run.platform_support, "process_cmdline",
        lambda pid, system=None: "/usr/bin/python3 /opt/discordbot/src/app.py",
    )

    assert run._running_pid() == 4242


def test_launcher_trusts_the_lock_when_cmdline_is_unreadable(monkeypatch, tmp_path):
    """Refusing to ever recover would be worse than a rare wrong guess, so an
    unreadable cmdline falls back to trusting the lock file."""
    lock = tmp_path / ".bot.lock"
    lock.write_text('{"pid": 4242}', encoding="utf-8")
    monkeypatch.setattr(run, "OWNERSHIP_FILES", (lock,))
    monkeypatch.setattr(run.platform_support, "is_process_alive", lambda pid, system=None: True)
    monkeypatch.setattr(run.platform_support, "process_cmdline", lambda pid, system=None: None)

    assert run._running_pid() == 4242


def test_launcher_skips_a_dead_pid(monkeypatch, tmp_path):
    lock = tmp_path / ".bot.lock"
    lock.write_text('{"pid": 4242}', encoding="utf-8")
    monkeypatch.setattr(run, "OWNERSHIP_FILES", (lock,))
    monkeypatch.setattr(run.platform_support, "is_process_alive", lambda pid, system=None: False)
    assert run._running_pid() is None


def test_launcher_reads_both_lock_formats(tmp_path):
    """app.py writes .bot.lock as JSON and .bot.pid as a bare int."""
    as_json = tmp_path / ".bot.lock"
    as_json.write_text('{"pid": 17}', encoding="utf-8")
    as_int = tmp_path / ".bot.pid"
    as_int.write_text("23\n", encoding="utf-8")

    assert run._read_pid(as_json) == 17
    assert run._read_pid(as_int) == 23
    assert run._read_pid(tmp_path / "absent") is None

    garbage = tmp_path / "garbage"
    garbage.write_text("not a pid", encoding="utf-8")
    assert run._read_pid(garbage) is None
