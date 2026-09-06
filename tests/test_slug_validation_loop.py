"""The confirmed-live store has to maintain itself.

Harvesting is optimistic: an archived URL proves a board existed when it was
crawled, not that it exists now, and measured live rates for fresh slugs run
from 87% down to 11%. `scripts/validate_ats_slugs.py` is what closes that --
it writes the dead marks the scraper reads to stop asking, and the
confirmed-live store the monitor samples from.

Nothing had ever run it. Every file under `data/ats_checked` exists because
someone typed the command, and the proof is what was missing: the two platforms
added most recently had no confirmed-live file at all, so the monitor reported
them as unwatchable rather than as working. A store maintained by hand is a
store that stops being maintained the day attention moves on.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from watchers.manager import ATS_VALIDATION_BUDGET_S, WatcherManager  # noqa: E402


class _Config:
    def __init__(self, base_dir):
        self.base_dir = base_dir


def _code_of(fn) -> str:
    """The function's source with its docstring removed.

    Written this way on purpose: the docstring explains *why* no --platform is
    passed, and a naive substring check over the whole source reads that
    explanation as the thing it forbids. That is the exact failure the orphan
    audit had -- documenting a rule made the check for it stop working.
    """
    import ast

    tree = ast.parse(inspect.getsource(fn).lstrip())
    body = tree.body[0].body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return " ".join(" ".join(ast.unparse(n) for n in body).split())


def _manager(tmp_path):
    mgr = WatcherManager.__new__(WatcherManager)
    mgr.config = _Config(tmp_path)
    return mgr


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Record the command without probing a single board."""
    seen = {"argv": None, "label": None, "rc": 0, "stdout": "", "boom": False}

    async def _to_thread(self, fn, *args, label=None, **kwargs):
        seen["label"] = label
        return fn(*args, **kwargs)

    def _subprocess_run(argv, **_kwargs):
        seen["argv"] = list(argv)
        if seen["boom"]:
            raise OSError("no python on PATH")
        return type("R", (), {"returncode": seen["rc"],
                              "stdout": seen["stdout"], "stderr": ""})()

    monkeypatch.setattr(WatcherManager, "_tracked_to_thread", _to_thread)
    monkeypatch.setattr("subprocess.run", _subprocess_run)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "validate_ats_slugs.py").write_text("", encoding="utf-8")
    return seen


# ── it runs at all ──────────────────────────────────────────────────────────

def test_the_validator_is_actually_invoked(run, tmp_path):
    asyncio.run(_manager(tmp_path)._validate_company_slugs())
    assert run["argv"] is not None
    assert run["argv"][1].endswith("validate_ats_slugs.py")


def test_it_runs_under_its_own_cost_label(run, tmp_path):
    """A bounded network sweep of the whole fleet, pooled with the git fetch
    beside it, would skew what the scheduler arbitrates on."""
    asyncio.run(_manager(tmp_path)._validate_company_slugs())
    assert run["label"] == "ats_slug_validation"


def test_it_is_bounded_by_a_wall_clock_budget(run, tmp_path):
    """Unbounded it is tens of thousands of probes, and it sits in front of the
    day's first scrape."""
    asyncio.run(_manager(tmp_path)._validate_company_slugs())
    argv = run["argv"]
    assert "--total-budget-seconds" in argv
    assert argv[argv.index("--total-budget-seconds") + 1] == str(ATS_VALIDATION_BUDGET_S)


def test_the_budget_leaves_the_day_its_scrapes():
    """Four scrapes a day means six hours of room; the sweep must be a small
    bite out of one of them, not a competitor for the day."""
    assert 0 < ATS_VALIDATION_BUDGET_S <= 3600


# ── the reason it passes no --platform ──────────────────────────────────────

def test_no_platform_is_named_so_a_new_one_is_picked_up_automatically():
    """This is the whole failure being fixed. oracle and personio had no
    confirmed-live file because adding a platform did not add it to anything
    that runs; a hard-coded list here would reproduce that exactly.
    """
    assert "--platform" not in _code_of(WatcherManager._validate_company_slugs)


def test_a_new_platform_is_validated_without_touching_this_module(run, tmp_path):
    asyncio.run(_manager(tmp_path)._validate_company_slugs())
    assert not any(a.startswith("--platform") for a in run["argv"])


# ── failure is not an outage ────────────────────────────────────────────────

def test_a_nonzero_exit_keeps_the_stores_and_does_not_raise(run, tmp_path, capsys):
    run["rc"] = 1
    asyncio.run(_manager(tmp_path)._validate_company_slugs())
    out = capsys.readouterr().out
    assert "exited 1" in out
    assert "keeping the dead and confirmed-live stores" in out


def test_a_crash_does_not_stop_the_scrape(run, tmp_path, capsys):
    run["boom"] = True
    asyncio.run(_manager(tmp_path)._validate_company_slugs())  # must not raise
    assert "could not validate company slugs" in capsys.readouterr().out


def test_a_missing_script_is_not_an_error(run, tmp_path, capsys):
    """A checkout without the scripts directory is a smaller install, not a
    broken one."""
    (tmp_path / "scripts" / "validate_ats_slugs.py").unlink()
    asyncio.run(_manager(tmp_path)._validate_company_slugs())
    assert run["argv"] is None
    assert capsys.readouterr().out == ""


def test_what_it_did_is_reported(run, tmp_path, capsys):
    """A sweep that probed nothing looks identical to one that worked."""
    run["stdout"] = "[validate] lever: nothing to probe\n[validate] 412 live, 39 dead"
    asyncio.run(_manager(tmp_path)._validate_company_slugs())
    assert "412 live, 39 dead" in capsys.readouterr().out


# ── it actually recurs ──────────────────────────────────────────────────────

def test_the_daily_branch_runs_it():
    """Source-level pin: the branch is inside a `while True` no test can drive,
    and a validator nothing calls is the exact defect being fixed."""
    compact = " ".join(
        inspect.getsource(WatcherManager._run_ats_scrape_loop).split()
    )
    assert "await self._validate_company_slugs()" in compact


def test_it_runs_after_the_company_sync():
    """Slugs arrive from the sync. Probing first validates yesterday's fleet
    and leaves this morning's arrivals to cost a full cycle of requests."""
    compact = " ".join(
        inspect.getsource(WatcherManager._run_ats_scrape_loop).split()
    )
    assert (compact.index("_refresh_company_lists()")
            < compact.index("_validate_company_slugs()"))


def test_it_is_not_wired_into_every_cycle():
    """Four sweeps a day is four times the probes for an answer that changes
    on the scale of days."""
    lines = inspect.getsource(WatcherManager._run_ats_scrape_loop).splitlines()
    indent = lambda l: len(l) - len(l.lstrip())
    guard = next(i for i, l in enumerate(lines) if "if rolled_over:" in l)
    call = next(i for i, l in enumerate(lines) if "_validate_company_slugs()" in l)

    assert call > guard, "it runs before the day-rollover guard"
    # Every line between the guard and the call must stay inside the guard's
    # block. Comparing the call's indent to the guard's alone is not enough:
    # a branch that has been emptied out still leaves an `if rolled_over:`
    # line above a call that no longer sits inside it.
    body = [l for l in lines[guard + 1:call + 1] if l.strip()]
    assert all(indent(l) > indent(lines[guard]) for l in body), (
        "it is not inside the day-rollover branch")


def test_it_does_not_probe_on_the_event_loop():
    """Minutes of network probing inside the loop stalls every watcher and
    every command for the duration."""
    compact = " ".join(
        inspect.getsource(WatcherManager._validate_company_slugs).split()
    )
    assert "await self._tracked_to_thread( _validate" in compact
