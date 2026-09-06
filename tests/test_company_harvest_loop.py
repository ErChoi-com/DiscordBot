"""Discovery has to start itself.

Finding ATS company boards is the stage this whole fleet is built to feed, and
it was the one stage nothing ever ran. The weekly cron in
.github/workflows/ats-harvest.yml cannot fire: GitHub evaluates `schedule:`
only from the default branch, and that workflow lives on a feature branch --
the same reason the workflow's own comment gives for adding a push trigger,
applied to `schedule:` too. So a harvest happened when someone pushed one of
three paths, or when a person ran it. Measured while writing this: the local
fleet carries 14,278 slugs that have never gone back upstream, and
data/ats_discovery_state.json -- the previous attempt at automating this -- was
written once on 2026-06-15 and is referenced by nothing since.

A step that works perfectly and runs only when remembered is the defect. These
tests pin the three things that make running it daily both safe and affordable:
it is gated so most mornings cost no request, it is bounded so it cannot delay
the day's first scrape, and it is ahead of validation so what it finds today is
probed today.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import scheduler_labels  # noqa: E402
from watchers.manager import ATS_HARVEST_BUDGET_S, WatcherManager  # noqa: E402


class _Config:
    def __init__(self, base_dir):
        self.base_dir = base_dir


def _manager(tmp_path):
    mgr = WatcherManager.__new__(WatcherManager)
    mgr.config = _Config(tmp_path)
    return mgr


def _loop_source() -> str:
    return " ".join(inspect.getsource(WatcherManager._run_ats_scrape_loop).split())


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Record the command without sweeping a single index."""
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
    (tmp_path / "scripts" / "harvest_ats.py").write_text("", encoding="utf-8")
    return seen


# -- it runs at all ---------------------------------------------------------

def test_the_harvester_is_actually_invoked(run, tmp_path):
    asyncio.run(_manager(tmp_path)._harvest_companies())
    assert run["argv"] is not None
    assert run["argv"][1].endswith("harvest_ats.py")


def test_a_missing_script_is_not_an_error(run, tmp_path):
    """A checkout without the harvester still has to scrape."""
    (tmp_path / "scripts" / "harvest_ats.py").unlink()
    asyncio.run(_manager(tmp_path)._harvest_companies())
    assert run["argv"] is None


# -- what makes a daily schedule affordable ---------------------------------

def test_it_asks_only_for_crawls_it_has_not_swept(run, tmp_path):
    """Without this the hook re-sweeps the same crawl every morning for the
    month it stays newest, which is what would make a daily schedule
    unaffordable and send discovery back to being run by hand."""
    asyncio.run(_manager(tmp_path)._harvest_companies())
    assert "--only-new-crawls" in run["argv"]


def test_it_is_bounded_by_the_declared_budget(run, tmp_path):
    """It sits ahead of validation, so an unbounded harvest would push the
    day's first scrape back by its own runtime plus validation's."""
    asyncio.run(_manager(tmp_path)._harvest_companies())
    argv = run["argv"]
    assert "--total-budget-seconds" in argv
    assert argv[argv.index("--total-budget-seconds") + 1] == str(ATS_HARVEST_BUDGET_S)


def test_it_sweeps_the_bulk_index_and_not_wayback(run, tmp_path):
    """Wayback is an order of magnitude slower -- measured at 45 minutes
    against 7 for the same fleet -- and the one platform that needs it is
    better served by a CI run with hours than by a rollover with fifteen
    minutes."""
    argv_index = "--index"
    asyncio.run(_manager(tmp_path)._harvest_companies())
    argv = run["argv"]
    assert argv[argv.index(argv_index) + 1] == "ccbulk"
    assert "wayback" not in argv


def test_it_names_no_platform(run, tmp_path):
    """Same reason the validator passes none: the next platform added gets
    harvested because it exists, not because someone edited this call."""
    asyncio.run(_manager(tmp_path)._harvest_companies())
    assert "--platform" not in run["argv"]


def test_it_runs_at_background_tier_under_its_own_label(run, tmp_path):
    """A user's command must outrank discovery, and pooling the mornings this
    returns instantly with the mornings it really sweeps would give the
    scheduler a median that describes neither."""
    asyncio.run(_manager(tmp_path)._harvest_companies())
    assert run["label"] == scheduler_labels.ATS_COMPANY_HARVEST


# -- failure must not cost the scrape ---------------------------------------

def test_a_failing_harvest_does_not_raise(run, tmp_path):
    """Exit 1 means the index or the queries broke, not that the fleet is
    gone. A harvest is additive, so yesterday's companies are all still
    there and the scrape about to run is unaffected."""
    run["rc"] = 1
    asyncio.run(_manager(tmp_path)._harvest_companies())  # must not raise


def test_a_harvest_that_cannot_start_does_not_raise(run, tmp_path):
    """Three more hooks and the day's first scrape are queued behind this."""
    run["boom"] = True
    asyncio.run(_manager(tmp_path)._harvest_companies())  # must not raise


# -- where it sits in the rollover ------------------------------------------

def test_the_daily_branch_runs_it():
    """Source-level pin: the branch lives inside a `while True` no test can
    drive, and a harvester nothing calls is the exact defect being fixed."""
    assert "await self._harvest_companies()" in _loop_source()


def test_it_runs_after_the_company_sync():
    """The sync pulls what CI and upstream already found. Harvesting first
    would sweep, then immediately have its output unioned with a list it could
    have started from."""
    compact = _loop_source()
    assert (compact.index("_refresh_company_lists()")
            < compact.index("_harvest_companies()"))


def test_it_runs_before_validation():
    """The whole point of the ordering. validate_ats_slugs reads
    data/ats_harvest as well as data/ats_companies, so a board harvested this
    morning is probed this morning instead of costing a scrape cycle's worth
    of requests to a board that may not exist."""
    compact = _loop_source()
    assert (compact.index("_harvest_companies()")
            < compact.index("_validate_company_slugs()"))


def test_it_is_not_wired_into_every_cycle():
    """Four sweeps a day is four times the index traffic for an answer that
    changes roughly monthly."""
    src = inspect.getsource(WatcherManager._run_ats_scrape_loop)
    lines = src.splitlines()
    call = next(i for i, l in enumerate(lines) if "_harvest_companies()" in l)
    guard = next(i for i, l in enumerate(lines) if "if rolled_over:" in l)
    assert guard < call
    indent = lambda l: len(l) - len(l.lstrip())
    assert indent(lines[call]) > indent(lines[guard])
