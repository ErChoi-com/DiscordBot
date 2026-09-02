"""Checks that the harvest workflow invokes what the scripts provide.

Every other test here covers what the harvester and validator *do*. Nothing
covered whether CI actually calls them correctly, and that gap cost four
silent failures in one week: --budget-seconds, --audit-sample and the
_crawls.json restore were all described in commit messages, committed as done,
and never written to the file. Each was a multi-line replacement over a shell
block that failed to match without saying so.

The failure mode is specific and worth naming: a feature exists, its unit tests
pass, the workflow looks plausible, and the step simply never passes the flag.
Nothing errors. The only way to notice is to read the YAML and compare it
against what the scripts offer -- which is what this file does.

These are deliberately assertions about wiring, not behaviour. They parse the
workflow and check the commands; they never run it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import harvest_ats as hc  # noqa: E402
import validate_ats_slugs as v  # noqa: E402

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ats-harvest.yml"


@pytest.fixture(scope="module")
def steps() -> dict[str, str]:
    """Every run-step's shell body, keyed by step name."""
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return {s["name"]: s.get("run", "") for s in data["jobs"]["harvest"]["steps"]}


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _joined(step: str) -> str:
    """Shell body with line continuations folded, so a flag split across lines
    still matches."""
    return " ".join(step.replace("\\\n", " ").split())


# --------------------------------------------------------------------------
# The scripts are called with the options that exist for them
# --------------------------------------------------------------------------

def test_validation_passes_a_time_budget(steps):
    """Without it the audit and working passes are unbounded, and one slow
    platform can consume the whole job."""
    assert "--budget-seconds" in _joined(steps["Validate"])


def test_validation_requests_the_audit_sample(steps):
    """The summary prints an audit-derived 'true live rate' column. Without
    the flag that column is a dash for every platform, and the only rate
    anyone sees is the working pass's, which is biased high by design."""
    assert "--audit-sample" in _joined(steps["Validate"])


def test_collection_enables_crawl_discovery(steps):
    """CI checks out fresh, so with collinfo.json down and no cache there is
    nothing to fall back on unless discovery is asked for."""
    assert "--discover-years" in _joined(steps["Collect"])


def test_prune_runs_before_collection(workflow):
    """Pruning replays current extraction rules over the seeded harvest, so it
    has to happen after seeding and before new slugs are added."""
    names = [s["name"] for s in workflow["jobs"]["harvest"]["steps"]]
    assert names.index("Fetch the previously published state") \
        < names.index("Re-apply current rules to the seeded harvest") \
        < names.index("Collect")


def test_prune_targets_the_harvest_directory(steps):
    joined = _joined(steps["Re-apply current rules to the seeded harvest"])
    assert "--prune" in joined and "data/ats_harvest" in joined


# --------------------------------------------------------------------------
# Publish and restore agree
# --------------------------------------------------------------------------

@pytest.mark.parametrize("directory,prefix", [
    ("data/ats_harvest", ""),
    ("data/dead_slugs", "dead/"),
    ("data/ats_checked", "checked/"),
])
def test_every_state_directory_survives_a_run(steps, directory, prefix):
    """Each directory the tools write must be created, restored and published.

    CI checks out fresh, so a state directory that is written but not seeded
    back starts empty every week and whatever it was for silently does not
    work. That has happened twice: the crawl cache was published and never
    restored, and ats_checked -- which stops the validator re-probing companies
    it has already confirmed, 93% of one run's budget -- was not referenced by
    the workflow at all when it was added.
    """
    seed = steps["Fetch the previously published state"]
    manifest = steps["Build the manifest"]
    publish = steps["Publish to the ats-harvest branch"]
    assert directory in seed, f"{directory} is not created or restored"
    assert directory in manifest, f"{directory} is not published"
    if prefix:
        assert f"FETCH_HEAD:{prefix}" in seed, f"{prefix} is not seeded back"
        assert prefix.rstrip("/") + "/*.json" in publish,             f"{prefix} is not committed"


def test_change_detection_sees_every_state_directory(steps):
    """A directory left out here is republished only when something else
    changed, so its updates can sit unpublished indefinitely."""
    changed = steps["Skip when nothing changed"]
    for rel in ("$p.json", "dead/$p.json", "checked/$p.json"):
        assert rel in changed, rel


def test_every_published_artefact_is_restored(steps):
    """The round trip has to close.

    _crawls.json was published every run and restored never, so the cache was
    inert in exactly the situation it exists for. A published file that nothing
    seeds back is invisible: the run still succeeds.
    """
    seed = steps["Fetch the previously published state"]
    assert "FETCH_HEAD:_crawls.json" in seed, "crawl cache is published but never restored"
    assert 'FETCH_HEAD:$p.json' in seed
    assert 'FETCH_HEAD:dead/$p.json' in seed


def test_publish_includes_the_crawl_cache(steps):
    """`./*.json` covers _crawls.json because a bash glob matches a leading
    underscore. If that ever became an explicit list, the cache would drop out
    silently."""
    publish = _joined(steps["Publish to the ats-harvest branch"])
    assert "git add ./*.json dead/*.json" in publish


# --------------------------------------------------------------------------
# Platform lists stay in step with the code
# --------------------------------------------------------------------------

PLATFORM_NAMES = [p.name for p in hc.PLATFORMS]


@pytest.mark.parametrize("platform", PLATFORM_NAMES)
def test_seed_step_covers_every_platform(steps, platform):
    """A platform missing from the shell loop is never seeded, so its harvest
    restarts empty every week and the run still reports success."""
    assert platform in steps["Fetch the previously published state"], platform


@pytest.mark.parametrize("platform", PLATFORM_NAMES)
def test_change_detection_covers_every_platform(steps, platform):
    assert platform in steps["Skip when nothing changed"], platform


@pytest.mark.parametrize("platform", PLATFORM_NAMES)
def test_summary_covers_every_platform(steps, platform):
    assert platform in steps["Summary"], platform


def test_harvester_and_validator_agree_on_platforms():
    """They are separate lists in separate files; a platform added to one and
    not the other harvests without ever being validated, or vice versa."""
    assert set(PLATFORM_NAMES) == set(v.PLATFORMS)


# --------------------------------------------------------------------------
# The verification gates cannot be bypassed by accident
# --------------------------------------------------------------------------

def test_tests_run_before_anything_touches_the_network(workflow):
    names = [s["name"] for s in workflow["jobs"]["harvest"]["steps"]]
    assert names.index("Verify harvester and validator before running them") \
        < names.index("Collect")


def test_both_test_suites_are_run(steps):
    verify = _joined(steps["Verify harvester and validator before running them"])
    assert "tests/test_harvest_ats.py" in verify
    assert "tests/test_validate_ats_slugs.py" in verify


def test_publishing_is_gated_on_something_having_changed(workflow):
    publish = next(s for s in workflow["jobs"]["harvest"]["steps"]
                   if s["name"] == "Publish to the ats-harvest branch")
    assert publish.get("if") == "steps.changed.outputs.publish == 'true'"


def test_the_token_never_reaches_a_command_line(steps):
    """A secret interpolated into `run:` lands in process arguments, where a
    shell trace or crash dump carries it."""
    publish = steps["Publish to the ats-harvest branch"]
    assert "secrets.GITHUB_TOKEN" not in publish
    assert "${GH_TOKEN}" in publish


def test_floors_exist_for_every_platform_that_can_have_one(steps):
    """Lever is the sole exception: it blocks CCBot, so zero from Common Crawl
    is the correct result rather than a regression."""
    verify = steps["Verify the collection before validating it"]
    for platform in PLATFORM_NAMES:
        if platform == "lever":
            continue
        assert f'"{platform}"' in verify, platform
