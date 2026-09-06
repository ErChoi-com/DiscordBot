"""Pulling the harvest the ats-harvest workflow publishes.

Two defects motivated this, and both failed closed and silently, which is the
worst combination the ATS pipeline can produce -- an empty company list and a
successful-looking scrape are indistinguishable downstream.

  1. The fetch named the remote `origin`. This checkout calls its remote
     `DiscordBot`, so every fetch exited 128 and the handler reported "No
     ats-harvest branch yet". The branch had existed for days.

  2. The platform list was six hardcoded names against the sixteen the
     workflow publishes, so ten platforms would never have been synced even
     once the fetch worked.

These tests run against a real local git repository rather than a mocked
subprocess: the bug was in what git actually does with a remote name, and a
mock of `subprocess.run` would have happily passed with the bug in place.
Nothing here touches the network.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sync_ats_companies as S  # noqa: E402


# ── a real published branch, built locally ───────────────────────────────────

def _git(*args, cwd):
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert out.returncode == 0, f"git {' '.join(args)}: {out.stderr}"
    return out


def _publish(upstream: Path, files: dict[str, object], manifest: object | None = "auto"):
    """Create `upstream` with an ats-harvest branch carrying `files`.

    `manifest="auto"` builds a correct manifest; pass an explicit object to
    publish a wrong one, or None to publish no manifest at all.
    """
    upstream.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main", cwd=upstream)
    _git("config", "user.email", "t@example.com", cwd=upstream)
    _git("config", "user.name", "t", cwd=upstream)
    (upstream / "README").write_text("seed", encoding="utf-8")
    _git("add", "-A", cwd=upstream)
    _git("commit", "-qm", "seed", cwd=upstream)

    _git("checkout", "-q", "--orphan", S.HARVEST_BRANCH, cwd=upstream)
    _git("rm", "-rqf", ".", cwd=upstream)

    entries: dict[str, dict] = {}
    for name, payload in files.items():
        blob = (payload if isinstance(payload, bytes)
                else json.dumps(payload).encode("utf-8"))
        (upstream / name).write_bytes(blob)
        if isinstance(payload, list):
            entries[name] = {"sha256": hashlib.sha256(blob).hexdigest(),
                             "count": len(payload)}

    if manifest == "auto":
        manifest = {"updated": "2026-09-05T00:00:00Z", "files": entries}
    if manifest is not None:
        (upstream / S.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    _git("add", "-A", cwd=upstream)
    _git("commit", "-qm", "harvest", cwd=upstream)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A working checkout whose remote is deliberately NOT called `origin`."""
    work = tmp_path / "work"
    work.mkdir()
    _git("init", "-q", "-b", "main", cwd=work)
    _git("config", "user.email", "t@example.com", cwd=work)
    _git("config", "user.name", "t", cwd=work)

    monkeypatch.chdir(work)
    monkeypatch.setattr(S, "HARVEST_DIR", tmp_path / "harvest")
    monkeypatch.setattr(S, "DEAD_SLUG_DIR", tmp_path / "dead")

    class Env:
        root = tmp_path
        checkout = work
        harvest = tmp_path / "harvest"
        dead = tmp_path / "dead"

        def add_remote(self, name, path):
            _git("remote", "add", name, str(path), cwd=work)

    return Env()


_SIXTEEN = [
    "applicantpro", "ashby", "bamboohr", "breezy", "greenhouse", "icims",
    "jazzhr", "jobvite", "lever", "paylocity", "recruitee", "rippling",
    "smartrecruiters", "teamtailor", "workable", "workday",
]


def _fleet(names=_SIXTEEN, size=3):
    files = {f"{n}.json": [f"{n}-co{i}" for i in range(size)] for n in names}
    files["_crawls.json"] = ["CC-MAIN-2026-30"]
    return files


# ── the remote-name bug ──────────────────────────────────────────────────────

def test_the_harvest_syncs_from_a_remote_not_called_origin(env):
    """The actual defect. With `origin` hardcoded this returned (0, 0) and
    printed "No ats-harvest branch yet" while the branch sat there.
    """
    _publish(env.root / "upstream", _fleet())
    env.add_remote("DiscordBot", env.root / "upstream")

    written, available = S._sync_harvest()
    assert (written, available) == (16, 16)


def test_origin_is_tried_first_when_it_exists(env):
    _publish(env.root / "upstream", _fleet())
    env.add_remote("origin", env.root / "upstream")
    assert S._remote_names()[0] == "origin"
    assert S._fetch_harvest() == "origin"


def test_a_remote_without_the_branch_does_not_stop_one_that_has_it(env):
    """Several remotes is normal. Giving up on the first refusal would make
    this depend on the alphabetical accident of which remote sorts first.
    """
    _publish(env.root / "upstream", _fleet())
    empty = env.root / "empty"
    empty.mkdir()
    _git("init", "-q", "-b", "main", cwd=empty)
    env.add_remote("aaa-no-branch", empty)
    env.add_remote("zzz-has-branch", env.root / "upstream")

    assert S._fetch_harvest() == "zzz-has-branch"


def test_no_remotes_at_all_is_reported_not_crashed(env, capsys):
    assert S._sync_harvest() == (0, 0)
    assert "No git remotes configured" in capsys.readouterr().out


def test_a_missing_branch_names_the_remotes_it_tried(env, capsys):
    """The old message claimed there was no branch. When the real cause is a
    bad remote or no network, saying so is the difference between a five-minute
    fix and weeks of a silently empty harvest.
    """
    empty = env.root / "empty"
    empty.mkdir()
    _git("init", "-q", "-b", "main", cwd=empty)
    env.add_remote("DiscordBot", empty)

    assert S._fetch_harvest() is None
    assert "DiscordBot" in capsys.readouterr().err


def test_outside_a_git_checkout_it_skips_rather_than_raising(env, monkeypatch, tmp_path):
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    monkeypatch.chdir(plain)
    assert S._sync_harvest() == (0, 0)


# ── the hardcoded platform list ──────────────────────────────────────────────

def test_every_published_platform_is_pulled_not_a_hardcoded_six(env):
    _publish(env.root / "upstream", _fleet())
    env.add_remote("DiscordBot", env.root / "upstream")
    S._sync_harvest()

    got = {p.stem for p in env.harvest.glob("*.json")}
    assert got == set(_SIXTEEN)


def test_a_platform_added_to_the_workflow_needs_no_change_here(env):
    """The regression guard. Ten platforms went unsynced because this list was
    written once and never revisited; the fix is that there is no list.
    """
    _publish(env.root / "upstream", _fleet(_SIXTEEN + ["brandnewats"]))
    env.add_remote("DiscordBot", env.root / "upstream")

    written, available = S._sync_harvest()
    assert (written, available) == (17, 17)
    assert (env.harvest / "brandnewats.json").exists()


def test_bookkeeping_files_are_not_mistaken_for_platforms(env):
    _publish(env.root / "upstream", _fleet(["lever"]))
    env.add_remote("DiscordBot", env.root / "upstream")
    S._sync_harvest()

    assert not (env.harvest / "manifest.json").exists()
    assert not (env.harvest / "_crawls.json").exists()
    assert (env.harvest / "lever.json").exists()


def test_the_branch_listing_carries_it_when_the_manifest_is_missing(env):
    """A missing manifest must not take the fleet down with it -- that would
    make a bookkeeping file the single point of failure for every platform.
    """
    _publish(env.root / "upstream", _fleet(["lever", "ashby"]), manifest=None)
    env.add_remote("DiscordBot", env.root / "upstream")

    written, available = S._sync_harvest()
    assert (written, available) == (2, 2)


def test_a_corrupt_manifest_falls_back_rather_than_refusing(env):
    _publish(env.root / "upstream", _fleet(["lever"]), manifest="{ truncated")
    env.add_remote("DiscordBot", env.root / "upstream")
    assert S._sync_harvest() == (1, 1)


# ── integrity: never replace a good list with a bad one ──────────────────────

def test_a_file_that_fails_its_checksum_keeps_the_local_copy(env, capsys):
    """`ats_service` cannot tell a truncated company list from a complete one;
    it reads both as "these are the companies". So the check has to happen
    before the write, not after.
    """
    files = _fleet(["lever"])
    _publish(env.root / "upstream", files,
             manifest={"files": {"lever.json": {"sha256": "0" * 64, "count": 3}}})
    env.add_remote("DiscordBot", env.root / "upstream")

    env.harvest.mkdir(parents=True)
    (env.harvest / "lever.json").write_text('["keep-me"]', encoding="utf-8")

    written, available = S._sync_harvest()
    assert (written, available) == (0, 1)
    assert json.loads((env.harvest / "lever.json").read_text()) == ["keep-me"]
    assert "checksum" in capsys.readouterr().err


def test_a_file_whose_count_disagrees_keeps_the_local_copy(env, capsys):
    """A file can lose entries and still be valid JSON. The checksum catches
    corruption in transit; the count catches a short but well-formed list.
    """
    blob = json.dumps(["a", "b"]).encode("utf-8")
    _publish(env.root / "upstream", {"lever.json": blob},
             manifest={"files": {"lever.json": {
                 "sha256": hashlib.sha256(blob).hexdigest(), "count": 900}}})
    env.add_remote("DiscordBot", env.root / "upstream")

    assert S._sync_harvest() == (0, 1)
    assert not (env.harvest / "lever.json").exists()
    assert "manifest says 900" in capsys.readouterr().err


def test_an_unparseable_file_keeps_the_local_copy(env):
    _publish(env.root / "upstream", {"lever.json": b"{ not json"}, manifest=None)
    env.add_remote("DiscordBot", env.root / "upstream")

    env.harvest.mkdir(parents=True)
    (env.harvest / "lever.json").write_text('["keep-me"]', encoding="utf-8")

    assert S._sync_harvest() == (0, 1)
    assert json.loads((env.harvest / "lever.json").read_text()) == ["keep-me"]


def test_a_json_object_is_not_accepted_as_a_company_list(env):
    _publish(env.root / "upstream", {"lever.json": b'{"a": 1}'}, manifest=None)
    env.add_remote("DiscordBot", env.root / "upstream")
    assert S._sync_harvest() == (0, 1)


def test_one_bad_file_does_not_stop_the_other_fifteen(env):
    files = _fleet()
    files["icims.json"] = b"{ truncated"
    _publish(env.root / "upstream", files, manifest=None)
    env.add_remote("DiscordBot", env.root / "upstream")

    written, available = S._sync_harvest()
    assert (written, available) == (15, 16)   # icims replaced, not added
    assert not (env.harvest / "icims.json").exists()


def test_a_correct_file_verifies_and_is_written(env):
    _publish(env.root / "upstream", _fleet(["lever"], size=5))
    env.add_remote("DiscordBot", env.root / "upstream")
    S._sync_harvest()
    assert len(json.loads((env.harvest / "lever.json").read_text())) == 5


def test_no_temp_files_are_left_behind(env):
    """A sync killed mid-write must not leave a truncated list the next boot
    reads as a smaller fleet, so writes go through tmp+replace.
    """
    _publish(env.root / "upstream", _fleet())
    env.add_remote("DiscordBot", env.root / "upstream")
    S._sync_harvest()
    assert list(env.harvest.glob("*.tmp")) == []


# ── what the caller is told ──────────────────────────────────────────────────

def test_written_and_available_are_both_reported(env):
    """"6 written" is healthy against 6 available and a silent regression
    against 16. One number cannot be read on its own.
    """
    files = _fleet()
    files["breezy.json"] = b"{ bad"
    files["jazzhr.json"] = b"{ bad"
    _publish(env.root / "upstream", files, manifest=None)
    env.add_remote("DiscordBot", env.root / "upstream")

    assert S._sync_harvest() == (14, 16)


def test_a_shortfall_is_called_out_on_stderr(env, capsys):
    files = _fleet(["lever", "ashby"])
    files["ashby.json"] = b"{ bad"
    _publish(env.root / "upstream", files, manifest=None)
    env.add_remote("DiscordBot", env.root / "upstream")
    S._sync_harvest()
    assert "1 of 2 harvest file(s) were not updated" in capsys.readouterr().err


# ── dead marks ───────────────────────────────────────────────────────────────

def test_no_dead_marks_published_is_a_clean_no_op(env):
    """ATS probing is deliberately off the runners, so the branch carries no
    dead marks today. That must be a quiet zero, not an error.
    """
    _publish(env.root / "upstream", _fleet(["lever"]))
    env.add_remote("DiscordBot", env.root / "upstream")
    S._sync_harvest()
    assert S._merge_dead_marks() == 0


def test_published_dead_marks_merge_into_the_local_ones(env):
    files = _fleet(["lever"])
    files["dead/lever.json"] = {"gone-co": "2026-09-01"}
    upstream = env.root / "upstream"
    upstream.mkdir(parents=True)
    (upstream / "dead").mkdir(parents=True, exist_ok=True)
    _publish(upstream, files)
    env.add_remote("DiscordBot", upstream)

    env.dead.mkdir(parents=True)
    (env.dead / "lever.json").write_text(
        json.dumps({"local-co": "2026-08-01"}), encoding="utf-8")

    S._sync_harvest()
    merged = json.loads((env.dead / "lever.json").read_text())
    assert merged == {"local-co": "2026-08-01", "gone-co": "2026-09-01"}


def test_a_newer_local_mark_is_not_aged_backwards(env):
    """The local file is live state the running bot mutates. An older
    published mark must not overwrite a fresher local one.
    """
    files = _fleet(["lever"])
    files["dead/lever.json"] = {"co": "2026-08-01"}
    upstream = env.root / "upstream"
    upstream.mkdir(parents=True)
    (upstream / "dead").mkdir(parents=True, exist_ok=True)
    _publish(upstream, files)
    env.add_remote("DiscordBot", upstream)

    env.dead.mkdir(parents=True)
    (env.dead / "lever.json").write_text(
        json.dumps({"co": "2026-09-01"}), encoding="utf-8")

    S._sync_harvest()
    assert json.loads((env.dead / "lever.json").read_text()) == {"co": "2026-09-01"}
