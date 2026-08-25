"""Committing and pushing job archives from the bot's own working tree.

Drives real git against real repositories in tmp_path -- a bare remote and a
clone -- rather than mocking subprocess. The properties that matter here are
properties of git's behaviour (does the pathspec really keep unrelated edits out
of the commit? does a rejected push really leave the commit local?), and a mock
of subprocess.run would assert only that we build the argv we think we build.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from services.jba import merge_data


def _git(repo: Path, *argv: str) -> str:
    result = subprocess.run(
        ["git", *argv], cwd=str(repo), capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A clone with an archive layout, wired to a bare origin."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)],
                   check=True, capture_output=True)

    work = tmp_path / "work"
    subprocess.run(["git", "clone", str(origin), str(work)], check=True, capture_output=True)
    _git(work, "config", "user.name", "seed")
    _git(work, "config", "user.email", "seed@example.invalid")

    # Mirrors the real .gitignore: *.zip and *.db ignored, archive zips
    # un-ignored by negation. The *.db rule is load-bearing -- the live jobs.db
    # and archive_index.db sit inside the archive pathspec being staged.
    (work / ".gitignore").write_text(
        "*.zip\n!data/jba/jobs/*/*.zip\n*.db\n*.db-shm\n*.db-wal\n", encoding="utf-8")
    jobs = work / "data" / "jba" / "jobs"
    jobs.mkdir(parents=True)
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "seed")
    _git(work, "push", "-u", "origin", "main")

    monkeypatch.setattr(merge_data, "_JOBS_DIR", jobs)
    monkeypatch.setattr(merge_data, "_last_archive_commit_day", None, raising=False)
    monkeypatch.setenv("JBA_ARCHIVE_GIT_COMMIT", "1")
    monkeypatch.delenv("JBA_ARCHIVE_GIT_PUSH", raising=False)
    return work


def _archive(repo: Path, month: str, day: str = "01") -> Path:
    month_dir = repo / "data" / "jba" / "jobs" / month
    month_dir.mkdir(parents=True, exist_ok=True)
    path = month_dir / f"{month}-w1.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{month}-{day}.json", "[]")
    return path


def _committed_files(repo: Path) -> list[str]:
    return _git(repo, "show", "--name-only", "--pretty=format:", "HEAD").split()


# ---------------------------------------------------------------------------
# Committing
# ---------------------------------------------------------------------------

def test_weekly_archive_is_committed(repo):
    """Weekly zips were never committed before -- only the month-end
    consolidation was -- so a week of data sat untracked until then."""
    _archive(repo, "2026-08")

    assert merge_data.commit_archives() is True
    assert _committed_files(repo) == ["data/jba/jobs/2026-08/2026-08-w1.zip"]


def test_nothing_to_commit_is_not_an_error(repo):
    before = _git(repo, "rev-parse", "HEAD")

    assert merge_data.commit_archives() is False
    assert _git(repo, "rev-parse", "HEAD") == before


def test_unrelated_working_tree_edits_stay_out_of_the_commit(repo):
    """The bot's tree routinely carries the operator's own edits. An unscoped
    commit here would sweep them into an archive commit."""
    (repo / "src_change.py").write_text("# operator work in progress\n", encoding="utf-8")
    _git(repo, "add", "src_change.py")
    _archive(repo, "2026-08")

    assert merge_data.commit_archives() is True
    assert _committed_files(repo) == ["data/jba/jobs/2026-08/2026-08-w1.zip"]
    # Still staged, still uncommitted.
    assert "src_change.py" in _git(repo, "diff", "--cached", "--name-only")


def test_deleted_archives_are_staged_too(repo):
    """Monthly consolidation unlinks the weeklies it folded in. Committing the
    addition but not the deletions would leave the tree permanently dirty."""
    weekly = _archive(repo, "2026-08")
    merge_data.commit_archives()

    weekly.unlink()
    monthly = repo / "data" / "jba" / "jobs" / "2026-08" / "2026-08.zip"
    with zipfile.ZipFile(monthly, "w") as zf:
        zf.writestr("2026-08-01.json", "[]")

    assert merge_data.commit_archives() is True
    assert _git(repo, "status", "--porcelain", "data/jba") == ""


def test_commit_is_disabled_by_default(repo, monkeypatch):
    monkeypatch.delenv("JBA_ARCHIVE_GIT_COMMIT", raising=False)
    _archive(repo, "2026-08")

    assert merge_data.commit_archives() is False
    assert _git(repo, "status", "--porcelain", "data/jba") != ""


def test_commit_message_names_the_month(repo):
    _archive(repo, "2026-08")
    merge_data.commit_archives()

    assert _git(repo, "log", "-1", "--pretty=%s") == "archive: 2026-08 job data"


def test_commit_message_spans_a_range_when_several_months_land(repo):
    _archive(repo, "2026-07")
    _archive(repo, "2026-08")
    merge_data.commit_archives()

    assert _git(repo, "log", "-1", "--pretty=%s") == "archive: job data (2026-07..2026-08)"


# ---------------------------------------------------------------------------
# Pushing
# ---------------------------------------------------------------------------

def test_push_is_opt_in_separately_from_commit(repo):
    """Wanting local archive commits is not the same as asking to publish."""
    _archive(repo, "2026-08")

    assert merge_data.commit_archives() is True
    assert _git(repo, "rev-list", "--count", "origin/main..main") == "1"


def test_push_publishes_the_commit(repo, monkeypatch):
    monkeypatch.setenv("JBA_ARCHIVE_GIT_PUSH", "1")
    _archive(repo, "2026-08")

    assert merge_data.commit_archives() is True
    _git(repo, "fetch", "origin")
    assert _git(repo, "rev-list", "--count", "origin/main..main") == "0"


def test_rejected_push_leaves_the_commit_local_and_the_tree_clean(repo, monkeypatch, tmp_path):
    """A diverged remote must not lose the commit, and must not trigger an
    unattended rebase across the operator's working tree."""
    other = tmp_path / "other"
    subprocess.run(["git", "clone", str(tmp_path / "origin.git"), str(other)],
                   check=True, capture_output=True)
    _git(other, "config", "user.name", "other")
    _git(other, "config", "user.email", "other@example.invalid")
    (other / "unrelated.txt").write_text("landed first\n", encoding="utf-8")
    _git(other, "add", "-A")
    _git(other, "commit", "-m", "remote moves ahead")
    _git(other, "push", "origin", "main")

    monkeypatch.setenv("JBA_ARCHIVE_GIT_PUSH", "1")
    _archive(repo, "2026-08")

    assert merge_data.commit_archives() is True
    assert _git(repo, "log", "-1", "--pretty=%s") == "archive: 2026-08 job data"
    assert _git(repo, "status", "--porcelain", "data/jba") == ""


def test_detached_head_is_not_pushed(repo, monkeypatch):
    monkeypatch.setenv("JBA_ARCHIVE_GIT_PUSH", "1")
    _git(repo, "checkout", "--detach", "HEAD")
    _archive(repo, "2026-08")

    assert merge_data.commit_archives() is True
    assert merge_data._current_branch() is None


# ---------------------------------------------------------------------------
# The daily cadence
# ---------------------------------------------------------------------------

def test_daily_commit_runs_once_per_day(repo, monkeypatch):
    monkeypatch.setattr(merge_data, "_today_str", lambda: "2026-08-24")
    _archive(repo, "2026-08")

    assert merge_data.commit_archives_daily() is True
    _archive(repo, "2026-08", day="02")
    assert merge_data.commit_archives_daily() is False

    monkeypatch.setattr(merge_data, "_today_str", lambda: "2026-08-25")
    assert merge_data.commit_archives_daily() is True


def test_daily_commit_survives_a_git_failure(repo, monkeypatch):
    """Archive bookkeeping must never take down the ATS scrape loop."""
    def _boom(**_):
        raise RuntimeError("git exploded")

    monkeypatch.setattr(merge_data, "commit_archives", _boom)
    assert merge_data.commit_archives_daily() is False


# ---------------------------------------------------------------------------
# Remote resolution -- this repo's remote is named DiscordBot, not origin
# ---------------------------------------------------------------------------

def test_single_remote_is_used_whatever_it_is_called(repo, monkeypatch):
    monkeypatch.setenv("JBA_ARCHIVE_GIT_PUSH", "1")
    _git(repo, "remote", "rename", "origin", "DiscordBot")
    _archive(repo, "2026-08")

    assert merge_data._resolve_remote() == "DiscordBot"
    assert merge_data.commit_archives() is True
    _git(repo, "fetch", "DiscordBot")
    assert _git(repo, "rev-list", "--count", "DiscordBot/main..main") == "0"


def test_explicit_remote_setting_wins(repo, monkeypatch):
    monkeypatch.setenv("JBA_GIT_REMOTE", "somewhere-else")
    assert merge_data._resolve_remote() == "somewhere-else"


def test_origin_preferred_when_several_remotes_exist(repo):
    _git(repo, "remote", "add", "backup", "https://example.invalid/x.git")
    assert merge_data._resolve_remote() == "origin"


def test_no_remote_means_the_commit_simply_stays_local(repo, monkeypatch):
    monkeypatch.setenv("JBA_ARCHIVE_GIT_PUSH", "1")
    _git(repo, "remote", "remove", "origin")
    _archive(repo, "2026-08")

    assert merge_data.commit_archives() is True
    assert merge_data._resolve_remote() is None


# ---------------------------------------------------------------------------
# Ignored working files must stay out
# ---------------------------------------------------------------------------

def test_gitignored_databases_are_never_committed(repo):
    """jobs.db and archive_index.db live inside the archive pathspec.

    Regression: staging with `git add -A -f` overrode .gitignore for the whole
    directory and force-added them -- 137MB and 294MB respectively, plus WAL
    sidecars. GitHub rejected the push over its 100MB limit, which is the only
    reason it was caught; a self-hosted remote would have taken it.
    """
    jobs = repo / "data" / "jba" / "jobs"
    (jobs / "jobs.db").write_bytes(b"sqlite-ish payload")
    (jobs / "jobs.db-wal").write_bytes(b"wal")
    (jobs / "archive_index.db").write_bytes(b"derived index")
    _archive(repo, "2026-08")

    assert merge_data.commit_archives() is True
    assert _committed_files(repo) == ["data/jba/jobs/2026-08/2026-08-w1.zip"]

    tracked = _git(repo, "ls-files", "--", "data/jba/jobs").splitlines()
    assert not [name for name in tracked if ".db" in name]


def test_ignored_databases_do_not_by_themselves_trigger_a_commit(repo):
    """A day where only the live DBs changed must produce no commit at all."""
    jobs = repo / "data" / "jba" / "jobs"
    (jobs / "jobs.db").write_bytes(b"changing constantly")

    assert merge_data.commit_archives() is False
