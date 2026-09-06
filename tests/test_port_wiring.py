"""Integration-level tests that the Windows->Linux port is actually WIRED UP.

The unit tests in test_platform_support.py prove the shim computes the right
answers. These prove the rest of the app actually asks it -- a distinction that
matters, because every one of these would still pass the shim's own unit tests
while the consumer quietly used a hardcoded Windows value.

Every test here drives an explicit `system=` or patches the shim and asserts the
consumer's observable behaviour CHANGES. A test that passes whether or not the
wiring exists is worthless, so none of these assert mere truthiness.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from services import browser_service, capacity, health, job_service, platform_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402


# ---------------------------------------------------------------------------
# browser_service actually uses the shim
# ---------------------------------------------------------------------------

def test_browser_user_agent_is_not_hardcoded_windows():
    """browser_service._UA must come from the shim, so a Linux host advertises
    Linux -- Chrome sends Sec-CH-UA-Platform from the real OS and a mismatch is
    detectable."""
    assert browser_service._UA == platform_support.chrome_user_agent()
    # And the shim genuinely differentiates, otherwise the above is vacuous.
    assert platform_support.chrome_user_agent(platform_support.LINUX) != \
        platform_support.chrome_user_agent(platform_support.WINDOWS)


def test_browser_find_chrome_delegates_to_the_shim(monkeypatch):
    monkeypatch.setattr(platform_support, "find_chrome", lambda: "/sentinel/chrome")
    assert browser_service._find_chrome() == "/sentinel/chrome"


class _DanglingSymlink:
    """Stands in for a POSIX dangling symlink: exists() is False but
    is_symlink() is True.

    Creating a real one needs privilege on Windows, so a real-symlink test
    SKIPS there -- and a skipped test protects nothing on the machine this
    suite actually runs on. This stub exercises the same branch everywhere.
    """

    def __init__(self) -> None:
        self.unlinked = False

    def exists(self) -> bool:
        return False

    def is_symlink(self) -> bool:
        return True

    def unlink(self) -> None:
        self.unlinked = True


def test_clear_profile_locks_removes_a_dangling_symlink_stub(monkeypatch):
    """Runs on every OS. On POSIX the Singleton* entries are symlinks to a
    host-pid token that stops resolving once Chrome dies, so exists() is False
    for exactly the stale locks we must remove -- only is_symlink() catches
    them."""
    stub = _DanglingSymlink()
    monkeypatch.setattr(platform_support, "profile_lock_paths", lambda _p: [stub])

    browser_service._clear_profile_locks("/irrelevant")

    assert stub.unlinked is True, "dangling symlink lock was not removed"


@pytest.mark.skipif(
    not hasattr(os, "symlink"), reason="platform has no symlink support"
)
def test_clear_profile_locks_removes_a_real_dangling_symlink(tmp_path):
    """The same branch against a genuine dangling symlink. Skips where symlink
    creation is not permitted; the stub test above is the portable guarantee."""
    profile = tmp_path / "profile"
    profile.mkdir()
    link = profile / "SingletonLock"
    try:
        os.symlink(tmp_path / "does-not-exist", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this host")

    assert link.exists() is False and link.is_symlink() is True

    browser_service._clear_profile_locks(str(profile))

    assert link.is_symlink() is False, "dangling symlink lock was not removed"


def test_clear_profile_locks_removes_every_artifact_the_shim_lists(tmp_path):
    """Guards against the lock list and the removal loop drifting apart."""
    profile = tmp_path / "profile"
    (profile / "Default").mkdir(parents=True)
    created = []
    for path in platform_support.profile_lock_paths(profile):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("lock", encoding="utf-8")
        created.append(path)
    assert len(created) >= 5

    browser_service._clear_profile_locks(str(profile))

    still_there = [p for p in created if p.exists()]
    assert still_there == [], f"not removed: {still_there}"


def test_clear_profile_locks_leaves_unrelated_files_alone(tmp_path):
    """Over-deletion would destroy the harvested Reddit session."""
    profile = tmp_path / "profile"
    default = profile / "Default"
    default.mkdir(parents=True)
    cookies = default / "Cookies"
    cookies.write_bytes(b"precious-session")
    (profile / "SingletonLock").write_text("lock", encoding="utf-8")

    browser_service._clear_profile_locks(str(profile))

    assert cookies.read_bytes() == b"precious-session"


def test_sandbox_args_reach_the_launch_arguments(monkeypatch):
    """--no-sandbox must actually make it into the Chrome argv, not just exist
    as a helper. Captures the real launch call."""
    captured = {}

    class _FakeChromium:
        def launch_persistent_context(self, **kwargs):
            captured.update(kwargs)
            return object()

    class _FakePW:
        chromium = _FakeChromium()

    monkeypatch.setattr(browser_service, "_pw", _FakePW())
    monkeypatch.setattr(browser_service, "_find_chrome", lambda: "/usr/bin/chromium")
    monkeypatch.setattr(platform_support, "chrome_sandbox_args", lambda: ["--no-sandbox"])

    browser_service._launch_context("/tmp/profile")

    assert "--no-sandbox" in captured["args"]
    # executable_path, not channel="chrome": the channel lookup only knows
    # branded-Chrome locations and fails where only Chromium exists.
    assert captured["executable_path"] == "/usr/bin/chromium"
    assert "channel" not in captured


def test_launch_context_passes_no_sandbox_flag_when_not_requested(monkeypatch):
    """The negative case -- otherwise the test above would pass a hardcoded
    always-on --no-sandbox, which would be a security regression."""
    captured = {}

    class _FakeChromium:
        def launch_persistent_context(self, **kwargs):
            captured.update(kwargs)
            return object()

    class _FakePW:
        chromium = _FakeChromium()

    monkeypatch.setattr(browser_service, "_pw", _FakePW())
    monkeypatch.setattr(browser_service, "_find_chrome", lambda: "/usr/bin/chromium")
    monkeypatch.setattr(platform_support, "chrome_sandbox_args", lambda: [])

    browser_service._launch_context("/tmp/profile")

    assert "--no-sandbox" not in captured["args"]


# ---------------------------------------------------------------------------
# health dashboard: profile name rendering
# ---------------------------------------------------------------------------

def _render_all_health(tracker) -> str:
    """Render the real /health embed and flatten it to text for assertions."""
    embed = health.build_all_health_embed(
        tracker, channel_job_tasks={}, channel_names={},
        active_job_watchers=0, active_reddit_watchers=0,
    )
    parts = [embed.title or "", embed.description or ""]
    for field in embed.fields:
        parts.append(f"{field.name}\n{field.value}")
    return "\n".join(parts)


def test_health_renders_profile_leaf_name_for_a_posix_path():
    """The old backslash rsplit had no separator to split on for a POSIX path,
    so it rendered the entire absolute path instead of the leaf directory."""
    tracker = health.WatcherHealthTracker()
    tracker.record_browser_event("active_profile", "/opt/discordbot/chrome_profile_runtime")

    rendered = _render_all_health(tracker)

    assert "`chrome_profile_runtime`" in rendered
    assert "/opt/discordbot" not in rendered, "rendered the full path, not the leaf name"


def test_health_renders_profile_leaf_name_for_a_windows_path():
    """The Windows behaviour must be preserved by the Path() change."""
    tracker = health.WatcherHealthTracker()
    tracker.record_browser_event("active_profile", "C:\\Users\\ernes\\bot\\chrome_profile_runtime")

    rendered = _render_all_health(tracker)

    assert "`chrome_profile_runtime`" in rendered


def test_health_queue_field_reports_detected_hardware(monkeypatch):
    monkeypatch.setattr(capacity, "describe", lambda: "cpu=4 mem=8.0GB scale=0.33")
    field = health._format_queue_field({"queued": 1, "active": 1, "workers": 4})

    assert "cpu=4 mem=8.0GB scale=0.33" in field


# ---------------------------------------------------------------------------
# jobspy interpreter discovery is platform-correct
# ---------------------------------------------------------------------------

def test_jobspy_discovery_does_not_spawn_the_windows_py_launcher_on_linux(monkeypatch):
    """`py` is the Windows launcher. On Linux it spawned four doomed processes
    per call, and an unrelated `py` binary could return a wrong interpreter."""
    monkeypatch.setattr(platform_support, "is_windows", lambda *a, **k: False)

    spawned: list[list[str]] = []

    def _record(cmd, *a, **k):
        spawned.append(cmd)
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(job_service.subprocess, "run", _record)
    monkeypatch.setattr(job_service.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name == "python3" else None)

    job_service.jobspy_python_executable()

    assert not any(cmd and cmd[0] == "py" for cmd in spawned), \
        f"spawned the Windows py launcher on Linux: {spawned}"


def test_jobspy_discovery_still_uses_py_launcher_on_windows(monkeypatch):
    """The Windows path must not have been broken by the Linux branch."""
    monkeypatch.setattr(platform_support, "is_windows", lambda *a, **k: True)

    spawned: list[list[str]] = []

    def _record(cmd, *a, **k):
        spawned.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "")

    monkeypatch.setattr(job_service.subprocess, "run", _record)

    job_service.jobspy_python_executable()

    assert any(cmd and cmd[0] == "py" for cmd in spawned), \
        "Windows py-launcher probe was lost"


# ---------------------------------------------------------------------------
# LaTeX discovery
# ---------------------------------------------------------------------------

def test_latex_search_dirs_include_texlive_on_linux(tmp_path):
    """systemd gives a service a minimal PATH, so a TeX Live install must be
    found by directory probing rather than shutil.which alone."""
    texlive = tmp_path / "usr" / "local" / "texlive"
    binary_dir = texlive / "2024" / "bin" / "x86_64-linux"
    binary_dir.mkdir(parents=True)

    dirs = platform_support.latex_search_dirs(
        system=platform_support.LINUX, env={"LATEX_BIN_DIR": str(binary_dir)},
    )
    assert binary_dir in dirs
    assert dirs[0] == binary_dir, "explicit override must be probed first"


def test_find_latex_executable_prefers_path_over_directory_probing():
    resolved = platform_support.find_latex_executable(
        "pdflatex", system=platform_support.LINUX,
        env={}, which=lambda name: "/usr/bin/pdflatex",
    )
    assert resolved == "/usr/bin/pdflatex"


def test_find_latex_executable_falls_back_to_off_path_install(tmp_path):
    binary_dir = tmp_path / "texbin"
    binary_dir.mkdir()
    (binary_dir / "pdflatex").write_text("#!/bin/sh\n", encoding="utf-8")

    resolved = platform_support.find_latex_executable(
        "pdflatex", system=platform_support.LINUX,
        env={"LATEX_BIN_DIR": str(binary_dir)}, which=lambda name: None,
    )
    assert resolved == str(binary_dir / "pdflatex")


def test_find_latex_executable_adds_exe_suffix_only_on_windows(tmp_path):
    binary_dir = tmp_path / "miktex"
    binary_dir.mkdir()
    (binary_dir / "pdflatex.exe").write_text("MZ", encoding="utf-8")

    on_windows = platform_support.find_latex_executable(
        "pdflatex", system=platform_support.WINDOWS,
        env={"LATEX_BIN_DIR": str(binary_dir)}, which=lambda name: None,
    )
    assert on_windows == str(binary_dir / "pdflatex.exe")

    # The same directory must NOT resolve on Linux, where the name has no suffix.
    on_linux = platform_support.find_latex_executable(
        "pdflatex", system=platform_support.LINUX,
        env={"LATEX_BIN_DIR": str(binary_dir)}, which=lambda name: None,
    )
    assert on_linux is None


def test_find_latex_executable_returns_none_when_absent(tmp_path):
    assert platform_support.find_latex_executable(
        "pdflatex", system=platform_support.LINUX,
        env={"LATEX_BIN_DIR": str(tmp_path)}, which=lambda name: None,
    ) is None


def test_resume_module_delegates_latex_discovery_to_the_shim(monkeypatch):
    from services.resumes import resume

    monkeypatch.setattr(
        platform_support, "find_latex_executable",
        lambda name, **kw: f"/sentinel/{name}",
    )
    assert resume._find_latex_executable("xelatex") == "/sentinel/xelatex"


# ---------------------------------------------------------------------------
# process_cmdline / pid corroboration
# ---------------------------------------------------------------------------

def test_process_cmdline_reads_proc_for_the_current_process():
    """Exercised against a REAL process (this one) rather than a fake, so it
    would catch a wrong /proc field or a decoding bug."""
    if platform_support.is_windows():
        pytest.skip("POSIX /proc layout")
    cmdline = platform_support.process_cmdline(os.getpid())
    assert cmdline is not None
    assert "python" in cmdline.lower() or "pytest" in cmdline.lower()


def test_process_cmdline_returns_none_for_an_impossible_pid():
    assert platform_support.process_cmdline(2 ** 31 - 1) is None


def test_forcerun_refuses_to_kill_a_process_that_is_not_the_bot(monkeypatch, tmp_path):
    """The dangerous case: a recycled pid in a stale lock must never be killed."""
    lock = tmp_path / ".bot.lock"
    lock.write_text('{"pid": 4242}', encoding="utf-8")
    monkeypatch.setattr(run, "OWNERSHIP_FILES", (lock,))
    monkeypatch.setattr(run.platform_support, "is_process_alive", lambda pid, system=None: True)
    monkeypatch.setattr(run.platform_support, "process_cmdline",
                        lambda pid, system=None: "/usr/sbin/sshd -D")

    killed: list[int] = []
    monkeypatch.setattr(run.platform_support, "terminate_process",
                        lambda pid, **kw: killed.append(pid) or (True, ""))

    assert run._forcerun() is True
    assert killed == [], "forcerun killed a process that was not the bot"


def test_forcerun_kills_the_real_bot_and_clears_ownership_files(monkeypatch, tmp_path):
    """The positive case -- otherwise the test above would pass on a _forcerun
    that never kills anything at all."""
    lock = tmp_path / ".bot.lock"
    lock.write_text('{"pid": 4242}', encoding="utf-8")
    monkeypatch.setattr(run, "OWNERSHIP_FILES", (lock,))
    monkeypatch.setattr(run.platform_support, "process_cmdline",
                        lambda pid, system=None: "/usr/bin/python3 /opt/bot/src/app.py")

    alive = {"value": True}
    monkeypatch.setattr(run.platform_support, "is_process_alive",
                        lambda pid, system=None: alive["value"])

    killed: list[int] = []

    def _terminate(pid, **kw):
        killed.append(pid)
        alive["value"] = False
        return True, ""

    monkeypatch.setattr(run.platform_support, "terminate_process", _terminate)

    assert run._forcerun() is True
    assert killed == [4242]
    assert not lock.exists(), "stale ownership file was not cleared"


def test_forcerun_terminates_against_an_unreapable_zombie(monkeypatch, tmp_path):
    """Regression: _forcerun used an unbounded `while pid is not None` loop.

    A POSIX zombie stays in the process table until its parent reaps it, so
    it keeps reporting alive while SIGTERM/SIGKILL are no-ops against it --
    terminate_process reports success, the pid is still there, and the loop
    spins forever. This models exactly that (is_process_alive pinned to
    always-True, the abstraction both platforms' _running_pid go through) and
    must return rather than hang.
    """
    lock = tmp_path / ".bot.lock"
    lock.write_text('{"pid": 4242}', encoding="utf-8")
    monkeypatch.setattr(run, "OWNERSHIP_FILES", (lock,))
    monkeypatch.setattr(run.platform_support, "is_process_alive", lambda pid, system=None: True)
    monkeypatch.setattr(run.platform_support, "process_cmdline",
                        lambda pid, system=None: "/usr/bin/python3 src/app.py")

    attempts: list[int] = []
    # Hard stop well above the legitimate bound, so a regression FAILS FAST here
    # instead of hanging the suite -- an unbounded loop would otherwise stall CI
    # rather than report anything.
    runaway_limit = run.MAX_STOP_ATTEMPTS * 4

    def _terminate(pid, **kw):
        attempts.append(pid)          # reports success, but the pid never goes away
        if len(attempts) > runaway_limit:
            raise AssertionError(
                f"_forcerun looped {len(attempts)} times without giving up -- "
                "the stop loop is unbounded"
            )
        return True, ""

    monkeypatch.setattr(run.platform_support, "terminate_process", _terminate)

    assert run._forcerun() is False, "must give up rather than loop forever"
    assert len(attempts) <= run.MAX_STOP_ATTEMPTS
    assert lock.exists(), "must not clear ownership files it could not free"


def test_forcerun_reports_failure_when_the_process_survives(monkeypatch, tmp_path):
    lock = tmp_path / ".bot.lock"
    lock.write_text('{"pid": 4242}', encoding="utf-8")
    monkeypatch.setattr(run, "OWNERSHIP_FILES", (lock,))
    monkeypatch.setattr(run.platform_support, "is_process_alive", lambda pid, system=None: True)
    monkeypatch.setattr(run.platform_support, "process_cmdline",
                        lambda pid, system=None: "/usr/bin/python3 src/app.py")
    monkeypatch.setattr(run.platform_support, "terminate_process",
                        lambda pid, **kw: (False, "permission denied"))

    assert run._forcerun() is False
    assert lock.exists(), "ownership file must survive a failed kill"


# ---------------------------------------------------------------------------
# Physical memory detection
# ---------------------------------------------------------------------------

def test_total_physical_memory_is_none_on_non_windows():
    """The Linux path reads /proc/meminfo in capacity; this helper is the
    Windows branch and must not claim to answer for POSIX."""
    assert platform_support.total_physical_memory(platform_support.LINUX) is None


def test_total_physical_memory_reports_a_plausible_value_on_windows():
    if not platform_support.is_windows():
        pytest.skip("Windows-only API")
    total = platform_support.total_physical_memory()
    assert total is not None
    assert 512 * 1024 ** 2 < total < 4096 * 1024 ** 3


def test_capacity_uses_the_shim_for_physical_memory(monkeypatch):
    """Wiring check: capacity must not grow its own ctypes call again."""
    monkeypatch.setattr(capacity, "_meminfo_bytes", lambda *a, **k: None)
    monkeypatch.setattr(platform_support, "total_physical_memory",
                        lambda *a, **k: 7 * 1024 ** 3)
    monkeypatch.setattr(capacity, "_CGROUP_V2_MEM", Path("/nonexistent-v2"))
    monkeypatch.setattr(capacity, "_CGROUP_V1_MEM", Path("/nonexistent-v1"))

    assert capacity.memory_limit_bytes() == 7 * 1024 ** 3


# ---------------------------------------------------------------------------
# Config: profile key case normalization
# ---------------------------------------------------------------------------

def test_main_user_profile_is_lowercased(monkeypatch, tmp_path):
    """Profile dirs on disk are all lowercase and discord_profile_key()
    lowercases everything it generates. A mixed-case .env value resolved on
    Windows only because the filesystem is case-insensitive."""
    import config as config_module

    monkeypatch.setenv("MAIN_USER_PROFILE", "Ephlex")
    monkeypatch.setenv("discordtoken", "x")
    cfg = config_module.load_config(base_dir=tmp_path)

    assert cfg.main_user_profile_key == "ephlex"


def test_main_user_profile_key_matches_generated_key_casing(monkeypatch, tmp_path):
    """The two must agree, or a .env-configured profile points at a directory
    the bot would never itself create."""
    import config as config_module
    from services.resumes.resume import discord_profile_key

    monkeypatch.setenv("MAIN_USER_PROFILE", "MixedCaseName")
    monkeypatch.setenv("discordtoken", "x")
    cfg = config_module.load_config(base_dir=tmp_path)

    assert cfg.main_user_profile_key == discord_profile_key(1, "MixedCaseName")
