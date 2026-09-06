"""The bot's log has to be readable while the bot is still running.

app.py's stdout is a file here, not a terminal, and Python block-buffers to a
file. Without -u the log appears only in ~8KB lumps whenever the buffer happens
to fill, so a bot running for hours can leave a log that stopped at startup.

That is not hypothetical. A two-day outage of the Glassdoor and ZipRecruiter
job sources produced no visible line anywhere: every "Skipping glassdoor",
every traceback and every [ats-scrape] was sitting in a buffer nobody could
read, and the outage had to be found by querying the archive by hand. Measured
while fixing it: a child writing three lines then sleeping leaves 0 bytes in
the file without -u, and 24 bytes with it.

deploy/discordbot.service already sets PYTHONUNBUFFERED=1, so the deployment
had ruled this out and only the launcher had not -- which meant every restart
driven by .reset silently reintroduced it.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import run as run_module  # noqa: E402


class _FakeProc:
    pid = 4321

    def wait(self) -> int:
        return 0

    def send_signal(self, signum) -> None:
        pass


def _captured_argv(monkeypatch) -> list[str]:
    seen: dict[str, list[str]] = {}

    def _popen(args, *rest, **kwargs):
        seen["argv"] = list(args)
        return _FakeProc()

    monkeypatch.setattr(run_module.subprocess, "Popen", _popen)
    monkeypatch.setattr(run_module.signal, "signal", lambda sig, handler: None)
    run_module._run_bot_forwarding_signals("python")
    return seen["argv"]


def test_the_bot_is_launched_unbuffered(monkeypatch):
    """Otherwise the log is written in lumps and reads as a bot that stopped."""
    assert "-u" in _captured_argv(monkeypatch)


def test_the_flag_reaches_the_interpreter_not_the_script(monkeypatch):
    """-u is an interpreter flag. After the script path it is an argument to
    the bot instead, which would leave stdout buffered while looking correct in
    the argv."""
    argv = _captured_argv(monkeypatch)
    app = next(i for i, a in enumerate(argv) if str(a).endswith("app.py"))
    assert argv.index("-u") < app


def test_the_launcher_and_the_unit_file_agree():
    """The two ways this bot starts must not disagree about buffering.

    The unit file sets PYTHONUNBUFFERED=1 and run.py did not, so the same code
    logged one way under systemd and another way after a .reset. A future edit
    that drops either half should fail here rather than be discovered during
    the next outage.
    """
    unit = (REPO_ROOT / "deploy" / "discordbot.service").read_text(encoding="utf-8")
    assert "PYTHONUNBUFFERED=1" in unit
