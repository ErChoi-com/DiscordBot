"""Strict harvest/sync validation (item 3): per-artifact logging, hard failure
when the critical cookie DB is missing after copy, tolerance for non-critical
artifact failures."""
from __future__ import annotations

from pathlib import Path

from services import browser_service


def _make_source_profile(root: Path, *, cookies: bool = True, network: bool = True,
                         preferences: bool = True) -> Path:
    src = root / "src_profile"
    default = src / "Default"
    default.mkdir(parents=True)
    if cookies:
        (default / "Cookies").write_bytes(b"sqlite-cookie-data")
    if preferences:
        (default / "Preferences").write_text("{}")
    (default / "Login Data").write_bytes(b"login")
    if network:
        (default / "Network").mkdir()
        (default / "Network" / "Cookies").write_bytes(b"network-cookie-db")
    (default / "Local Storage").mkdir()
    (default / "Local Storage" / "leveldb").write_bytes(b"ls")
    (src / "Local State").write_text("{}")
    return src


def test_sync_succeeds_with_all_artifacts(tmp_path, capsys):
    src = _make_source_profile(tmp_path)
    runtime = tmp_path / "runtime_profile"

    assert browser_service._sync_profile_snapshot(str(src), str(runtime)) is True

    out = capsys.readouterr().out
    assert "sync: Cookies -> OK" in out
    assert "sync: Network/ -> OK" in out
    assert (runtime / "Default" / "Network" / "Cookies").exists()


def test_sync_fails_when_critical_cookie_db_missing(tmp_path, capsys):
    src = _make_source_profile(tmp_path, cookies=False, network=False)
    runtime = tmp_path / "runtime_profile"

    assert browser_service._sync_profile_snapshot(str(src), str(runtime)) is False
    assert "sync FAILED: no cookie DB" in capsys.readouterr().out


def test_sync_tolerates_missing_noncritical_artifacts(tmp_path, capsys):
    src = _make_source_profile(tmp_path, preferences=False)
    runtime = tmp_path / "runtime_profile"

    # Preferences absent from source: not copied, not fatal.
    assert browser_service._sync_profile_snapshot(str(src), str(runtime)) is True
    out = capsys.readouterr().out
    assert "Preferences -> OK" not in out
    assert "FAILED" not in out


def test_harvest_fails_when_no_chrome_profile_found(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(browser_service, "_find_chrome_profile_with_reddit_session", lambda: None)
    assert browser_service._harvest_chrome_session(str(tmp_path / "bot_profile")) is False
    assert "no Chrome profile with a valid reddit_session" in capsys.readouterr().out


def test_harvest_succeeds_and_logs_artifacts(monkeypatch, capsys, tmp_path):
    src = _make_source_profile(tmp_path)
    # Harvest sources from a Chrome profile dir (the Default-equivalent level).
    monkeypatch.setattr(
        browser_service, "_find_chrome_profile_with_reddit_session", lambda: src / "Default"
    )
    target = tmp_path / "bot_profile"

    assert browser_service._harvest_chrome_session(str(target)) is True
    out = capsys.readouterr().out
    assert "harvest: Cookies -> OK" in out
    assert "Copied Reddit session" in out
    assert (target / "Default" / "Cookies").exists()


def test_harvest_fails_hard_when_cookie_db_absent_after_copy(monkeypatch, capsys, tmp_path):
    src = _make_source_profile(tmp_path, cookies=False, network=False)
    monkeypatch.setattr(
        browser_service, "_find_chrome_profile_with_reddit_session", lambda: src / "Default"
    )
    target = tmp_path / "bot_profile"

    assert browser_service._harvest_chrome_session(str(target)) is False
    out = capsys.readouterr().out
    assert "harvest FAILED: no cookie DB present" in out


def test_cookie_db_present_requires_nonempty_file(tmp_path):
    default = tmp_path / "Default"
    default.mkdir()
    assert browser_service._cookie_db_present(default) is False

    (default / "Cookies").write_bytes(b"")  # empty file must not count
    assert browser_service._cookie_db_present(default) is False

    (default / "Cookies").write_bytes(b"data")
    assert browser_service._cookie_db_present(default) is True
