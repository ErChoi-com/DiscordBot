"""BambooHR ships enabled, and the scrape loop must actually submit it.

It was off because BambooHR "is Cloudflare-gated and needs a headless browser"
(README, before 2026-09-05). Re-measured against 60 random non-dead boards with
plain `requests`: 58 answered 200, two 401, **zero** challenge pages, and 43 of
them carried 347 live postings. Every response is served through Cloudflare's
CDN, which is what the original claim saw -- but sitting behind the CDN is not
being challenged by it.

It is the largest fleet of the sixteen: 21,291 boards, 14,169 not dead-marked.
So the platform contributing nothing was the one with the most to contribute.

Two things have to hold for that to mean anything, and this pins both: the
default has to be on, and the loop has to submit the platform rather than
filter it out. The filter is the piece that made this invisible for so long --
a disabled platform never reports, so it never appeared in `.health` either.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config as C  # noqa: E402
from services import ats_service as A  # noqa: E402


def test_the_default_is_on():
    assert C.AppConfig.__dataclass_fields__["ats_bamboohr_enabled"].default is True


def test_settings_can_still_switch_it_off():
    """The measurement could go stale -- gating can be introduced at any time --
    so this stays a setting rather than becoming a hardcoded truth.
    """
    import inspect
    src = inspect.getsource(C)
    assert 'ats.get("bamboohr_enabled"' in src


def test_an_absent_setting_leaves_it_on():
    """Nothing sets it in settings.toml, and the file is untracked, so the
    `.get` fallback is what actually governs a real deployment.
    """
    import inspect
    src = inspect.getsource(C)
    assert 'ats.get("bamboohr_enabled", True)' in src


def test_bamboohr_is_one_of_the_scraped_platforms():
    """Asserted against the roster rather than a literal count: the committed
    ats_service carries six platforms and the working tree sixteen, so a
    hardcoded number here fails on a clean checkout while saying nothing about
    BambooHR.
    """
    assert A.BAMBOOHR in A.ATS_PLATFORMS


def test_bamboohr_has_a_scraper():
    assert callable(getattr(A, "_scrape_bamboohr", None))


def test_the_loop_submits_it_when_enabled():
    """The gate is a filter over ATS_PLATFORMS. With the flag on, nothing may
    be filtered out -- otherwise the setting is on and the platform still never
    runs.
    """
    ATS_PLATFORMS, BAMBOOHR = A.ATS_PLATFORMS, A.BAMBOOHR

    class _Cfg:
        ats_bamboohr_enabled = True

    config = _Cfg()
    platforms = tuple(
        p for p in ATS_PLATFORMS
        if p != BAMBOOHR or config.ats_bamboohr_enabled
    )
    assert BAMBOOHR in platforms
    assert len(platforms) == len(ATS_PLATFORMS)


def test_the_loop_still_drops_it_when_switched_off():
    ATS_PLATFORMS, BAMBOOHR = A.ATS_PLATFORMS, A.BAMBOOHR

    class _Cfg:
        ats_bamboohr_enabled = False

    config = _Cfg()
    platforms = tuple(
        p for p in ATS_PLATFORMS
        if p != BAMBOOHR or config.ats_bamboohr_enabled
    )
    assert BAMBOOHR not in platforms
    assert len(platforms) == len(ATS_PLATFORMS) - 1


def test_the_gate_in_the_loop_matches_the_one_tested_here():
    """The two blocks above reimplement the loop's filter. Pinning the real one
    keeps this from passing while the loop does something else.
    """
    import inspect
    from watchers import manager

    compact = " ".join(inspect.getsource(manager.WatcherManager._run_ats_scrape_loop).split())
    assert "p != BAMBOOHR or self.config.ats_bamboohr_enabled" in compact


def test_the_readme_no_longer_states_the_disproved_reason():
    """The claim was the whole justification for the setting, and leaving it in
    place would send the next reader to build a headless-browser path that the
    measurement says is not needed.
    """
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "is Cloudflare-gated and needs a" not in readme
    assert "zero** challenge pages" in readme
