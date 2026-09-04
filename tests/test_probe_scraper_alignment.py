"""A probe must ask the same service its scraper asks.

When they diverge, the failure is a *loop* rather than a wrong answer, and it is
invisible in any single run because each side looks locally correct:

  live_rippling read ats.rippling.com/<slug>/jobs while _scrape_rippling reads
  api.rippling.com/platform/api/ats/v1/board/<slug>/jobs. The board page answers
  200 for companies whose API board is gone, so the validator revived the mark,
  the scraper's next pass got a 404 and marked it dead again, and the two traded
  the same three slugs indefinitely -- spending requests on both sides every
  cycle, forever.

Static, so it costs nothing and cannot flake: it reads the URLs out of the two
functions' source. That is enough to catch a whole-host divergence, which is the
shape that produced the loop. Path differences on the same host are deliberate
in several places -- teamtailor probes /jobs and scrapes /jobs.json -- and were
checked live and found to agree, so they are not asserted here.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import validate_ats_slugs as v  # noqa: E402
from services import ats_service  # noqa: E402

_URL = re.compile(r"https://[^\"')\s]+")
#: Any f-string placeholder becomes a wildcard: the tenant's name is not the
#: part being compared, the service is.
_PLACEHOLDER = re.compile(r"\{[^}]*\}")


def _hosts(fn) -> set[str]:
    out = set()
    for url in _URL.findall(inspect.getsource(fn)):
        host = url.split("//", 1)[1].split("/", 1)[0]
        out.add(_PLACEHOLDER.sub("*", host))
    return out


@pytest.mark.parametrize("platform", sorted(ats_service.ATS_PLATFORMS))
def test_probe_and_scraper_talk_to_the_same_service(platform):
    probe_hosts = _hosts(v.PROBES[platform])
    scraper_hosts = _hosts(ats_service._SCRAPERS[platform])
    assert probe_hosts, f"no URL found in the {platform} probe"
    assert scraper_hosts, f"no URL found in the {platform} scraper"
    assert probe_hosts & scraper_hosts, (
        f"{platform}: the probe asks {sorted(probe_hosts)} but the scraper asks "
        f"{sorted(scraper_hosts)}. When these differ, one can call a company "
        f"live while the other cannot read it, and the dead mark oscillates "
        f"forever."
    )


def test_every_platform_is_covered():
    """A platform with no probe or no scraper would be skipped silently."""
    assert set(ats_service.ATS_PLATFORMS) == set(v.PLATFORMS)
    for platform in ats_service.ATS_PLATFORMS:
        assert platform in v.PROBES
        assert platform in ats_service._SCRAPERS


def test_the_check_can_actually_fail():
    """Guards the guard.

    _hosts returning an empty set for everything would make the assertion above
    vacuous, and the parametrised test would pass for all sixteen platforms
    while checking nothing.
    """
    def probe_like():
        return "https://ats.rippling.com/x/jobs"

    def scraper_like():
        return "https://api.rippling.com/platform/api/ats/v1/board/x/jobs"

    assert _hosts(probe_like) == {"ats.rippling.com"}
    assert _hosts(scraper_like) == {"api.rippling.com"}
    assert not (_hosts(probe_like) & _hosts(scraper_like))


def test_placeholders_do_not_hide_a_real_difference():
    """"{slug}.breezy.hr" and "{slug}.teamtailor.com" must not both collapse to
    the same wildcard host, or every platform would trivially match."""
    def a():
        return "https://{slug}.breezy.hr/json"

    def b():
        return "https://{slug}.teamtailor.com/jobs.json"

    assert not (_hosts(a) & _hosts(b))
