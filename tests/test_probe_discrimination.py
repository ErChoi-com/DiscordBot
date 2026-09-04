"""The probe self-check must catch both ways a probe fails.

Probes decide whether a company's board still exists, and the answer is written
to data/dead_slugs/, which the bot reads. A probe stuck on True marks nothing
dead and looks exactly like a healthy platform; a probe stuck on False marks
every company on the platform dead. Neither announces itself.

The script itself talks to all 16 platforms, so it cannot be exercised here.
`assess` holds the judgement and is pure, which is the point of splitting it
out -- the live half is a transport detail, the decision is not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_probe_discrimination as c  # noqa: E402


def test_a_healthy_probe_passes():
    assert c.assess([False, False, False], [True, True, False]) == c.OK
    # One live answer is enough: a board in the confirmed-live file can still
    # have closed since it was last checked, so demanding all of them would
    # fail honest probes.
    assert c.assess([False, False, False], [False, True, False]) == c.OK


def test_a_probe_that_cannot_say_no_is_caught():
    """Reports every company live and marks nothing dead."""
    assert c.assess([True, True, True], [True, True, True]) == c.STUCK_TRUE


def test_a_probe_that_cannot_say_yes_is_caught():
    """The destructive direction: every real company marked dead for a TTL.

    The historical one-liner only probed invented slugs, so this half went
    unchecked -- a probe in this state passed it perfectly.
    """
    assert c.assess([False, False, False], [False, False, False]) == c.STUCK_FALSE


def test_an_outage_is_not_a_broken_probe():
    """Unreachable and raised exceptions are neither a yes nor a no.

    Reading a refusal as "cannot say yes" would report every probe broken
    during an outage -- the same false-closure shape that once made lever and
    iCIMS mark companies dead when one of two hosts merely refused.
    """
    assert c.assess([False, False, False], [None, None, True]) == c.OK
    assert c.assess([False, False, False], [None, None, None]) == c.NO_DATA
    assert c.assess([None, None, None], [True, True, True]) == c.NO_DATA
    assert c.assess([False, False, False],
                    [RuntimeError("boom"), True]) == c.OK


def test_no_answers_at_all_does_not_pass_by_default():
    """Silence must not read as health -- that is the bug class this guards."""
    assert c.assess([], []) == c.NO_DATA
    assert c.assess([False], []) == c.NO_DATA


def test_non_bool_answers_are_not_counted_as_yes():
    """A probe returning a truthy non-bool (a Response, a dict) is not a yes."""
    assert c.assess([False, False], [{"live": True}, "yes"]) == c.NO_DATA


@pytest.mark.parametrize("platform", [
    "greenhouse", "lever", "ashby", "workday", "icims", "bamboohr", "workable",
    "breezy", "smartrecruiters", "recruitee", "teamtailor", "rippling",
    "jazzhr", "jobvite", "applicantpro", "paylocity",
])
def test_every_platform_is_covered_by_the_checker(platform):
    """A platform with no probe would be skipped silently by the checker."""
    import validate_ats_slugs as v

    assert platform in v.PROBES
    assert platform in v.PLATFORMS


def test_bogus_slugs_are_plural():
    """One invented slug is not enough.

    A single 404 from an overloaded edge node reads as a working probe, so a
    lone sample would let a stuck-true probe through on luck.
    """
    assert len(c.BOGUS_SLUGS) >= 3
    assert len(set(c.BOGUS_SLUGS)) == len(c.BOGUS_SLUGS)


# ── the third direction: the vendor's own hostnames ─────────────────────────
#
# Invented slugs 404 everywhere, so a probe that reads the vendor's own host as
# a tenant passes both of the original checks. Every VENDOR_HOSTS entry was a
# real false positive before the probes learned to check where the response
# landed: api.breezy.hr serves Breezy's API docs, www4.icims.com redirects to
# iCIMS's marketing site, and login.recruitee.com lands on a *different
# tenant's* board -- which would file that company's jobs under "login".

def test_a_vendor_host_read_as_live_is_caught():
    assert c.assess([False, False], [True, True], [True]) == c.VENDOR_LIVE
    assert c.assess([False, False], [True, True], [False, True]) == c.VENDOR_LIVE


def test_vendor_hosts_reading_dead_is_the_healthy_case():
    assert c.assess([False, False], [True, True], [False, False]) == c.OK


def test_no_vendor_hosts_configured_is_not_a_failure():
    """Only the subdomain platforms have a known vendor host; the rest pass
    without one rather than being reported as unchecked."""
    assert c.assess([False, False], [True, True], []) == c.OK
    assert c.assess([False, False], [True, True]) == c.OK


def test_an_unreachable_vendor_host_is_not_a_verdict():
    """Same rule as the other two directions: an outage is not evidence."""
    assert c.assess([False, False], [True, True], [None]) == c.OK
    assert c.assess([False, False], [True, True],
                    [RuntimeError("boom")]) == c.OK


def test_the_more_fundamental_failures_are_reported_first():
    """A stuck probe is broken in a way that makes the vendor result noise."""
    assert c.assess([True, True], [True], [True]) == c.STUCK_TRUE
    assert c.assess([False, False], [False, False], [True]) == c.STUCK_FALSE


@pytest.mark.parametrize("platform, hosts", sorted(c.VENDOR_HOSTS.items()))
def test_every_configured_vendor_host_belongs_to_a_real_platform(platform, hosts):
    import validate_ats_slugs as v

    assert platform in v.PROBES
    assert hosts, "an empty tuple silently disables the check for this platform"


def test_vendor_hosts_cover_the_platforms_that_needed_the_fix():
    """These five are the host-based probes that check where a 200 landed.

    A platform dropping out of this map would lose the only live check for the
    regression it was added for.
    """
    assert {"icims", "breezy", "recruitee", "bamboohr", "jazzhr"} <= set(c.VENDOR_HOSTS)


def test_check_platform_actually_probes_the_vendor_hosts(monkeypatch):
    """Pins the wiring, not the judgement.

    assess() can be entirely correct while check_platform never gathers the
    vendor answers -- a mutation that passed an empty list left every other test
    in this file passing, because they all call assess() directly.
    """
    import random

    import validate_ats_slugs as v

    asked: list[str] = []

    def probe(slug):
        asked.append(slug)
        return slug.startswith("real")

    monkeypatch.setitem(v.PROBES, "breezy", probe)
    monkeypatch.setattr(v, "load_checked", lambda p: {"real1": "2026-09-01"})
    monkeypatch.setitem(c.VENDOR_HOSTS, "breezy", ("api",))

    verdict, bogus, live, vendor = c.check_platform("breezy", 1, random.Random(0))
    assert "api" in asked, "the vendor host was never probed"
    assert vendor == [False]
    assert verdict == c.OK


def test_check_platform_reports_a_vendor_host_that_reads_live(monkeypatch):
    """And the verdict has to come back out, not just be gathered."""
    import random

    import validate_ats_slugs as v

    monkeypatch.setitem(v.PROBES, "breezy", lambda slug: True)
    monkeypatch.setattr(v, "load_checked", lambda p: {"real1": "2026-09-01"})
    monkeypatch.setitem(c.VENDOR_HOSTS, "breezy", ("api",))

    verdict, _bogus, _live, vendor = c.check_platform("breezy", 1, random.Random(0))
    # Invented slugs also come back True here, so the more fundamental failure
    # is the one reported -- that ordering is deliberate.
    assert verdict == c.STUCK_TRUE
    assert vendor == [True]
