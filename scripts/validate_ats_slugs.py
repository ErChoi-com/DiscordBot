"""Probe harvested ATS company slugs and record which boards are actually gone.

Harvesting is optimistic: an archived URL proves a board existed when it was
crawled, not that it exists now. Measured live rates for newly harvested slugs
run from 87% (BambooHR) down to 11% (iCIMS), so a list that is only ever added
to fills with companies that will never answer, and every one of them costs a
request per scrape cycle forever.

This closes that loop. It asks each platform whether a board exists and writes
the misses into ``data/dead_slugs/<platform>.json`` -- the same file, the same
``{slug: YYYY-MM-DD}`` shape, and the same meaning that ``ats_service`` already
uses, so the bot needs no new concept to benefit. Nothing is deleted: a dead
mark suppresses a slug until ``DEAD_SLUG_RECHECK_DAYS`` elapses and then it is
probed again, which is what lets a board that was down for a day come back.

Why probe here rather than let the bot discover deadness: the bot only learns a
slug is dead by scraping it during a real cycle, so a freshly harvested batch
costs thousands of wasted requests spread over weeks of user-facing runs. Doing
it once, in CI, in parallel, gets the same answer before the slugs ever reach a
scrape cycle.

Usage:
    python scripts/validate_ats_slugs.py                    # everything unchecked
    python scripts/validate_ats_slugs.py --platform lever
    python scripts/validate_ats_slugs.py --limit 2000       # bound a CI run
    python scripts/validate_ats_slugs.py --sample 200 --dry-run
    python scripts/validate_ats_slugs.py --recheck-dead     # re-probe dead marks

Exit codes:
    0  validation ran
    1  every platform failed to probe (network or endpoints changed)
    2  harness error
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harvest_ats as _harvest  # noqa: E402  (same directory)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
COMPANY_DIR = DATA_DIR / "ats_companies"
HARVEST_DIR = DATA_DIR / "ats_harvest"
DEAD_DIR = DATA_DIR / "dead_slugs"
# Slugs confirmed live, and when. Separate from dead_slugs because ats_service
# owns that directory and reads every file in it by platform name.
CHECKED_DIR = DATA_DIR / "ats_checked"

PLATFORMS = ("greenhouse", "lever", "ashby", "workday", "icims", "bamboohr",
             "paylocity", "workable", "breezy", "smartrecruiters",
             "rippling", "teamtailor", "jazzhr", "recruitee", "jobvite",
             "applicantpro")

COMPANY_FILES = {p: f"{p}_companies.json" for p in PLATFORMS}
# Upstream ships this one under a different name, and as {guid, name, jobs}
# objects rather than a flat list of identifiers.
COMPANY_FILES["paylocity"] = "paylocity_companies_clean.json"

# Mirrors ats_service.DEAD_SLUG_RECHECK_DAYS. A slug marked dead more recently
# than this is not re-probed, so repeated runs cost nothing for known-dead
# companies while still letting them back in eventually.
RECHECK_DAYS = 7

# How long a confirmed-live slug is left alone before being probed again.
#
# Without this the validator has no memory of a live answer, so every run
# re-probes every slug it has ever confirmed. Measured on a full run: 39,043
# probes, 36,438 of them live, and the due count afterwards was unchanged --
# lever probed 2,469 slugs and still reported 2,469 due. The budget was being
# spent almost entirely on re-confirming companies already known to be there,
# which is why the backlog never moved.
#
# Thirty days is well inside the 90-day dead TTL, so a board that closes is
# still noticed within a month, and it leaves each run free to spend its budget
# on slugs nobody has checked.
#
# Measured after: the next full run took the due count from 53,498 to 15,429 in
# 45 minutes, against 2,229 in 49 minutes for the run before it. Five of the
# seven platforms went to zero or single digits; what remains is BambooHR and
# Paylocity, the two largest, and Paylocity only because it defers most of its
# work at two workers.
LIVE_RECHECK_DAYS = 30

REQUEST_TIMEOUT = 25

# Wall-clock ceiling for one platform's audit sample. The audit is a
# measurement, not work: a partial sample still estimates the rate, while an
# unbounded one can outlast the run it annotates.
AUDIT_BUDGET_SECONDS = 240.0

# Per-platform concurrency. These hit real ATS endpoints, so they stay at or
# below what ats_service already uses for the same host.
WORKERS = {"greenhouse": 16, "lever": 16, "ashby": 8, "workday": 12,
           "icims": 12, "bamboohr": 12,
           # Paylocity sheds load by refusing connections rather than slowing
           # down, and it does so steeply. Measured over 60-slug batches:
           #
           #   workers  unreachable  raw rate  useful rate
           #      2         0/60      3.2/s      3.2/s
           #      4        20/60      5.3/s      3.5/s
           #      8        48/60     13.2/s      2.7/s
           #     12        53/60     21.5/s      2.5/s
           #
           # Raw throughput keeps climbing while useful throughput does not:
           # past two workers most requests come back with no verdict, so the
           # extra concurrency buys refusals. A refusal is recorded unknown and
           # changes no marks, so the cost is wasted requests rather than wrong
           # answers -- but the slug still has to be probed again later.
           "paylocity": 2,
           # No measured concurrency curve for these two yet, so they take the
           # conservative default rather than a guess that reads as evidence.
           "breezy": 8, "smartrecruiters": 8, "rippling": 8,
           "teamtailor": 8, "jazzhr": 8, "recruitee": 8, "jobvite": 8,
           "applicantpro": 8, "recruitee": 4,
           # Paced rather than throttled down. See DELAYS below.
           "workable": 4}

# Seconds to wait before each probe, per platform. Only for endpoints that
# refuse a normal pace: an unpaced Workable pass at eight workers ran about
# 45/s, came back 91% refused with 429, and left the address throttled for
# minutes afterwards.
#
# The delay is what fixes that, not low concurrency, and the difference is
# worth 4x. Measured after the throttle cleared, 429s in every case zero:
#
#   serial, 0.5s   80 requests   1.5/s
#   serial, 0.25s  80 requests   2.4/s
#   4 workers, 0.25s  60 requests  10.0/s
#
# So four workers with a quarter-second pace, rather than the one worker a
# guess had put here -- same safety, four times the throughput. Eight workers
# with a delay is untested; the unpaced eight-worker run is the one that broke.
DELAYS: dict[str, float] = {"workable": 0.25, "recruitee": 0.25}

# Most probes a single run may spend on a platform, for endpoints that meter a
# quota rather than a rate. Workable is the case: pacing alone does not buy
# more answers, it only spreads the same allowance out.
#
# Two full runs measured the size of that allowance. The first answered 601 of
# 6,843 before every later request came back 429; the second, paced to a
# quarter-second across four workers, answered 437 of 3,056 before the same
# thing happened. So roughly 500-600 answers are available per window, and the
# 5,000-odd requests after that are spent to be refused -- and refusals are
# what deepens the throttle for the run after.
#
# Capping at the measured allowance costs nothing: whatever is not probed is
# deferred, and target selection prefers never-probed slugs, so the platform
# still converges. It just converges over several runs instead of burning nine
# tenths of each one.
# Recruitee meters the same way: a full pass probed 2,581 and answered 718 --
# 716 live, 2 dead -- with the remaining 1,863 refused. Back-to-back rate tests
# could not size it any more precisely, because each test runs inside the
# throttle the previous one earned; the allowance from a cold start is the only
# figure worth trusting.
MAX_PROBES: dict[str, int] = {"workable": 600, "recruitee": 700}

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36")


class Unreachable(Exception):
    """The probe could not reach the platform at all (as opposed to a 404)."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Report a redirect as itself instead of following it."""

    def redirect_request(self, *args, **kwargs):  # noqa: D102
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)


def _request(url: str, *, method: str = "GET", payload: dict | None = None,
             timeout: int = REQUEST_TIMEOUT,
             follow_redirects: bool = True,
             read_bytes: int = 2048) -> tuple[int | None, bytes, str]:
    """Returns (status, first bytes, final url). None status means no answer.

    The final URL matters: BambooHR answers an unknown tenant with a 200 that
    redirects to its marketing site, so status alone cannot tell them apart.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json,*/*"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    opener = urllib.request.urlopen if follow_redirects else _NO_REDIRECT_OPENER.open
    try:
        with opener(req, timeout=timeout) as resp:
            return resp.status, resp.read(read_bytes), resp.geturl()
    except urllib.error.HTTPError as exc:
        return exc.code, b"", url
    except Exception:
        return None, b"", url


# --------------------------------------------------------------------------
# Per-platform liveness. Each returns True (board exists), False (it does not),
# or raises Unreachable when the answer is "we could not tell".
#
# Every one of these was calibrated against known-live and known-dead slugs;
# the obvious URL is wrong for four of the six platforms.
# --------------------------------------------------------------------------

def _decide(status: int | None) -> bool:
    if status is None:
        raise Unreachable("no response")
    if status == 200:
        return True
    if status in (301, 302, 404, 410, 422):
        return False
    # 403 says the endpoint refused to answer, not that the company is gone --
    # it is also what a blocked or rate-limited client is served, so reading it
    # as "absent" would record a block as a mass closure and suppress live
    # boards for the full dead TTL. Precautionary rather than observed: sampling
    # known-dead slugs found greenhouse, ashby and lever answer 404 and workday
    # 404/422, and no platform returned 403 for a genuinely missing board. So
    # this costs nothing today and closes the path if one starts blocking.
    # 5xx and rate limits likewise say nothing about whether the company exists.
    raise Unreachable(f"HTTP {status}")


def live_greenhouse(slug: str) -> bool:
    status, _, _ = _request(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs")
    return _decide(status)


# Lever's US and EU regions are separate deployments and a company exists in
# exactly one: amicustherapeutics is 200 on api.eu.lever.co and 404 on
# api.lever.co. Checking one host marks the entire EU population dead.
LEVER_API_HOSTS = ("api.lever.co", "api.eu.lever.co")


def live_lever(slug: str) -> bool:
    unreachable = 0
    for host in LEVER_API_HOSTS:
        status, _, _ = _request(f"https://{host}/v0/postings/{slug}?mode=json")
        try:
            if _decide(status):
                return True
        except Unreachable:
            unreachable += 1
    # Dead only when every region gave a definite answer and all of them said
    # absent. A region that refused told us nothing, and the company may live
    # there: one refusal paired with one 404 used to fall through to "dead",
    # which is a false closure produced by load rather than by the company
    # going away -- and refusals are most likely exactly when a pass is running
    # sixteen workers against the API.
    if unreachable:
        raise Unreachable(f"{unreachable} region(s) did not answer")
    return False


def live_ashby(slug: str) -> bool:
    # jobs.ashbyhq.com is a single-page app that answers 200 for every route,
    # including nonsense, so it cannot distinguish a real board. The posting API
    # 404s properly.
    status, _, _ = _request(f"https://api.ashbyhq.com/posting-api/job-board/{slug}")
    return _decide(status)


def live_bamboohr(slug: str) -> bool:
    # An unknown tenant is also 200: it redirects to www.bamboohr.com. The tells
    # are the final URL leaving the tenant host and a real board answering JSON
    # rather than a marketing page.
    status, body, final = _request(f"https://{slug}.bamboohr.com/careers/list")
    if status is None:
        raise Unreachable("no response")
    if status == 401:
        # A board that demands authentication is one the bot can never read, so
        # it is unusable rather than unknown. Without this the answer falls
        # through to _decide, which has no rule for 401 and raises -- leaving
        # these permanently unresolved and re-probed every run. That was 358 of
        # bamboohr's 1,012 probes in one run, a third of the platform's budget
        # spent on slugs that cannot produce an answer.
        #
        # Measured before changing the meaning: 40 such slugs returned 401 on
        # two passes twenty seconds apart, identical both times, while fifteen
        # confirmed-live and fifteen known-dead tenants all answered 200. So
        # 401 is a property of these tenants, not of the client or the moment.
        # Marking them dead rather than unscrapeable keeps the 90-day recheck,
        # which is what should happen if a board is later made public.
        return False
    if status != 200:
        return _decide(status)
    if f"{slug}.bamboohr.com" not in final:
        return False
    return body.lstrip()[:1] in (b"{", b"[")


def live_icims(slug: str) -> bool:
    # iCIMS runs two mutually exclusive host conventions. Over a 120-slug
    # sample, 46 resolved only as careers-<slug>.icims.com, 34 only as
    # <slug>.icims.com, and zero resolved both ways -- so testing one form
    # misreports roughly 42% of live boards as dead.
    forms = [slug] if slug.startswith("careers-") else [f"careers-{slug}", slug]
    unreachable = 0
    for host in forms:
        status, _, final = _request(f"https://{host}.icims.com/sitemap.xml")
        # An iCIMS infrastructure host answers 200 and redirects to the vendor's
        # own site: www4.icims.com/sitemap.xml ends at www.icims.com. Without
        # this the probe reads that as a live board, which is how www4 came to
        # sit in the candidate list looking valid. Real tenants stay on their
        # own host -- measured across four of them, every 200 kept its hostname.
        # Same tell live_bamboohr uses, for the same reason.
        if status == 200 and final and f"{host}.icims.com" not in final:
            continue
        try:
            if _decide(status):
                return True
        except Unreachable:
            unreachable += 1
    # Same rule as lever: the two host forms are mutually exclusive, so a 404
    # on one form is only evidence of absence if the other form actually
    # answered. iCIMS has the lowest live rate and the most refusals of any
    # platform here, so this is where a partial-answer false closure was most
    # likely to happen.
    if unreachable:
        raise Unreachable(f"{unreachable} host form(s) did not answer")
    return False


def live_paylocity(slug: str) -> bool:
    """Paylocity answers 200 for an unknown company -- but only if you follow
    the redirect it sends first.

    A missing board 302s to a shell titled "Job Not Found"; a real one answers
    200 directly. Following redirects collapses both to 200 and forces reading
    the body to tell them apart, which is what this used to do. Not following
    them makes the status itself the answer.

    Sturdier, not faster. A single dead board resolves in 0.2s against 1.4s,
    but end-to-end throughput did not move: 2.4/s on a live-heavy batch and
    3.0/s on a mixed one, against 3.2/s for the body check. Live boards still
    send their page either way, and they dominate. The reason to prefer this is
    that a redirect is structural while a page title is copy -- "Job Not Found"
    can be reworded and the probe would start calling every dead board live,
    silently.

    Checked against the body-based version on 14 real GUIDs plus a synthetic
    one: identical verdicts throughout.
    """
    status, _, _ = _request(
        f"https://recruiting.paylocity.com/recruiting/jobs/All/{slug}",
        follow_redirects=False)
    if status is None:
        raise Unreachable("no response")
    if status == 200:
        return True
    if str(status).startswith("3"):
        return False
    return _decide(status)


def live_workday(slug: str) -> bool:
    # The CXS endpoint needs a real JSON body: without one a live tenant answers
    # 500 and a nonexistent one 422, and the plain page GET is an SPA that 200s
    # for anything. Only a bodied POST separates them.
    parts = slug.split("|")
    if len(parts) != 3:
        return False
    tenant, host, site = parts
    status, _, _ = _request(
        f"https://{tenant}.{host}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs",
        method="POST",
        payload={"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""},
    )
    return _decide(status)


def live_workable(slug: str) -> bool:
    # The board page itself is a JS shell that answers 200 for anything; the
    # widget account endpoint is what separates a real account from a missing
    # one. Verified against 25 harvested slugs, all 25 live.
    status, _, _ = _request(
        f"https://apply.workable.com/api/v1/widget/accounts/{slug}")
    return _decide(status)


def live_breezy(slug: str) -> bool:
    # Breezy gives each company a subdomain and 404s an unknown one outright,
    # so the status is the whole answer. Verified against 25 harvested slugs,
    # 24 live.
    # ...except when it redirects instead. api.breezy.hr answers 200 and lands
    # on developer.breezy.hr, Breezy's API documentation, which the status alone
    # reads as a live board. Same tell as bamboohr and jazzhr: a real tenant's
    # response stays on the tenant's own host.
    status, _, final = _request(f"https://{slug}.breezy.hr/")
    if status == 200 and final and f"{slug}.breezy.hr" not in final:
        return False
    return _decide(status)


def live_smartrecruiters(slug: str) -> bool:
    # The postings API answers 200 for anything, so status says nothing: an
    # unknown company returns a well-formed {"totalFound":0,"content":[]}. Only
    # the count separates it from a real board. The /companies/{slug} endpoint
    # would be the honest test but 404s without credentials, including for
    # companies that certainly exist.
    #
    # A zero count used to end it, which conflated "no such company" with "a
    # real company advertising nothing today". Measured: of six candidates
    # returning totalFound=0, four had a real careers page. Those were being
    # marked dead, and the bot skips dead slugs -- so the first job each of them
    # posted would be missed until a recheck cleared the mark.
    #
    # The careers page settles it, because it answers about the *company*
    # rather than its postings: a real slug stays at
    # careers.smartrecruiters.com/<slug>, an unknown one is redirected to the
    # bare jobs.smartrecruiters.com. Same redirect tell as the subdomain
    # platforms. It is only consulted when the count is zero, so the ordinary
    # case still costs one request.
    #
    # /v1/companies/<slug> would be the direct existence test, but it 404s
    # without credentials for real companies too -- re-verified against five
    # known-live slugs, all 404.
    # Read past the default cap. The verdict depends on parsing the body, and a
    # truncated JSON document raises, which this turns into "unknown". That
    # failure is not neutral: a company with postings returns the long response
    # and would go unknown, while one with none returns a short
    # {"totalFound":0,...} that parses cleanly and is marked dead. So truncation
    # would bias the platform toward dead precisely where it matters. Measured
    # at limit=1 across twenty real companies, none truncated at 2048 -- but the
    # margin is not worth relying on when the fix is a larger read.
    status, body, _ = _request(
        f"https://api.smartrecruiters.com/v1/companies/{slug}/postings?limit=1",
        read_bytes=16384)
    if status != 200:
        return _decide(status)
    try:
        if int(json.loads(body or b"{}").get("totalFound", 0)) > 0:
            return True
    except (ValueError, TypeError, AttributeError):
        raise Unreachable("unparseable postings response")
    return _smartrecruiters_company_exists(slug)


def _smartrecruiters_company_exists(slug: str) -> bool:
    """Whether the careers page belongs to *slug* rather than being a bounce.

    Answers about the company, not its postings, which is what separates a real
    board advertising nothing from one that never existed.
    """
    status, _, final = _request(f"https://careers.smartrecruiters.com/{slug}")
    if status != 200:
        # A refusal here must not become a dead mark: the postings call already
        # said "no jobs", which is not evidence of absence on its own.
        return _decide(status)
    if not final:
        return False
    # The *path*, not a substring of the URL. The bounce target is
    # jobs.smartrecruiters.com, so for a company slugged "jobs" or "careers" the
    # slug appears inside the host of the very redirect that means "no such
    # company" -- and "/jobs" is even a substring of "https://jobs..." because
    # of the two slashes in the scheme. Comparing the path is the only form that
    # cannot read its own failure as success.
    return urllib.parse.urlparse(final).path.strip("/").lower() == slug.lower()


def live_rippling(slug: str) -> bool:
    # The board page and the API disagree, and the API is the one that matters:
    # it is what _scrape_rippling reads, so a company this calls live is one the
    # bot can actually pull jobs from.
    #
    # The board page 200s for companies whose API board 404s. That produced a
    # loop rather than a one-off wrong answer: this probe revived the mark, the
    # scraper's next pass got a 404 from the API and marked it dead again, and
    # the pair traded the same three slugs indefinitely, spending requests on
    # both sides each cycle. qucareers, rabot-energy and whitehatgaming were
    # each marked dead on 2026-09-03 and answering "live" here the next day.
    #
    # Verified: the API 404s invented slugs and answers 200 for all ten
    # known-live companies sampled, including ones advertising nothing -- so
    # this does not reintroduce the "no jobs means no company" conflation that
    # smartrecruiters had.
    status, _, _ = _request(
        f"https://api.rippling.com/platform/api/ats/v1/board/{slug}/jobs")
    return _decide(status)


def live_teamtailor(slug: str) -> bool:
    # Teamtailor 404s its own non-tenant hosts (www, api, support, blog all
    # verified), so nothing is redirecting today. The host check is here anyway
    # because the other four subdomain platforms all needed it eventually, and
    # a vendor that starts redirecting would otherwise turn every colliding
    # slug live without anything failing.
    status, _, final = _request(f"https://{slug}.teamtailor.com/jobs")
    if status == 200 and final and f"{slug}.teamtailor.com" not in final:
        return False
    return _decide(status)


def live_jazzhr(slug: str) -> bool:
    # Every subdomain answers 200; an unknown one is bounced to
    # www.jazzhr.com/job-seekers. The tell is whether the response is still on
    # the tenant's own host, the same shape as bamboohr.
    status, _, final = _request(f"https://{slug}.applytojob.com/apply")
    if status != 200:
        return _decide(status)
    return f"{slug}.applytojob.com" in final


def live_recruitee(slug: str) -> bool:
    # The careers page answers 200 for anything: an unknown subdomain is
    # redirected to recruitee.com and, because redirects are followed, arrives
    # as a 200 like any real board. Probing it could only ever say "live" --
    # three invented company names all came back live, which is the same
    # signature that four other probes were fixed for.
    #
    # The offers API 404s an unknown company outright, so the status is the
    # whole answer. It is also the endpoint _scrape_recruitee reads, so a
    # company this calls live is one the scraper can actually pull jobs from.
    #
    # The offers endpoint redirects too, in two ways that both read as live:
    # blog.recruitee.com lands on recruitee.com/blog, and login.recruitee.com
    # lands on loginsoftware.recruitee.com -- a *different company's* board,
    # which would credit one tenant's jobs to another slug. So the response has
    # to have stayed on the host that was asked for.
    status, _, final = _request(f"https://{slug}.recruitee.com/api/offers/")
    if status == 200 and final and f"{slug}.recruitee.com" not in final:
        return False
    return _decide(status)


def live_jobvite(slug: str) -> bool:
    # An unknown company is redirected to the marketing site's support page
    # with an "invalid" marker; a real one stays on jobs.jobvite.com.
    status, _, final = _request(f"https://jobs.jobvite.com/{slug}")
    if status != 200:
        return _decide(status)
    return "jobs.jobvite.com" in final


def live_applicantpro(slug: str) -> bool:
    # Wildcard DNS, so an unknown tenant resolves and answers 200 on its own
    # host -- the final-URL trick that works for jazzhr does not apply. What
    # separates them is the body: a real board serves an HTML document, while
    # both failure modes serve a bare sentence. "This career site has been
    # disabled" is a real company that switched its board off, and "You may
    # have typed the url ... incorrectly" is no such tenant. Neither can be
    # scraped, so both are not-live.
    status, body, _ = _request(f"https://{slug}.applicantpro.com/jobs/")
    if status != 200:
        return _decide(status)
    return body.lstrip()[:9].lower() == b"<!doctype"


PROBES: dict[str, Callable[[str], bool]] = {
    "greenhouse": live_greenhouse, "lever": live_lever, "ashby": live_ashby,
    "workday": live_workday, "icims": live_icims, "bamboohr": live_bamboohr,
    "paylocity": live_paylocity, "workable": live_workable,
    "breezy": live_breezy, "smartrecruiters": live_smartrecruiters,
    "rippling": live_rippling, "teamtailor": live_teamtailor,
    "jazzhr": live_jazzhr, "recruitee": live_recruitee,
    "jobvite": live_jobvite, "applicantpro": live_applicantpro,
}


# --------------------------------------------------------------------------
# Slug and dead-mark storage, matching ats_service's on-disk contract
# --------------------------------------------------------------------------

def _read_list(path: Path) -> list[str]:
    """Read an identifier list, tolerating upstream's object-shaped Paylocity
    file. Without the dict branch every entry would stringify to "{'guid': ...}"
    and the whole platform would probe as garbage."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for item in data:
        if isinstance(item, dict):
            item = item.get("guid") or item.get("id") or ""
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def load_candidates(platform: str) -> list[str]:
    """Every slug the bot would scrape: upstream list plus local harvest."""
    seen: set[str] = set()
    out: list[str] = []
    for path in (COMPANY_DIR / COMPANY_FILES[platform], HARVEST_DIR / f"{platform}.json"):
        for slug in _read_list(path):
            if slug not in seen:
                seen.add(slug)
                out.append(slug)
    return out


def load_checked(platform: str) -> dict[str, str]:
    """Slugs last confirmed live, and the date. Same shape as the dead map."""
    path = CHECKED_DIR / f"{platform}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def save_checked(platform: str, checked: dict[str, str]) -> None:
    CHECKED_DIR.mkdir(parents=True, exist_ok=True)
    target = CHECKED_DIR / f"{platform}.json"
    tmp = target.with_suffix(f".json.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(checked, indent=2, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def load_dead(platform: str) -> dict[str, str]:
    path = DEAD_DIR / f"{platform}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    # Older revisions stored a bare list; treat those as marked today so they
    # still expire rather than being dropped.
    if isinstance(data, list):
        today = _dt.date.today().isoformat()
        return {str(k): today for k in data}
    return {}


def save_dead(platform: str, dead: dict[str, str]) -> None:
    """Write atomically, in ats_service's exact format.

    write_text truncates in place, so an interrupted write leaves a half-written
    file that ats_service parses as empty -- silently resurrecting every dead
    slug on the platform.
    """
    DEAD_DIR.mkdir(parents=True, exist_ok=True)
    target = DEAD_DIR / f"{platform}.json"
    tmp = target.with_suffix(f".json.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(dead, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


#: Refuse to reconcile if it would drop more than this share of a store. The
#: candidate list is read from files, and a missing or truncated one reads as
#: "few candidates", which would orphan -- and therefore delete -- most of the
#: store. Same shape as COLLAPSE_RATE: the destructive path asks whether the
#: input is plausible before acting on it.
RECONCILE_MAX_DROP_RATE = 0.10
#: ...but always allow a handful. On a small store any real orphan breaches the
#: rate -- rippling holds 35 dead marks, so 4 orphans is 11% -- and a rate-only
#: guard would leave those there permanently. Dropping at most this many cannot
#: do meaningful damage even if the candidate list is wrong.
RECONCILE_MIN_DROP = 5


def reconcile_stores(dead: dict[str, str], checked: dict[str, str],
                     candidates: Iterable[str],
                     log: Callable[[str], None] = lambda _m: None,
                     platform: str = "") -> dict[str, int]:
    """Drop marks for slugs that are no longer in any company list.

    `prune_existing` makes filter fixes retroactive by removing junk from the
    harvest -- bare numbers, ads.txt, favicon.png, locale segments, undecoded
    percent-encoding. Nothing removed the marks those slugs had already
    collected, so they sit in the files the bot loads on every scrape until the
    90-day TTL happens to reach them. Measured before writing this: 539 orphaned
    dead marks and 19 orphaned confirmed-live entries, and the sample is exactly
    the junk list ('100', 'ads.txt', '2fwww', 'en-ca', 'llms.txt').

    Refuses rather than acts when the candidate list looks implausible. An
    empty or truncated harvest file makes *every* mark look orphaned, and this
    deletes what it is given -- so "no candidates" must never read as "nothing
    is real". That failure would be silent and would take the whole platform's
    memory of what is dead with it.
    """
    known = set(candidates)
    stats = {"dead_orphans": 0, "checked_orphans": 0, "refused": 0}
    if not known:
        log(f"[validate] {platform}: no candidates loaded -- refusing to "
            f"reconcile {len(dead)} dead / {len(checked)} confirmed-live marks")
        stats["refused"] = 1
        return stats

    dead_gone = [s for s in dead if s not in known]
    checked_gone = [s for s in checked if s not in known]
    for store, gone, label in ((dead, dead_gone, "dead"),
                               (checked, checked_gone, "confirmed-live")):
        if (store and len(gone) > RECONCILE_MIN_DROP
                and len(gone) / len(store) > RECONCILE_MAX_DROP_RATE):
            log(f"[validate] {platform}: {len(gone)} of {len(store)} {label} "
                f"marks look orphaned -- refusing, the candidate list is "
                f"probably incomplete")
            stats["refused"] = 1
            return stats

    for slug in dead_gone:
        del dead[slug]
    for slug in checked_gone:
        del checked[slug]
    stats["dead_orphans"] = len(dead_gone)
    stats["checked_orphans"] = len(checked_gone)
    if dead_gone or checked_gone:
        log(f"[validate] {platform}: dropped {len(dead_gone)} dead and "
            f"{len(checked_gone)} confirmed-live marks for slugs no longer "
            f"in any company list")
    return stats


def apply_checked(checked: dict[str, str], result: dict,
                  today: _dt.date) -> dict[str, str]:
    """Fold this run's live answers into the confirmed-live map, in place.

    Split out so a dry run can report the same numbers a real one would write
    without re-implementing the rules here -- two copies of this would drift,
    and a dry run that disagrees with the real run is worse than no dry run.

    A slug found dead drops out, so it is never treated as recently-confirmed,
    and entries past twice the recheck window are dropped so the store cannot
    grow without bound.
    """
    stamp = today.isoformat()
    for slug in result["_live"]:
        checked[slug] = stamp
    for slug in result["_dead"]:
        checked.pop(slug, None)
    cutoff = (today - _dt.timedelta(days=LIVE_RECHECK_DAYS * 2)).isoformat()
    for slug in [s for s, when in checked.items() if when < cutoff]:
        del checked[slug]
    return checked


def save_dead_merged(platform: str, result: dict, today: _dt.date,
                     ttl_days: int) -> dict[str, int]:
    """Re-read the on-disk marks, apply this run's verdicts, then write.

    The map this run loaded is minutes to tens of minutes stale by the time it
    is written, and the running bot marks the same files from its own scrape
    cycles -- on this machine it wrote ~3,500 greenhouse marks during a single
    validation pass. Writing back the map we loaded would silently discard
    every one of them, and the bot would do the same to us on its next flush.
    Both writers are individually atomic, which makes the loss invisible rather
    than corrupt.

    Re-reading immediately before the write narrows the race to the write
    itself. It cannot close it without a cross-process lock, but the remaining
    window is milliseconds instead of minutes, and the recheck cycle repairs
    anything lost in it.
    """
    dead_map = load_dead(platform)
    expired = purge_expired(dead_map, today, ttl_days)
    changes = apply_results(dead_map, result, today)
    save_dead(platform, dead_map)

    # Record the live answers too, so the next run does not spend its budget
    # re-confirming them. Read-modify-write for the same reason as the dead
    # map, and a slug found dead drops out of here so it is never treated as
    # recently-confirmed.
    checked = apply_checked(load_checked(platform), result, today)
    save_checked(platform, checked)

    changes["expired"] = expired
    changes["dead_total"] = len(dead_map)
    changes["checked_total"] = len(checked)
    return changes


def _stale(marked: str, today: _dt.date, recheck_days: int) -> bool:
    """True when a dead mark is old enough to be worth re-probing."""
    try:
        when = _dt.date.fromisoformat(marked)
    except ValueError:
        return True
    return (today - when).days >= recheck_days


def select_targets(platform: str, dead: dict[str, str], *, recheck_dead: bool,
                   limit: int | None, sample: int | None, today: _dt.date,
                   recheck_days: int = RECHECK_DAYS, seed: int = 0) -> list[str]:
    """Slugs worth probing this run.

    Unchecked slugs first, then dead marks old enough to deserve another look.
    Ordering matters when --limit bounds the run: spending the budget on slugs
    nobody has ever probed beats re-confirming what we already believe.

    Note what this means for --sample, which draws from the result: a slug
    marked dead too recently to be stale is not a target, so it cannot be
    sampled. The sample is therefore of survivors, and its live rate is not an
    estimate of the population's -- measured on the same data the working pass
    reported 97.5% against a uniform sample's 38.8%. audit_live_rate is the one
    that samples the population.
    """
    candidates = load_candidates(platform)
    # Uncorroborated slugs first among the unchecked ones. What predicts a live
    # board is appearing in BOTH lists, not appearing in either one:
    #
    #                        both    harvest-only   upstream-only
    #     greenhouse        72.0%          42.0%           44.0%
    #     bamboohr          79.2%          51.0%           50.0%
    #     workday           81.6%          62.0%    0.0% (n=6,213)
    #     lever             41.7%          11.7%            1.7%
    #
    # Two independent harvests finding the same slug means something still
    # links to that board. Either list alone is mostly captures of boards that
    # have since closed -- and Workday shows how sharp that can be: every one
    # of a 50-slug sample from its 6,213 upstream-only entries was dead.
    #
    # A bounded run exists to retire dead slugs before the bot spends a request
    # on each every scrape cycle, so it is worth the most probing the group
    # that is mostly dead. Ordering on membership in a single list gets this
    # backwards for exactly the Workday case above, which is why the key is
    # corroboration rather than "is it upstream".
    #
    # This changes the order, never the set.
    upstream = set(_read_list(COMPANY_DIR / COMPANY_FILES[platform]))
    harvested = set(_read_list(HARVEST_DIR / f"{platform}.json"))
    # Skip slugs confirmed live recently. A live answer used to be recorded
    # nowhere, so every run re-probed every company it had ever confirmed and
    # the due count never fell -- lever probed 2,469 slugs and still reported
    # 2,469 due afterwards.
    checked = load_checked(platform)
    fresh = [s for s in candidates
             if s not in dead
             and (recheck_dead
                  or _stale(checked.get(s, ""), today, LIVE_RECHECK_DAYS))]
    fresh.sort(key=lambda s: s in upstream and s in harvested)
    stale = [s for s in candidates
             if s in dead and (recheck_dead or _stale(dead[s], today, recheck_days))]
    ordered = fresh + stale
    if sample is not None and len(ordered) > sample:
        rng = random.Random(seed)
        ordered = rng.sample(ordered, sample)
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


def _only_guessed_infra(slug: str) -> bool:
    """Whether the extractor's sole objection to *slug* is a guess about names.

    The harvester drops subdomains like www, api and cdn, plus numbered forms
    of them, as infrastructure. As a harvest-time filter that is sound: www4 and
    api2 both redirect to the vendor's own marketing site, so nothing is lost by
    never collecting them.

    Retiring an *existing* candidate on the same rule is a different act. This
    function's caller marks slugs dead without probing, on the contract that
    they cannot correspond to a board whatever the network says -- and a numbered
    label can. mx51.bamboohr.com serves tenant JSON from its own host and had
    been dead-marked since 2026-09-01 while answering. A bare label cannot: www,
    api and cdn are host roles, not tenants, so those stay retired.

    So the guess costs one probe instead of a dead mark, and the answer then
    comes from the network. Restating the pattern here rather than asking the
    extractor is deliberate and narrow: the extractor cannot report *why* it
    refused, and the two uses genuinely want different answers.
    """
    if not _harvest._INFRA_SUBDOMAIN_RE.match(slug):
        return False
    return any(ch.isdigit() for ch in slug)


def partition_unscrapeable(platform: str, slugs: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split slugs into (unscrapeable, worth probing).

    Some identifiers cannot correspond to a board whatever the network says.
    The clearest case is a Workday host label in the tenant slot -- wd1|wd1|
    careers would have to resolve wd1.wd1.myworkdayjobs.com, which does not
    exist -- and upstream ships 6,068 of those, 47% of its Workday list.
    Spending a request to discover that is waste twice over: once here, and
    again in the bot for every scrape cycle until a mark lands.

    The test is the harvester's own: would its extractor produce this
    identifier? Reusing it rather than restating the rules keeps the two from
    drifting, and it is deliberately conservative -- an identifier the
    extractor would merely *normalise* (iCIMS "careers-2u" -> "2u", 3,255 of
    them) is left alone, because ats_service probes both host forms and those
    boards are reachable.
    """
    plat = _harvest.PLATFORM_BY_NAME.get(platform)
    if plat is None:
        return [], list(slugs)
    unscrapeable: list[str] = []
    probe_me: list[str] = []
    for slug in slugs:
        if _harvest._current_identifier(plat, slug) is None and not _only_guessed_infra(slug):
            unscrapeable.append(slug)
        else:
            probe_me.append(slug)
    return unscrapeable, probe_me


PROGRESS_SECONDS = 60.0

#: Give up on a platform that is refusing nearly everything. A 429 is
#: Unreachable, not dead, so no mark is ever wrong because of one -- but the
#: live-rate collapse guard only watches dead verdicts, so a platform answering
#: 429 to every request never trips it and spends its entire budget learning
#: nothing. Workable is quota-metered per window rather than rate-limited, and
#: was observed refusing 40 of 40 probes; an earlier eight-worker pass came back
#: 91% refused and left the address throttled for minutes afterwards, so
#: continuing also makes the next run worse.
#:
#: Deliberately high, and only after enough probes to mean it: a platform that
#: is merely slow or partly flaky must keep going, because those runs still
#: resolve most of what they touch.
UNREACHABLE_STORM_RATE = 0.9
UNREACHABLE_STORM_MIN_PROBES = 20


def validate_platform(platform: str, slugs: Iterable[str], *,
                      probe: Callable[[str], bool] | None = None,
                      workers: int | None = None,
                      budget_seconds: float | None = None,
                      delay: float | None = None,
                      now: Callable[[], float] = time.monotonic,
                      log: Callable[[str], None] = print) -> dict[str, int]:
    """Probe slugs concurrently. Returns counts and the resulting verdicts.

    Bounded by wall clock, not just count. Platforms differ by an order of
    magnitude in how fast they answer -- Workday needs a bodied POST and iCIMS
    two requests per slug -- so a slug budget that suits one starves the rest.
    Whatever is not reached this run is simply picked up next run: the target
    selection prefers never-probed slugs, so progress accumulates.
    """
    slugs = list(slugs)
    if probe is None:
        probe = PROBES[platform]
    if workers is None:
        workers = WORKERS.get(platform, 8)
    if delay is None:
        delay = DELAYS.get(platform, 0.0)
    started = now()

    live: list[str] = []
    dead: list[str] = []
    unknown: list[str] = []
    lock = threading.Lock()

    def one(slug: str) -> None:
        # Pace the request if this platform demands it. Workable answers 429 to
        # anything faster: an eight-worker pass over 6,843 slugs came back 6,242
        # unknown -- 91% refused -- and left the address throttled for minutes
        # afterwards, so even a serial retry was refused. Nothing was corrupted,
        # because 429 is unreachable rather than dead, but the whole pass was
        # wasted and the next one would be too.
        if delay:
            time.sleep(delay)
        try:
            ok = probe(slug)
        except Unreachable:
            with lock:
                unknown.append(slug)
            return
        except Exception:
            # Any unexpected failure is "we could not tell", never "dead":
            # marking on a bug would suppress a real company for a week.
            with lock:
                unknown.append(slug)
            return
        with lock:
            (live if ok else dead).append(slug)

    # Submitted in chunks so the budget can be checked between them. Cancelling
    # in-flight probes is not possible once submitted, so a chunk is the
    # granularity: small enough to stop promptly, large enough to keep every
    # worker busy.
    chunk = max(workers * 4, 1)
    probed = 0
    stopped_early = False
    storm = False
    last_beat = started
    if slugs:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(slugs), chunk):
                # Say something periodically. A platform emitted nothing between
                # its first line and its last, so a paylocity pass that takes
                # forty minutes at two workers was indistinguishable in the log
                # from one that had hung -- and the counts on disk are no help
                # either, since results are written when the platform finishes.
                # Twice that ambiguity had to be resolved by inspecting process
                # tables, and once it was resolved wrongly.
                if now() - last_beat >= PROGRESS_SECONDS:
                    last_beat = now()
                    elapsed = last_beat - started
                    rate = probed / elapsed if elapsed else 0.0
                    log(f"[validate] {platform}: {probed}/{len(slugs)} probed, "
                        f"{len(live)} live, {len(dead)} dead, "
                        f"{rate:.1f}/s after {elapsed:.0f}s")
                if budget_seconds is not None and start and                         now() - started > budget_seconds:
                    stopped_early = True
                    break
                if (probed >= UNREACHABLE_STORM_MIN_PROBES
                        and len(unknown) / probed >= UNREACHABLE_STORM_RATE):
                    stopped_early = True
                    storm = True
                    break
                batch = slugs[start:start + chunk]
                list(pool.map(one, batch))
                probed += len(batch)

    if storm:
        log(f"[validate] {platform}: {len(unknown)} of {probed} probes refused -- "
            f"stopping, {len(slugs) - probed} slug(s) deferred. Nothing is marked "
            f"from a refusal, so no verdict is lost; continuing would only "
            f"deepen the throttling.")
    elif stopped_early:
        log(f"[validate] {platform}: budget spent, {len(slugs) - probed} slug(s) "
            "deferred to the next run")
    log(f"[validate] {platform}: probed {probed} -> "
        f"{len(live)} live, {len(dead)} dead, {len(unknown)} unknown")
    return {"probed": probed, "live": len(live), "dead": len(dead),
            "unknown": len(unknown), "deferred": len(slugs) - probed,
            "_live": live, "_dead": dead}


ENDPOINT_CANARIES = 6
COLLAPSE_RATE = 0.5
#: Below this many probes a collapsed rate is not yet evidence of anything, and
#: the damage a wrong call could do is bounded by the same number.
COLLAPSE_MIN_PROBES = 20


def collapse_suspected(probed: int, live: int) -> bool:
    """Whether this platform's live rate collapsed far enough to be suspicious.

    Split out of the run loop so it can be tested: it is the gate in front of
    the only path that discards a platform's results, and everything behind it
    -- endpoint_healthy, the canaries -- is unreachable when this is wrong.

    A zero-probe run is not a collapse. Reading it as one (0/0 as a 0% rate)
    would send every skipped platform down the outage path and cost a round of
    canary probes each time.
    """
    if probed < COLLAPSE_MIN_PROBES:
        return False
    return (live / probed) < COLLAPSE_RATE


def endpoint_healthy(platform: str, canaries: list[str], log=lambda m: None) -> bool:
    """Ask whether the platform's endpoint is answering, using slugs already
    confirmed live as the control.

    A probe that fails returns "not live", and a failure that hits every request
    -- the endpoint down, DNS dead, the network dropped -- is therefore
    indistinguishable from every company having closed.

    An absolute floor cannot separate the two, because a collapsed rate is
    often real: the residual population, once the recently-confirmed are
    skipped, is mostly boards that are genuinely gone, and workday and bamboohr
    both validate near 0% on it. Slugs recovered from 2022 crawls come in
    around 12%. What distinguishes an outage is that companies *known* to be
    live stop answering, so those are what this asks about -- and on the real
    near-0% runs the controls answered, so the closures were recorded.
    """
    if not canaries:
        return True  # nothing to compare against; assume healthy rather than stall
    probe = PROBES[platform]
    for slug in canaries:
        try:
            if probe(slug):
                return True
        except Exception:  # noqa: BLE001 - a raising probe is a failed probe
            continue
    log(f"[validate] {platform}: {len(canaries)} slugs confirmed live recently "
        "all failed to answer -- treating the endpoint as down")
    return False


def pick_canaries(checked: dict[str, str], dead_map: dict[str, str],
                  count: int = ENDPOINT_CANARIES) -> list[str]:
    """The most recently confirmed-live slugs, newest first."""
    fresh = [(d, s) for s, d in checked.items() if s not in dead_map]
    fresh.sort(reverse=True)
    return [s for _, s in fresh[:count]]


def apply_results(dead_map: dict[str, str], result: dict, today: _dt.date) -> dict[str, int]:
    """Fold verdicts into the dead map. Live slugs are cleared, dead ones dated.

    Unknown slugs are left exactly as they were: a timeout is not evidence.
    """
    added = revived = redated = 0
    for slug in result["_live"]:
        if dead_map.pop(slug, None) is not None:
            revived += 1
    stamp = today.isoformat()
    for slug in result["_dead"]:
        if slug in dead_map:
            redated += 1
        else:
            added += 1
        dead_map[slug] = stamp
    return {"added": added, "revived": revived, "redated": redated}


def purge_expired(dead_map: dict[str, str], today: _dt.date, ttl_days: int) -> int:
    """Drop marks older than the TTL so a long-dead slug is eventually retried."""
    cutoff = (today - _dt.timedelta(days=ttl_days)).isoformat()
    expired = [s for s, when in dead_map.items() if when < cutoff]
    for slug in expired:
        del dead_map[slug]
    return len(expired)


#: A dead mark that comes back live is normal -- companies repost. A *platform*
#: whose dead marks come back live in bulk is a broken probe, because the marks
#: were never earned. Measured baseline across fifteen platforms: 0% for
#: thirteen of them, 20% for icims (two of fourteen, and its two host forms make
#: it the flakiest), 7% for rippling. smartrecruiters sat at 71% -- 150 of its
#: 212 marks -- because its probe read "no postings today" as "no such company".
REVIVAL_ALARM_RATE = 0.40


def revival_rate(platform: str, sample: int, *,
                 probe: Callable[[str], bool] | None = None,
                 workers: int | None = None,
                 seed: int = 0,
                 budget_seconds: float | None = AUDIT_BUDGET_SECONDS,
                 log: Callable[[str], None] = print) -> dict[str, Any]:
    """What share of this platform's dead marks now answer live.

    Read-only, for the same reason audit_live_rate is: an audit that recorded
    its verdicts would repair the very evidence it exists to report, and the
    working pass reaches these slugs on its own anyway.

    This is the shape that catches a probe marking companies dead for a reason
    other than not existing. The postings-count bug on smartrecruiters was
    invisible in every other measure -- the live rate looked plausible, the
    probe answered no to invented slugs and yes to known-live ones, and no
    scraper contradicted a mark. Only re-asking the ones it had already
    condemned showed 71% of them answering.
    """
    dead = sorted(load_dead(platform))
    if not dead:
        return {"sampled": 0, "revived": 0, "rate": 0.0, "dead_total": 0,
                "alarm": False}
    picks = (dead if len(dead) <= sample
             else random.Random(seed).sample(dead, sample))
    result = validate_platform(platform, picks, probe=probe, workers=workers,
                               budget_seconds=budget_seconds, log=lambda _m: None)
    decided = result["live"] + result["dead"]
    rate = (result["live"] / decided) if decided else 0.0
    alarm = decided >= 10 and rate >= REVIVAL_ALARM_RATE
    log(f"[validate] {platform}: {result['live']} of {decided} re-probed dead "
        f"marks answered live ({100 * rate:.0f}%)"
        + ("  <-- the probe is condemning companies that exist" if alarm else ""))
    return {"sampled": result["probed"], "revived": result["live"],
            "rate": round(100 * rate, 1), "dead_total": len(dead),
            "alarm": alarm}


def audit_live_rate(platform: str, sample: int, *,
                     probe: Callable[[str], bool] | None = None,
                     workers: int | None = None,
                     seed: int = 0,
                     budget_seconds: float | None = AUDIT_BUDGET_SECONDS,
                     log: Callable[[str], None] = print) -> dict[str, int]:
    """Estimate a platform's true live rate from a uniform random sample.

    The working pass cannot answer this. It probes never-probed slugs first --
    correct for getting work done, since re-confirming known-dead companies is
    waste -- but that means it systematically samples survivors: everything the
    bot already found dead is excluded. Reading its live rate as the
    population's overstates it badly. Measured on the same data, the working
    pass reported 97.5% while a uniform sample of the same slugs was 38.8%.

    Deliberately read-only. Folding these verdicts into the dead map would let
    an audit reshape the population it is trying to measure, and the working
    pass reaches those slugs on its own soon enough.
    """
    candidates = load_candidates(platform)
    if not candidates:
        return {"sampled": 0, "live": 0, "dead": 0, "unknown": 0, "rate": 0.0}
    picks = (candidates if len(candidates) <= sample
             else random.Random(seed).sample(candidates, sample))
    # Budgeted like the working pass. Without this the audit is the one
    # unbounded network loop in the run: Paylocity probes at two workers and
    # refuses connections under load, so an audit there can outlast everything
    # it was meant to annotate. A short sample is worth less than a stalled job.
    result = validate_platform(platform, picks, probe=probe, workers=workers,
                               budget_seconds=budget_seconds,
                               log=lambda _m: None)
    decided = result["live"] + result["dead"]
    rate = round(100.0 * result["live"] / decided, 1) if decided else 0.0
    short = " (cut short by budget)" if result.get("deferred") else ""
    log(f"[validate] {platform}: audit sample {result['probed']} of "
        f"{len(candidates)} -> {rate}% live (population estimate){short}")
    return {"sampled": result["probed"], "live": result["live"],
            "dead": result["dead"], "unknown": result["unknown"],
            "rate": rate, "population": len(candidates)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--platform", action="append", dest="platforms",
                        choices=sorted(PLATFORMS), help="limit to a platform")
    parser.add_argument("--limit", type=int, default=None,
                        help="max slugs to probe per platform")
    parser.add_argument("--sample", type=int, default=None,
                        help="probe a random sample of this run's targets and "
                             "RECORD the verdicts. Not a measurement tool: "
                             "targets exclude slugs already marked dead, so the "
                             "live rate it reports is of survivors, not of the "
                             "population. Use --audit-sample for that.")
    parser.add_argument("--recheck-dead", action="store_true",
                        help="re-probe every dead mark, ignoring the recheck window")
    parser.add_argument("--ttl-days", type=int, default=90,
                        help="drop dead marks older than this (default 90)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--budget-seconds", type=float, default=1200.0,
                        help="wall-clock ceiling per platform (0 disables)")
    parser.add_argument("--total-budget-seconds", type=float, default=0.0,
                        help="wall-clock ceiling for the whole run; shares the "
                             "remaining time between the platforms still to go, "
                             "so a fast one leaves its slack to a slow one")
    parser.add_argument("--revival-audit", type=int, default=0,
                        metavar="N",
                        help="re-probe N random dead marks per platform, "
                             "read-only, and report how many answer live. A high "
                             "share means the probe is condemning companies that "
                             "exist, which no other measure here reveals.")
    parser.add_argument("--audit-sample", type=int, default=0,
                        help="also probe N slugs drawn uniformly from the whole "
                             "population INCLUDING known-dead ones, read-only, "
                             "to estimate the true population live rate")
    parser.add_argument("--dry-run", action="store_true",
                        help="probe and report, write nothing")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    log: Callable[[str], None] = (
        (lambda m: print(m, file=sys.stderr)) if args.json else print
    )

    # Same exclusion the harvester takes, for the same reason. Two validation
    # runs both read the dead map, probe for minutes, and write back; the later
    # writer loses whatever the earlier recorded in between. save_dead_merged
    # re-reads immediately before writing, which narrows that to the width of a
    # single write, but narrowing a race is not closing it -- and a dry run
    # takes no lock at all, since it writes nothing.
    if args.dry_run:
        lock_ctx = contextlib.nullcontext()
    else:
        try:
            lock_ctx = _harvest.output_lock(DEAD_DIR)
            lock_ctx.__enter__()
        except _harvest.HarvestLocked as exc:
            print(f"[validate] {exc}", file=sys.stderr)
            return 2
    try:
        return _run(args, log, lock_ctx)
    finally:
        if not args.dry_run:
            lock_ctx.__exit__(None, None, None)


def _run(args, log: Callable[[str], None], _lock) -> int:
    today = _dt.date.today()
    platforms = args.platforms or list(PLATFORMS)
    summary: dict[str, dict] = {}
    any_probed = False
    any_targets = False
    t0 = time.monotonic()

    for platform in platforms:
        dead_map = load_dead(platform)
        expired = purge_expired(dead_map, today, args.ttl_days)
        # Before selecting targets: a mark for a slug that is no longer in any
        # company list can never be re-probed, so it would otherwise sit in the
        # file the bot loads until the TTL happened to reach it.
        candidates = load_candidates(platform)
        checked_map = load_checked(platform)
        recon = reconcile_stores(dead_map, checked_map, candidates,
                                 log=log, platform=platform)
        if (recon["dead_orphans"] or recon["checked_orphans"]) and not args.dry_run:
            save_dead(platform, dead_map)
            save_checked(platform, checked_map)
        targets = select_targets(platform, dead_map, recheck_dead=args.recheck_dead,
                                 limit=args.limit, sample=args.sample, today=today)
        total = len(candidates)
        if targets:
            any_targets = True
        if not targets:
            log(f"[validate] {platform}: nothing to probe "
                f"({total} known, {len(dead_map)} dead)")
            summary[platform] = {"probed": 0, "live": 0, "dead": 0, "unknown": 0,
                                 "known": total, "dead_total": len(dead_map),
                                 "expired": expired, **recon}
            if expired and not args.dry_run:
                # Same race as the probing path: re-read so a TTL purge does
                # not roll back marks the bot made while this was running.
                fresh = load_dead(platform)
                purge_expired(fresh, today, args.ttl_days)
                save_dead(platform, fresh)
            continue

        # Retire the structurally impossible without asking the network. These
        # cost a request here and one per scrape cycle in the bot until a mark
        # lands, for an answer their shape already gives.
        unscrapeable, targets = partition_unscrapeable(platform, targets)
        cap = MAX_PROBES.get(platform)
        if cap is not None and len(targets) > cap:
            log(f"[validate] {platform}: metered endpoint, probing {cap} of "
                f"{len(targets)} this run and deferring the rest")
            targets = targets[:cap]
        if unscrapeable:
            log(f"[validate] {platform}: {len(unscrapeable)} unscrapeable by "
                "shape, marked without probing")

        # Share the remaining time between the platforms still to run. A fixed
        # per-platform ceiling wastes whatever a fast platform does not use,
        # and since the confirmed-live store landed most platforms finish in
        # seconds while Paylocity defers thousands of slugs at two workers --
        # so the slack is exactly what the slow one needs.
        budget = args.budget_seconds or None
        if args.total_budget_seconds > 0:
            left = args.total_budget_seconds - (time.monotonic() - t0)
            remaining_platforms = len(platforms) - platforms.index(platform)
            share = max(left, 0.0) / max(remaining_platforms, 1)
            budget = share if budget is None else max(budget, share)

        result = validate_platform(
            platform, targets, workers=args.workers,
            budget_seconds=budget, log=log)
        # Marked, but deliberately kept out of the live/dead counts. The rate
        # those feed is what the CI gate reads to detect an endpoint that broke,
        # and it should measure what the network said. Folding in 6,068 Workday
        # entries that were never asked would drive that platform's rate toward
        # zero on the first run and fail the build for a cleanup.
        # A collapsed live rate is either a real cull or a broken endpoint, and
        # the counts alone cannot tell them apart. Only ask when it collapses:
        # the check costs requests, and a healthy run should not pay for them.
        probed = result["probed"]
        rate = (result["live"] / probed) if probed else 1.0
        if collapse_suspected(probed, result["live"]):
            canaries = pick_canaries(load_checked(platform), dead_map)
            if not endpoint_healthy(platform, canaries, log=log):
                log(f"[validate] {platform}: live rate {rate:.1%} over {probed} "
                    f"probes with the endpoint down -- discarding this "
                    f"platform's results instead of marking them dead")
                summary[platform] = {
                    "probed": probed, "live": result["live"], "dead": 0,
                    "unknown": result["unknown"], "live_rate": round(100 * rate, 1),
                    "known": total, "deferred": result.get("deferred", 0),
                    "dead_total": len(dead_map), "expired": expired,
                    "added": 0, "revived": 0, "endpoint_down": True,
                }
                continue

        result["_dead"] = list(result["_dead"]) + unscrapeable
        result["unscrapeable"] = len(unscrapeable)
        if result["probed"] and result["unknown"] < result["probed"]:
            any_probed = True
        elif unscrapeable:
            any_probed = True
        if args.dry_run:
            # Against a copy, and then reported *from the copy*. Reading the
            # count off the untouched map made a dry run report the dead total
            # from before its own verdicts -- 0 where the real run wrote 1 --
            # so the one number someone previews a run for was the wrong one.
            preview = dict(dead_map)
            changes = apply_results(preview, result, today)
            changes["expired"] = expired
            changes["dead_total"] = len(preview)
            changes["checked_total"] = len(
                apply_checked(load_checked(platform), result, today))
        else:
            # Re-read before writing: the bot marks these same files from its
            # own scrape cycles while this runs.
            changes = save_dead_merged(platform, result, today, args.ttl_days)
            expired = changes["expired"]
            dead_map = load_dead(platform)
        rate = 100.0 * result["live"] / max(1, result["live"] + result["dead"])
        log(f"[validate] {platform}: live rate {rate:.1f}% "
            f"(+{changes['added']} dead, {changes['revived']} revived, "
            f"{expired} expired)")
        summary[platform] = {
            "probed": result["probed"], "live": result["live"],
            "dead": result["dead"], "unknown": result["unknown"],
            "live_rate": round(rate, 1), "known": total,
            "deferred": result.get("deferred", 0),
            "dead_total": len(dead_map), "expired": expired, **recon, **changes,
        }

    if args.audit_sample > 0:
        for platform in platforms:
            summary.setdefault(platform, {})["audit"] = audit_live_rate(
                platform, args.audit_sample, workers=args.workers, log=log)

    # Read-only, so it runs identically under --dry-run. Reported per platform
    # rather than only when it trips: a rate that is climbing is worth seeing
    # before it crosses the threshold.
    if args.revival_audit > 0:
        for platform in platforms:
            summary.setdefault(platform, {})["revival"] = revival_rate(
                platform, args.revival_audit, workers=args.workers, log=log)

    elapsed = time.monotonic() - t0
    log(f"[validate] done in {elapsed:.1f}s")
    if args.json:
        print(json.dumps({"elapsed_seconds": round(elapsed, 1),
                          "dry_run": args.dry_run, "platforms": summary}, indent=2))
    # "Nothing left to check" is a successful run -- every slug is either fresh
    # or inside its recheck window. Only a run that had work and could not do
    # any of it is a failure.
    if any_targets and not any_probed:
        print("[validate] had slugs to probe but every probe was unreachable",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
