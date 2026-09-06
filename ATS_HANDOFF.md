# ATS pipeline — handoff

Paste the "Prompt" section into a fresh session. Everything below it is context
that prompt refers to.

---

## Prompt

> Continue work on the ATS job-board pipeline in this repo. Read `ATS_HANDOFF.md`
> first — it has the current state, the standing constraints, and the open work.
>
> Standing constraints, which override convenience:
> - **Never commit** `src/services/ats_service.py`, `sync_ats_companies.py`,
>   `.gitignore`, `tests/test_dead_slug_ttl.py`, or `src/services/job_service.py`.
>   They carry my uncommitted work. Edit them freely; just stage files explicitly
>   by path and check the staged list before every commit. `git add tests/` once
>   swept 25 unrelated files into a commit.
> - **No Claude co-author trailer** on commits.
> - Work on branch `ats-commoncrawl-harvest`.
> - **ATS probing must not run on GitHub Actions** (decision 2026-08-24, and
>   `tests/test_workflow_wiring.py` enforces it). Collection may.
>
> How I want the work done, based on what actually caught bugs this session:
> - **Verify every claim against live endpoints before acting on it**, including
>   claims from subagents and from me. Three separate "confirmed" findings this
>   session were wrong when tested.
> - A scraper returning `[]` and a blank field are indistinguishable from "no
>   jobs" and "no data". Assume silence means broken until a live call proves
>   otherwise.
> - Mutation-test behavioural changes: break the fix on purpose, confirm a test
>   fails. A mutant surviving means the test is wrong, not that the code is fine.
> - Tests must not touch the network. The suite runs in ~2 minutes; if it jumps,
>   something is making real requests.
>
> Start with the open item at the top of "Open work" — iCIMS is currently
> returning no location for 8,747 companies, which drops them from every
> location-filtered search.

---

## What this is

A pipeline that finds company job boards and scrapes them into a Discord job bot.

| stage | file | runs where |
|---|---|---|
| **Collect** company boards from Common Crawl's bulk index | `scripts/harvest_ats.py` | GitHub Actions, weekly |
| **Validate** each board still exists | `scripts/validate_ats_slugs.py` | locally only — see below |
| **Scrape** jobs from live boards | `src/services/ats_service.py` | the bot |

**Why validation is local.** Runner IPs are Azure ranges that Workday and iCIMS
rate-limit. Validation writes dead marks, the bot reads them, so a blocked run
suppresses live companies and the corruption travels home. Collection reads
static index files and never contacts an ATS, so it cannot produce a dead mark —
that is the whole basis for the split.

A read-only volume test (`scripts/runner_volume_check.py`, plus its workflow)
probed 3,700 known-live boards from a runner and found **3 false-dead, identical
to the local baseline** — so the block does not currently reproduce. That is not
grounds to move validation onto Actions: 3,700 is not 100,000, and the decision
predates the probe hardening. Re-run it before revisiting.

## Numbers

- **~119,400 companies** across 16 platforms in `data/ats_harvest/`
- **~60,000 confirmed live** in `data/ats_checked/`, ~42,000 dead in `data/dead_slugs/`
- Upstream repo for comparison: 60,422 companies / 32,280 live
- Published to the `ats-harvest` orphan branch (harvest + crawl cache only —
  **never** dead or confirmed marks)

## Open work

### 1. iCIMS location — DONE, verified live
iCIMS stopped serving JobPosting JSON-LD on the plain job URL. The same URL with
`in_iframe=1` (the parameter its own portal uses) returns the rendered posting
with the JSON-LD intact. Verified across 12 random boards: 11 returned rows and
**30 of 30 rows carried a location**, plus a description and an ISO date.

One follow-on found in that output: iCIMS writes the literal string
`UNAVAILABLE` into unknown address fields rather than omitting them, so jobs
were displayed as `UNAVAILABLE, UNAVAILABLE, US` and matched a keyword search
for that word. Those placeholders are now dropped.

### 2. BambooHR descriptions — DONE, verified live
`_fetch_bamboohr_detail` is wired through `_enrich_rows`. Sampled 8 boards:
**15 of 15 rows carried a description and a date**, 13 of 15 a location.

### 3. The planner's roadmap (grounded, worth following)
In priority order, each verified against the code:

1. ~~**Level classifier**~~ — **DONE.** The claim that the job side had no
   counterpart was stale: `src/services/job_level.py` already existed, with a
   full classifier and 76 passing tests, and was imported by *nothing* in
   `src/`. Built, then discarded -- the same shape as the fields below.
   Now wired into `scrape_ats_platform` via `_stamp_levels`, centrally rather
   than across 16 scrapers: every row is stamped with `level` (and `level_term`
   where a term is present) whether or not a filter is set, and the new
   `levels=` parameter drops rows outside it.
   `employment_type` is now populated too, because `classify()` reads it and no
   scraper wrote it -- the argument was dead on arrival. Lever's
   `categories.commitment` (verified live: "Full-time" / "Internship"), Ashby's
   `employmentType`, schema.org's `employmentType` in both JSON-LD readers, and
   an optional sixth `shape()` element covering the eight `_board_rows`
   platforms. Live check: Lever 33/35 rows and Ashby 22/22 carry it.
   This is what lets "Software Developer (Winter 2027)" be found at all -- the
   OR-over-title-tokens matcher can never surface it, since none of a student's
   query words appear in the title.
   **The consumer turned out to be scoring, not filtering.** The watcher
   scrapes with empty filters into the archive and filters at query time, so
   passing `levels=` there would permanently discard rows -- wrong place. The
   whole job dict is persisted as JSON, so `level`/`level_term`/
   `employment_type` already reach `jobs.db` for free.
   The real counterpart was `job_match.score_level`, a junior/senior regex pair
   that got two classes wrong, silently and both against a student:
   **"Campus Recruiter" scored 1.0** -- a perfect match for a staff recruiting
   job -- and "Software Developer (Winter 2027)" scored 0.5 neutral when it is
   a student posting. It is now backed by `job_level`, with `conflict` checked
   first so the existing "Senior Intern Program Lead is neutral" contract is
   preserved exactly. `ArchivedJob` carries `level` and `employment_type`
   through `_normalize_record`, so the scrape-time stamp (which had the
   platform's own employment-type field) is trusted rather than re-derived from
   a title; an unrecognised stored value falls back to classifying.
   **Still open:** `levels=` on `scrape_ats_platform` remains caller-less, and
   that is now deliberate -- it is for a targeted query path (the watchlist
   item below), not for the archive scrape.
2. **Company watchlist** — `scrape_ats_platform` already accepts `company_slugs`
   and **no caller passes it**. Add a channel setting plus a short fast loop.
   Cheapest new capability in the repo.
3. **Hot-tier polling** — every board is polled at the same rate regardless of
   whether it posted in two years. Derive posting frequency from `jobs.db`, poll
   the active few thousand hourly. Cuts latency without more requests.
4. **Fields collected and discarded** — Lever `commitment` and Ashby
   `employmentType` are now used (see 1). Still discarded: Lever
   `workplaceType`; schema.org `validThrough` (**application deadline**) and
   `baseSalary`, both already parsed and thrown away.
5. **Yield-drop alarm** — `HealthTracker` is in-memory only, so it resets on
   restart and has no baseline. A scraper returning `[]` records as success.
6. **`.applied` tracking** — nothing records that you applied to anything, so
   `.bestjobs` keeps recommending roles you already sent.

Explicitly *not* worth doing: a web dashboard, auto-apply, sector
classification, time-to-fill analytics (needs a reliable disappearance signal
the scrape cannot currently produce).

### 4. Smaller known issues
- ~~`_matches_location('Toronto, ON', 'Toronto')` returns False~~ — **stale, it
  returns True.** Investigating it surfaced a much larger bug instead, now fixed:
  `_infer_country` resolves *cities and regions* to their country, not only
  country names, and `_matches_location` compared only those codes whenever both
  sides resolved. So every intra-country search was silently country-wide — a
  search for Vancouver returned Toronto jobs, and Boston returned Austin jobs.
  It looked correct in spot checks because a Canadian search mostly saw Canadian
  postings. Country match is now necessary but not sufficient: unless the search
  names a bare country, or the posting is remote/country-wide (nothing to
  contradict), the locality must overlap too. Region names are mapped to the
  codes boards publish ("Ontario" -> ON), case-sensitively, because two-letter
  codes match English words once case is discarded ("Remote in US" -> Indiana).
- A test-isolation leak, fixed: `test_geo_index_loads_exactly_once_under_concurrency`
  monkeypatched `_geo_loaded` and `_geo_cities` but not `_geo_admin1_name`, which
  the loader also assigns. The stub left it empty for the rest of the session, so
  region names stopped resolving and the failure landed in whatever test ran
  next. Worth remembering as a shape: a partially-restored global fails somewhere
  other than where it was caused.
- `_sync_harvest` **overwrites** local harvest files with the branch copy rather
  than merging. The branch is currently a deliberate superset so this is safe
  today, but it is a standing footgun.
- The harvest workflow has a `push` trigger added so it could run from a feature
  branch at all (`workflow_dispatch` only reaches workflows on the default
  branch). **Remove it when this merges to main**, or every push starts an hour
  of collection.
- 3 iCIMS percent-encoding artifacts remain, deliberately — their stripped form
  is not present, so there is no evidence they are residue.

### 5. The outage guard, now tested
`endpoint_healthy` is the only thing between a network failure and mass
dead-marking: a failed probe returns "not live", so an endpoint being down is
indistinguishable from every company on it having closed. It had **no tests at
all**. It does now (`tests/test_validator_outage_guard.py`, hermetic), covering
the guard, canary selection and `apply_results`.

The gate in front of it was inline in `_run` and therefore untestable, so it is
extracted as `collapse_suspected(probed, live)` with the magic 20 named
`COLLAPSE_MIN_PROBES`. Behaviour is unchanged -- verified at every boundary
(19/20 probes, 49/50%).

Two behaviours worth knowing, both now pinned so a change to them is
deliberate rather than accidental:
- **One answering canary is enough.** A real cull leaves the endpoint
  answering; only a dead endpoint silences every control at once. Requiring all
  of them would discard genuine closures.
- **No canaries means "assume healthy".** With no confirmed-live slugs there is
  nothing to compare against, and returning False would stall a platform's
  first run forever. The cost is that a first-ever run during an outage marks
  that platform dead. Every platform currently has a confirmed-live file, which
  is what keeps this safe -- if one is ever deleted, that safety goes with it.

### 6. The harvester's junk filter, now covered
`_looks_like_company` decides what enters `data/ats_harvest/`, and the harvest
is **additive and never removes** -- so a slug it lets through stays forever and
is probed on every validation run. Its test covered 12 cases and none of the
rules added after real junk was found: bare numbers, asset suffixes, locales,
percent-encoding residue, the dotted-TLD rule, `&`/`+`. Now covered in both
directions, each rejection paired with the real company that rule could eat.

**Two of my own tests were not testing anything**, found by mutation:
`advocateslawcareers` is 19 characters, and the vowel rule only applies at 20+,
so the accept case never reached the rule; the hex comparison slug was 30
characters against a 32-character threshold. Both now assert their own length
first. The general lesson: an accept-side example has to reach the rule *and*
sit between the real threshold and a plausible wrong one -- a slug at 45%
vowels passes almost any threshold and so tests almost nothing.

Deliberate, now pinned: bare two-letter codes (`fr`, `en`) are **kept**. Only
hyphenated pairs count as locales, because a bare segment is as likely to be a
short company as a language, and the live probe is the arbiter.

### 7. `--prune --dry-run` used to prune
`--dry-run` is documented as "write nothing", and the `--prune` branch ignored
it outright -- so auditing a prune performed one, on the only operation in the
harvester that removes already-published data. Fixed: `prune_existing` takes
`dry_run`, the CLI threads `args.dry_run` into it, and the log line says
`[dry-run, not written]` so it cannot be mistaken for a prune that happened.
The report is identical either way, which is what makes a dry run reviewable.

### 8. The validator's `--dry-run` reported the wrong numbers
It wrote nothing, correctly -- but its JSON summary disagreed with a real run
in two ways, and that summary is consumed by tooling:
- `dead_total` was read off the map from *before* the run's own verdicts (0
  where the real run wrote 1). "How many will be dead after this" is the
  question a dry run is run to answer.
- `checked_total` was missing from dry runs entirely, so the shape of the JSON
  depended on the flag.

Fixed by reporting from the copy the verdicts were applied to, and by
extracting the confirmed-live update as `apply_checked` so both paths share
one implementation -- two copies would drift, and a dry run that disagrees with
the real run is worse than no dry run, because it is trusted. Dry and real
summaries are now asserted equal.

**This is the third instance of the same bug class in three ticks** (`--prune`
ignoring `--dry-run`, then this). Worth checking any other preview/report path
against the operation it claims to preview.

### 9. `--sample` is not a measurement tool, despite its help text
Two sampling flags with overlapping names and opposite purposes:
- `--sample N` narrows **this run's targets** and its verdicts are **written**.
  Targets exclude slugs marked dead too recently to be stale, so the rate it
  reports is of survivors. Demonstrated: with 80 of 100 slugs already dead,
  `--sample 20` draws **zero** of them.
- `--audit-sample N` draws uniformly from the whole population **including
  known-dead slugs** and writes nothing.

Its help said "probe a random sample (for measuring live rates)", which sent
anyone wanting a measurement to the one flag that both mutates and measures the
wrong population -- the same bias `audit_live_rate` documents at 97.5% against
38.8%. Behaviour is unchanged (both flags do something legitimate); the help
and `select_targets`'s docstring now say which is which, and tests pin the
distinction so the two cannot quietly converge.

Verified read-only as claimed: `audit_live_rate` calls `validate_platform`,
which contains no writes. The harvester's collect `--dry-run` was checked too
and is correct -- its summary is computed before the write guard, so dry and
real agree.

### 10. employment_type now read on six platforms, not two
Live end-to-end check of all 16 platforms first: **every one returns rows**, so
nothing regressed from the `_board_rows` sixth-element change. `level` reads 0
when scrapers are called directly -- it is stamped in `scrape_ats_platform`,
not in the scrapers (verified 10/10 through the public entry).

`employment_type` was populated on lever and ashby only. Four more platforms
publish it in the listing response, so reading it costs no extra request:
`workable.employment_type`, `recruitee.employment_type_code`,
`smartrecruiters.typeOfEmployment.label`, `breezy.type.name`.

**Measured the payoff before wiring, and it is small:** intern-valued postings
were recruitee 2/103, smartrecruiters 3/242, breezy 0/241, workable 0/99 --
roughly 1-2%. Wired anyway because it is free and those are exactly the
postings an early-career search exists to find, but recorded here so nobody
re-runs the investigation expecting more. Only intern values change a level;
`fulltime_permanent`, `Full-time`, `Temps plein` and `Internal Audit` all stay
`mid` (`intern` does not fire on "Internal").

Watch the tuple order: smartrecruiters and breezy pass `""` as the fifth
element so the sixth can carry the type. Getting that wrong silently blanks the
description, which is the field the semantic matcher reads -- there is a test
for exactly that.

### 11. Orphaned marks are now reconciled away
`prune_existing` makes filter fixes retroactive by removing junk from the
harvest, but nothing removed the marks that junk had already collected. Found
**539 orphaned dead marks and 19 orphaned confirmed-live entries**, and the
sample is exactly the junk list: `100`, `ads.txt`, `favicon.png`, `2fwww`,
`en-ca`, `llms.txt`. They sat in the files the bot loads on every scrape until
the 90-day TTL happened to reach them.

`reconcile_stores` now drops marks for slugs no longer in any company list, at
the top of each platform's run, honouring `--dry-run` and reported in the
summary in both modes. Verified against real data: 539/19, zero refusals.

**The guard is the interesting part.** This function deletes what it is given,
and `load_candidates` reads from files -- a missing or truncated harvest file
makes *every* mark look orphaned, which would wipe a platform's entire memory
of what is dead, silently. So it refuses on an empty candidate list, and
refuses when the drop would exceed `RECONCILE_MAX_DROP_RATE` (10%) of a store.
There is also a `RECONCILE_MIN_DROP` floor of 5, because a rate-only guard
leaves small stores permanently un-reconcilable -- rippling holds 35 dead
marks, so four orphans is 11%.

### 12. live_icims could not tell iCIMS's own site from a tenant
`www4.icims.com/sitemap.xml` answers 200 and redirects to `www.icims.com`.
`live_icims` trusted the status alone, so it reported a live board -- the exact
failure `live_bamboohr` already guards against by checking the final URL. Real
tenants keep their hostname on every 200 (verified across several); infra hosts
do not. The check is now there, and the *other* host form is still tried after a
redirect, since a redirect rules out that form rather than the company.

`scripts/check_probe_discrimination.py` cannot catch this class: invented slugs
404 properly, so the probe looked fine. The blind spot is **real hostnames that
are not tenants**.

Related, in `partition_unscrapeable`, which marks slugs dead *without probing*:
it was retiring numbered infra-ish labels on the harvester's shape rule.
`mx51.bamboohr.com` serves tenant JSON from its own host and had been
dead-marked since 2026-09-01 without ever being asked. Bare labels (`www`,
`api`, `cdn`) stay retired -- those are host roles, not tenants -- and the
Workday `wd1|wd1` impossibility still is too. A numbered label now costs one
probe and the network decides.

**A wrong turn worth recording.** The first attempt widened
`_INFRA_SUBDOMAIN_RE` in the harvester so numbered labels were never treated as
infra. That broke an existing test, and checking the four hosts settled it: only
`mx51` is a real board -- `www4.icims.com`, `api2.bamboohr.com` and
`cdn3.bamboohr.com` all redirect to the vendor's marketing site. The harvest-time
filter is correct and was reverted intact; only the condemning path changed. A
live probe returning True is not proof of a tenant if the probe cannot see
redirects.

### 13. All host-based probes now reject an off-host redirect
Following last tick's recorded blind spot -- invented slugs 404 properly, so
`check_probe_discrimination.py` cannot see a probe that mistakes vendor
infrastructure for a tenant. Audited every probe. Five of the seven subdomain
platforms need the rule; bamboohr and jazzhr already had it, icims was fixed
last tick, and **breezy and recruitee were found here**:
- `api.breezy.hr` answers 200 and lands on `developer.breezy.hr` (Breezy's API
  docs).
- `blog.recruitee.com` lands on `recruitee.com/blog`.
- `login.recruitee.com` lands on **`loginsoftware.recruitee.com`** -- a
  *different company's* board, so that tenant's jobs would be filed under the
  slug "login". Note this also rules out substring-matching the vendor domain:
  both hosts end in `.recruitee.com`.

teamtailor 404s all of its own non-tenant hosts, so nothing is wrong there
today; the check is added anyway, since four platforms needed it eventually and
a vendor that starts redirecting would otherwise turn every colliding slug live
silently.

The check applies **only to a 200**. A 403/429/503 is a refusal carrying
whatever final URL the vendor's error page has, and letting the host rule answer
for those turns every refusal into a dead mark -- the false-closure shape. A 404
hides this (dead is right either way), which is why the tests use 403/429/503.

**Checked and found sound:** the vendor-looking segments that are real
candidates are real boards. `greenhouse/help` and `greenhouse/support` return
live listings on `job-boards.greenhouse.io`; `workable/careers` is Workable's own
board; `careers.icims.com` is a genuine iCIMS-hosted portal. A vendor-ish *name*
is not evidence -- only the redirect is.

### 14. The probe checker now tests a third direction
The blind spot recorded two ticks ago is closed in the live tool, not just in
hermetic tests. `scripts/check_probe_discrimination.py` now also probes real
**vendor-owned hostnames** and requires them to read dead. Invented slugs 404
everywhere, so a probe that reads the vendor's own site as a tenant passed both
original checks perfectly.

`VENDOR_HOSTS` is seeded with every case that was a real false positive:
`www4.icims.com`, `api.breezy.hr`, `blog`/`login.recruitee.com`,
`api2`/`cdn3.bamboohr.com`, `www.applytojob.com`. All seven read dead now. A
stuck-true or stuck-false probe is still reported ahead of the vendor result,
since those are broken more fundamentally.

**Scrapers audited too, and they are fine.** Five are host-based without the
check (breezy, recruitee, teamtailor, jazzhr, applicantpro) but **none of them
mark dead**, so there is no destructive path. Tested the worst case live:
`_scrape_recruitee("login")` returns 0 rows, and `applicantpro/careers` returns
real jobs correctly attributed to the slug asked for. Only `_scrape_bamboohr`
marks dead, and it already checks the final URL.

### 15. smartrecruiters no longer reads "no jobs" as "no company"
Its probe read `totalFound` from the postings API, and a zero ended it -- which
conflated an unknown slug with a real company advertising nothing today. The bot
skips dead slugs, so the first job each of those posted would have been missed
until a recheck cleared the mark. Measured: **of six candidates returning zero,
four had a real careers page.**

`careers.smartrecruiters.com/<slug>` settles it, because it answers about the
*company* rather than its postings: a real slug keeps the path, an unknown one
is bounced to the bare `jobs.smartrecruiters.com`. Consulted only when the count
is zero, so the ordinary case still costs one request. Verified: the three real
ones read live, the two genuinely gone stay dead, 10/10 confirmed-live sampled
still live, bogus still dead.

**Two claims re-tested rather than trusted.** `/v1/companies/<slug>` really does
404 without credentials for real companies (five known-live slugs, all 404), so
that route is genuinely unavailable -- the old comment was right. But the same
comment said the "90-day recheck" would pick the company back up; dead marks are
re-probed at `RECHECK_DAYS` = **7**, not 90, so the conflation was less harmful
than documented, and still worth removing.

**A bug in my own fix, caught by its own test.** Matching `f"/{slug}"` as a
substring of the final URL passes for slug `jobs`, because `https://jobs...`
contains `/jobs` via the two slashes in the scheme -- so the redirect that means
"no such company" would have read as success for exactly the slugs most likely
to hit it. It compares `urlparse(final).path` now.

### 16. The smartrecruiters fix, applied -- and a `--revival-audit` to catch the next one
Re-probed every smartrecruiters dead mark with the corrected probe: **150 of 212
answered live** (71%). Applied, so dead fell 212 -> 156 and the confirmed-live
store went 887 -> 1,758. Those 150 companies were being skipped by the bot
entirely.

That failure was invisible to every existing measure -- the live rate looked
plausible, the probe said no to invented slugs and yes to known-live ones, and
no scraper contradicted a mark. The only thing that showed it was **re-asking
the slugs it had already condemned**. So `--revival-audit N` now does that:
read-only, samples N dead marks per platform, reports the share that answer live
and flags anything at or above `REVIVAL_ALARM_RATE` (40%) with at least 10
decided.

Measured baseline, so the threshold is grounded: 0% on thirteen platforms, icims
20-33% (two host forms make it the flakiest), rippling 7%, smartrecruiters 71%.
Small samples cannot alarm -- two of three is 67% and means nothing.

**Three audits run this tick, all clean:**
- No dead-marked slug on any of the other 15 platforms yields jobs from its
  scraper (90 slugs checked).
- No other platform shows an anomalous revival rate.
- breezy's dead count nearly doubling in a day (849 -> 1,603) is legitimate:
  sampled slugs marked dead today 404 on **both** their board endpoint and their
  root, on their own host. `_fetch_board_json` marks dead only on 404/410, which
  is the conservative rule.

### 17. Fleet revival audit, and 2,268 iCIMS companies brought back
Ran `--revival-audit 40` across all 16 platforms. smartrecruiters is now **0%**
(the fix holds). Thirteen platforms sit at 0%, jazzhr 2%, rippling 9%, **icims
21%** -- consistently the highest, and it holds the largest dead store.

Investigated icims rather than assuming flakiness: the revived slugs answer live
**3 times out of 3** each, so they are genuinely live, and all were marked on a
single date (2026-08-27) with prefixed shapes (`careersus-`, `globalcareers-`,
`jobs-`) that an older probe version mishandled. Stale marks left behind by a
since-fixed probe -- not starvation, since `select_targets` does include them;
there simply had been no icims run since they aged in.

So it was run: **2,268 revived**, dead 9,909 -> 7,623, confirmed-live 2,721 ->
5,685. Those companies were being skipped by the bot entirely.

### 18. A refusal storm no longer burns a platform's whole budget
workable decided **0 of 40** in the audit -- every probe returned 429. It is
quota-metered per window, and the quota was spent. No mark is ever wrong from
this (429 is Unreachable, not dead), which is exactly why the live-rate collapse
guard never fires for it: that guard watches *dead* verdicts, and a refused
probe produces none. So the run spent its whole budget learning nothing, and
per the existing note in `validate_platform`, kept the address throttled for
minutes afterwards -- making the next run worse too.

`validate_platform` now stops a platform once at least 20 probes have run and
90% or more were refused, deferring the rest. Verified live: workable stopped
after 20 of 200 instead of grinding through all of them, and greenhouse/lever
ran 60/60 untouched. A half-refused platform keeps going, because those runs
still resolve half of what they touch.

### 19. Process note: a timed-out mutation run left a mutation behind
A mutation script hit the 2-minute tool timeout mid-run, leaving `if storm:`
rewritten as `if False:`. The next run read that as its baseline and "restored"
it, so a dead branch was nearly committed -- and checking one constant was not
enough to notice. **Review the whole `git diff` after any mutation run**, and
restore in a `finally`. That is how it was caught.

### 20. rippling: the probe and the scraper were fighting each other
Verified last tick's work first: icims revival dropped **21% -> 0%**,
smartrecruiters 0%. Both fixes hold.

rippling stayed at 9% (3 of ~32 dead marks, so nearly the whole store), and
those three answered live **3 times out of 3** after being marked dead the day
before. The cause was not flakiness:

| endpoint | the three | known-live |
|---|---|---|
| `ats.rippling.com/<slug>/jobs` (board page, what the probe read) | 200 | 200 |
| `api.rippling.com/.../board/<slug>/jobs` (API, what the scraper reads) | **404** | 200 |

The board page 200s for companies whose API board is gone. So this was a loop,
not a single wrong answer: the validator revived the mark, the scraper's next
pass got a 404 and marked it dead again, and the pair traded the same three
slugs indefinitely, spending requests on both sides every cycle. Each side
looked locally correct, which is why nothing flagged it.

`live_rippling` now reads the API. Verified: the three stay dead, 12/12
known-live still live, invented slugs still dead, and a real board advertising
nothing still reads live -- so this does not reintroduce the "no jobs means no
company" conflation smartrecruiters had. A test pins that both sides use the
same URL path, because either one drifting brings the loop back invisibly.

**Generalisable:** a probe should read the endpoint its scraper reads.
`live_recruitee` already said so in its own comment; rippling was the case where
it had not been applied.

### 21. Probe/scraper alignment is now a test, and the fleet audit is clean
Generalised last tick's rippling finding: audited all 16 platforms for a probe
and scraper talking to different services. **All 16 now align at the host
level** (rippling was the only one, and it is fixed).

Checked path-level divergence on the same host too, where the same loop could
hide:
- **breezy** -- probe `/`, scraper `/json`: **0 mismatches** across 10 live and
  10 dead slugs.
- **teamtailor** -- probe `/jobs`, scraper `/jobs.json`: one slug (`codra`)
  returned 400 on the probe path and 200 on the scraper's. Not a bug: `_decide`
  maps 400 to **Unreachable**, not dead, so no mark can be wrong from it, and
  `codra` probes live now and sits in the confirmed store.

`tests/test_probe_scraper_alignment.py` pins it. Static -- it reads the URLs out
of the two functions' source, so it costs nothing and cannot flake. Host level
only, deliberately: whole-host divergence is the shape that produced the loop,
while the path differences above are intentional and were verified live to
agree.

Verified the guard is not vacuous: reverting `live_rippling` to the board page
fails it, and so does moving `live_recruitee` off the offers API. It also
asserts `_hosts` can distinguish two placeholder hosts, since a collapsing
normaliser would make all 16 assertions trivially pass.

### 22. Two platforms are validated weekly and contribute nothing
Read the archive by source for the first time. Fourteen platforms contribute;
**bamboohr and paylocity contribute zero jobs over a full week**, against 13,755
and 9,325 confirmed-live companies. Different causes, both invisible:
- **bamboohr** is switched off by `ats_bamboohr_enabled`, which defaults to
  `False` in `src/config.py` with no comment saying why, and nothing overrides
  it. The watcher filters it out silently. Deliberate, but a disabled platform
  and a broken one look identical from the archive. **Worth deciding
  explicitly**: it is the largest confirmed-live store in the fleet.
- **paylocity** has a scraper now (added this session) but the running bot
  predates it -- that work is still uncommitted, so the deployed code has no
  paylocity scraper at all.

`scripts/check_platform_yield.py` reports jobs-per-platform against
confirmed-live counts and exits non-zero when a platform with real coverage
yields nothing. A platform below `--min-live` (default 100) cannot alarm --
a handful of companies having no openings this week proves nothing.

Also visible in that data: only **442 of 1,990** archived jobs carry a
description, the field `SEMANTIC_MATCH_TARGET` reads. The scraper-side
description work from earlier ticks is uncommitted and undeployed, so that
number should rise once it lands.

**A bug in my own code, caught by its own test.** `confirmed_live` took
`checked_dir: Path = CHECKED_DIR`, which binds at import -- so it read the real
store no matter what a caller passed, and the report was unfakeable in tests
while looking correct. Resolved at call time now.

### 23. Description coverage measured per platform -- it splits cleanly in two
`SEMANTIC_MATCH_TARGET` is `"description"`, so a platform supplying none is
invisible to matching rather than merely sparse. Measured over one week:

| coverage | platforms |
|---|---|
| **100%** | teamtailor, recruitee, rippling, jazzhr, workable, jobvite |
| 28-52% | applicantpro, breezy |
| **1-6%** | workday 2%, greenhouse 3%, ashby 4%, icims 2%, lever 1%, smartrecruiters 6% |

That bottom row is **1,624 of 2,184 archived ATS jobs -- 74%** -- and it is
exactly the set the uncommitted `ats_service` work adds descriptions to
(greenhouse `content=true`, lever `descriptionPlain`, the Ashby posting API,
Workday `jobDescription`, iCIMS JSON-LD, smartrecruiters detail enrichment). So
this is not a new bug; it is the size of the undeployed one, and the number to
watch once that work lands.

`check_platform_yield.py` now reports a `desc` column and lists thin platforms.
**Advisory, never fatal** -- a thin description is a quality problem while a
silent platform is a broken one, and collapsing the two would make the exit
status useless for gating. A silent platform is not also listed as thin: 0% of
nothing is not a description problem, and listing it twice buries the real
finding.

### 24. Location and date coverage measured -- two more undeployed fixes, sized
Extended the field audit to the other two fields the pipeline acts on. Two
platforms stand out, and both are already fixed in the uncommitted
`ats_service`:

- **iCIMS location: 2%** (173 of 175 jobs blank). `_matches_location` returns
  False for an empty location, so those jobs are **dropped from every
  location-scoped search**. This is the bug the `in_iframe=1` fix addressed on
  the very first tick -- the deployed bot does not have it.
- **Ashby date_posted: 4%** (284 of 296 blank). `_posting_age_ok` is explicit:
  *"Rows with no date_posted (bamboohr, ashby) always pass."* So these are
  **exempt** from "newer than N hours" rather than dropped -- a years-old
  posting reads as fresh. The Ashby posting API supplies `publishedDate`; the
  old GraphQL query never asked for it.

Everything else is at 99-100% on both fields.

**The two failures run in opposite directions**, which is why the report keeps
them separate and states each consequence: a thin *location* means jobs are
missing, a thin *date* means stale jobs are being shown. Reading one as the
other sends the next person looking for the wrong symptom entirely.

`check_platform_yield.py` now reports `desc`/`loc`/`date` columns from a single
pass over the archive, so the totals and per-field counts cannot disagree.

**Standing summary of the undeployed gap** (all in uncommitted `ats_service`):
descriptions on the six largest platforms (74% of archived jobs), iCIMS
location, Ashby dates, and the paylocity scraper. These are now all measurable
-- re-run the script after that work lands and the numbers should move.

### 25. A refusing platform is no longer reported as a broken probe
Ran the discrimination check as a regression pass over the six probes changed
this session (icims, breezy, recruitee, teamtailor, smartrecruiters, rippling).
**All 15 evaluable probes are healthy** in all three directions.

workable came back `NO-DATA` -- every probe refused, its per-window quota still
spent -- and the tool failed the run with *"Do not run a validation pass until
this is fixed"*, at someone with nothing to fix. That is a false alarm, and a
check that cries wolf gets ignored, which is worse than not having it. An outage
is never a verdict anywhere else in this pipeline; it is not one here now
either. Reported as "Could not evaluate", exit 0, with
`--fail-on-unevaluated` for strict gating. The summary counts only what was
actually evaluated -- "all 16 probes discriminate" overstates a run where one
was never asked.

### 26. Archive identity is clean -- two audits, no action
- **2,215 ATS rows, 2,215 distinct URLs, zero URLs carrying more than one
  title.** No mis-paired titles.
- **Zero jobs appearing from more than one source**, across 2,085 distinct
  (title, company) pairs.

Worth recording the near-miss: a sample printed with `url[:56]` made three
teamtailor URLs look identical, which read as a scraper mis-pairing titles. The
full URLs differ (`8318074`, `8318081`, `8318085`) -- **the truncation was the
bug, not the data.** Print full values before believing a duplicate.

### 27. 19% of the fleet has never been validated -- now measured
The bot skips dead slugs but scrapes everything else, so a company never
resolved either way costs a request on **every scrape cycle** until a validation
pass reaches it. Measured: **24,829 of 130,468 candidates (19%)**, concentrated
in platforms that simply had not been run.

| platform | unvalidated |
|---|---|
| workable | 8,815 (90%) -- its quota is spent, so it cannot be cleared right now |
| paylocity | 4,466 (28%) |
| recruitee | 3,183 (73%) |
| applicantpro | 2,511 (65%) |
| jazzhr | 2,233 (45%) |
| teamtailor | 1,601 (36%) |

greenhouse, ashby, icims, lever and smartrecruiters are at zero -- those have
had full passes.

`check_platform_yield.py` gains a `todo` column and a backlog summary.
**Reported, never fatal**: this is a backlog, and the fix is a run rather than a
code change. A dead mark counts as validated, since it is an answer -- counting
it as outstanding would make the backlog permanently unclearable.

Started clearing it: a pass over recruitee, jobvite, rippling and breezy is
running. First result, recruitee: 700 probed, 236 live, **107 newly dead** --
107 slugs the bot will stop paying for on every cycle.

### 28. Collection is saturated
The crawl cache holds all **124** indexes and matches `collinfo.json` exactly --
there is no unswept Common Crawl index. New companies have to come from new
platforms, not new crawls.

### 29. recruitee was paced with workable's numbers, and refusing a third of every run
Last tick's backlog pass retired **535 slugs** (jobvite 402, recruitee 107,
breezy 14, rippling 12) -- each one a request the bot stops paying on every
cycle. But recruitee came back **357 of 700 unknown (51%)** while every other
platform was near zero, and 51% never trips the 90% storm guard.

Reproduced at 28%, all 429s. The cause: `WORKERS` gave recruitee **workable's**
measured numbers (4 workers at 0.25s) without a curve of its own. Its actual
curve, 429s per 60 probes from cold: **17 at 8.6/s, 14 at 5.3/s, 0 at 2.4/s**.
Now 2 workers at 0.5s -- 2.5/s, and **0 refusals across 100 probes**.

**Order mattered, and checking it changed the conclusion.** The first sweep ran
fastest-first, so the slow settings could have been benefiting from the throttle
window resetting rather than from the pacing. Re-running slowest-first showed
the same 3-worker/1.0s setting refusing **0 of 60 from cold but 9 of 60 straight
after a fast burst** -- recruitee stays throttled once pushed, so one impatient
run spoils the next. That is the same behaviour the Workable note already
describes, and it makes the pacing matter more than the steady-state numbers
suggest.

**A latent defect found on the way:** `WORKERS` listed `"recruitee"` **twice**,
as 8 and then as 4. Python keeps the last, so it worked -- but the first entry
was dead, and editing it would have changed nothing while looking like it had. A
static test now rejects any duplicated key in any dict literal in the module.

### 30. Two consistency findings that need NO action
- **105 slugs appear in both the dead and confirmed-live stores.** In all 105
  the dead mark is the newer one: the bot marks dead from its own scrape cycles
  and never touches the confirmed-live store. It is benign and self-healing --
  `select_targets` builds `fresh` from slugs *not* in the dead map, so a stale
  confirmed-live entry cannot delay a re-probe; the 7-day dead recheck governs,
  and `apply_checked` pops the slug on the next verdict. Left alone
  deliberately.
- The reconciliation above does not touch these, since they are still real
  candidates.

### 13. Audits run 2026-09-04, both clean -- no need to repeat soon
- **Junk in the stored harvest: none.** Re-applying the current rules to all
  119,398 stored slugs rejects zero. Note the trap: `_looks_like_company` is
  the wrong test for Workday, whose entries are `tenant|host|site` triples and
  which applies the filter to the *tenant only*. Checking naively reports all
  8,799 Workday rows as junk; acting on that would have deleted the platform.
- **The prune round-trip is exact.** Every one of the 119,398 stored slugs
  returns itself from `_current_identifier`, so a prune today would drop and
  rewrite nothing. That is the check worth re-running after any extractor
  change, since prune is what makes filter fixes retroactive -- and destructive.

### 9. A guard test worth knowing about
`test_ci_preflight_names_every_untracked_module_the_container_needs` walks
imports from the real entry points and fails when a module reachable from them
is absent from a fresh clone. Importing `job_level` into `ats_service` tripped
it immediately -- a container build would have died on ImportError. Fixed by
naming `src/services/job_level.py` in the preflight list in
`.github/workflows/docker-image.yml`. Expect this test to fire whenever a new
untracked module becomes reachable; that is the point of it.

## Bugs fixed this session — the classes worth remembering

**Probes that cannot say "no".** Five probes returned `True` for invented
company names. jazzhr reported *100.0% live over 2,683 slugs with zero dead* —
which is what a healthy platform and a broken probe both look like. The tell was
the zero, not the rate. Real rates after fixing: 60.9%–89.3%.

**False closures from partial answers.** lever and iCIMS each try two hosts; one
refused plus one 404 fell through to "dead". Refusals cluster under load, so the
false closures arrived in bulk exactly when they were most damaging.

**Dead-marking inverted on two platforms.** BambooHR marked companies dead on
any non-JSON 200 — a Cloudflare page, captcha or maintenance page suppressed a
real company for the full TTL. Ashby could never mark anything dead (its GraphQL
never 404s) *and* actively cleared existing marks by marking alive on the bare 200.

**Filtering before the field was populated.** Location was filtered before
enrichment in jazzhr, breezy and applicantpro. An empty location matches no
search, so real jobs vanished and it looked like "no jobs matched".

**Dates that exempt rather than drop.** `_posting_age_ok` returns `True` when
parsing raises, so an unparseable date does not filter the row out — it exempts
the row from the age filter entirely. Recruitee, ApplicantPro and Workday were
all doing this.

**A field the matcher needs that nothing supplied.** `SEMANTIC_MATCH_TARGET` is
`"description"`, and none of the six original scrapers emitted one — the largest
platforms were invisible to semantic matching rather than merely sparse.

**Ashby via GraphQL was a dead end.** Asking its board query for
`descriptionPlain`/`publishedDate` makes the *whole query* error and return zero
jobs for ~5,000 companies, looking exactly like "no openings". It now uses the
public posting API, which carries description, date, real URL, `isRemote` and
compensation — and 404s properly, which is what makes dead-marking work.

## Verification commands

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

~2,100 tests. Runtime varies a lot (2-4 min) because a handful of
pre-existing tests compile LaTeX or exercise retry backoff -- one
resume-service test alone takes 40s. Check `--durations` before assuming a
slow run means something started hitting the network.

Live end-to-end check across every platform (this is what caught most of the
bugs above — it needs `data/ats_checked/` to be populated):

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=src python -c "import sys,json;sys.path.insert(0,'src');from services import ats_service as a;[print(p, len(a._SCRAPERS[p](sorted(json.load(open(f'data/ats_checked/{p}.json')))[0],'','',2))) for p in a.ATS_PLATFORMS]"
```

The probe self-check, which is how the broken probes were found. It was a
shell one-liner run from memory; it is now a script, and it checks **both**
directions:

```bash
PYTHONPATH=src python scripts/check_probe_discrimination.py
```

Non-zero exit means at least one probe is stuck. Add `--platform NAME` to check
one, `--sample N` for more known-live slugs, `--seed N` to reproduce a run.

The one-liner only ever probed invented slugs, so it caught a probe that could
not say **no** (reports everything live, marks nothing dead) and was blind to
one that could not say **yes** -- which is the worse failure, because it marks
every company on the platform dead for the full TTL and has no tell at all.
Both are checked now. `Unreachable` and raised exceptions count as neither
answer, so an outage does not read as a broken probe, and a probe that produced
no usable answer reports `no-data` rather than passing by default.

Verified 2026-09-04: all 16 probes discriminate in both directions.
