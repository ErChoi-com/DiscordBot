# Handoff — ATS company-site discovery

Repo: `C:\Users\ernes\.vscode\discordbot\rebuilt_app`, branch `ats-commoncrawl-harvest`.
Python 3.14, venv at `.venv/Scripts/python.exe`. Remote is named `DiscordBot`, not `origin`.

> **This file is the current handoff — start here.** Two older ones sit beside it and are
> reference, not instructions:
> - `ATS_HANDOFF.md` (2026-09-04) — the deep ATS pipeline reference: platform-by-platform
>   endpoint notes, probe/scraper alignment, the standing constraints. Consult it for
>   detail on how a specific platform works. Its "Prompt" section is superseded by this
>   file; do not follow both.
> - `HANDOFF.md` (2026-08-30) — a finished audit of an earlier session. Historical.
>
> `docs/USE_CASES.md` is referenced throughout and **may not exist yet** — the guardian
> subagent in §7 builds it. If it is missing, that is the first thing the guardian does.

---

# 1. The goal

**Find ATS company sites, scrape them, and build the list.** Growing and improving the
company/tenant discovery layer is the explicit focus. Everything else — scraper polish,
health reporting, test hygiene — is secondary and should only be touched when it blocks
this or when it is actively wrong.

Two standing requirements on *how* the work is done:

**Build broad, not specific.** A change that helps one platform is worth much less than a
change that helps a whole class of them, and far less than one that makes the next platform
free. Before writing a per-platform special case, ask what the general version is and
whether the general version is actually harder. Prefer an abstraction that makes
`platforms × sources` combinations work over N one-off additions. Per-platform code is
justified only where a platform genuinely differs in kind (Workday's `tenant|host|site`
triple is a real example; most are not).

**Plan extensively before editing.** State the problem, the options considered, why the
chosen one is general, what could break, and what will be measured to prove it worked —
before touching code. This session repeatedly found that the first plausible fix was the
wrong shape (see §7).

**Run a use-case guardian subagent, and do not break anything.** Before any other work,
spawn a subagent whose standing job is to know and maintain every use case this bot has,
keep `docs/USE_CASES.md` current, and answer "would this change break something a user
relies on?" Talk to it before cross-cutting changes and tell it after landing them. The
full charter and the exact spawn prompt are in **§7 — do that first**. The bot has a lot of
surface area (Discord commands, watchers, resume pipeline, archive, ATS scrapers) and the
discovery work in §3 touches modules five other scripts import; nothing in this plan is
worth regressing a command someone uses.

---

# 2. The core structural problem

`scripts/harvest_ats.py` models a platform as:

```python
class Platform:
    name: str
    queries: tuple[tuple[str, str], ...]   # (url pattern, CDX matchType)
    extract: Callable[[str], str | None]   # URL -> company slug
```

`harvest_platform()` runs those queries through `cdx_request()`. **Every platform's
discovery therefore comes from exactly one kind of source: a CDX index** (Common Crawl
bulk, or the Wayback Machine — both CDX-shaped). Adding a platform is one `Platform(...)`
entry. Adding a *source* means rewriting the harvester.

That single-source coupling is the ceiling on the whole fleet. Discovery is currently
limited to "whichever ATS URLs a web crawler happened to capture", which is why 23,496
slugs sit unvalidated and why some platforms are far better covered than others for
reasons that have nothing to do with how many customers they have.

`Platform.extract` is already the right abstraction and is source-agnostic: it maps a URL
to a slug and does not care where the URL came from. The work is to give it more URLs.

## Fleet shape, measured 2026-09-05

| platform | shape | CDX queries | wayback | slugs |
|---|---|---|---|---|
| applicantpro | subdomain | 1 | 0 | 3,843 |
| ashby | path-segment | 1 | 1 | 5,276 |
| bamboohr | subdomain | 1 | 1 | 21,229 |
| breezy | subdomain | 1 | 0 | 5,482 |
| greenhouse | path-segment | 3 | 3 | 13,490 |
| icims | subdomain | 1 | 0 | 8,719 |
| jazzhr | subdomain | 1 | 0 | 4,916 |
| jobvite | path-segment | 1 | 0 | 1,501 |
| lever | path-segment | 2 | 2 | 7,686 |
| oracle | whole-host | 2 | 0 | 1,418 |
| paylocity | custom | 1 | 1 | 15,908 |
| personio | subdomain | 2 | 0 | 736 |
| recruitee | subdomain | 1 | 0 | 4,386 |
| rippling | path-segment | 1 | 0 | 2,008 |
| smartrecruiters | path-segment | 1 | 0 | 1,914 |
| teamtailor | subdomain | 1 | 0 | 4,400 |
| workable | path-segment | 1 | 0 | 9,841 |
| workday | custom | 2 | 2 | 8,799 |
| **total** | | | | **121,552** |

**8 platforms are subdomain-shaped and 1 is whole-host** — 9 of 18. That shape is what
makes the first source below general rather than a special case.

---

# 3. The plan

## Phase 0 — measure before building (do this first, it changes the ordering)

None of the phases below should start until these numbers exist. Each is cheap and each
could invalidate a phase.

1. **Archive yield.** Extract every ATS URL already in `data/jba/jobs/*` and
   `jobs.db`, run each through the matching `Platform.extract`, and count slugs **not**
   already in `data/ats_harvest/<platform>.json`. This costs zero network requests. If it
   is large, Phase 1 is the cheapest win available and should go first.
2. **CT log coverage.** For two subdomain platforms (suggest `personio`, 736 slugs, and
   `teamtailor`, 4,400), query `crt.sh` for `%.<domain>` and count distinct subdomains
   against the current fleet. This tells you whether Phase 2's estimate is real.
3. **Live rate of newly-found slugs.** Whatever a new source finds, probe a random 100
   with `validate_ats_slugs`. A source that yields 10,000 slugs at 11% live is worse than
   one yielding 1,000 at 87% — the repo has already measured that dead slugs cost a request
   per cycle forever (see `d2e68f5`'s reasoning and the crawl-age table in
   `.github/workflows/ats-harvest.yml`). **Yield without a live rate is not a result.**

## Phase 1 — the discovery-source abstraction

The one change that makes everything after it cheap.

```python
class DiscoverySource(Protocol):
    name: str
    def applies_to(self, platform: Platform) -> bool: ...
    def iter_urls(self, platform: Platform) -> Iterator[str]: ...
```

`harvest_platform` becomes: for each applicable source, for each URL it yields, run
`platform.extract`. Existing CDX behaviour becomes `CdxSource` and must produce byte-identical
results — prove that before adding anything else.

Requirements that make this general rather than decorative:
- A new platform gets every applicable source with no source-side change.
- A new source works for every platform whose shape it fits, with no platform-side change.
- Per-source attribution in the manifest and in `--json` output, so the value of a source
  is measurable and a bad one can be switched off on evidence.
- Sources must be independently failable: one source timing out must not lose the others'
  results. The harvester already treats a failed platform this way; match it.
- `--source` / `--no-source` flags, mirroring the existing `--platform` and `--index`.

## Phase 2 — Certificate Transparency source (broad: 9 of 18 platforms)

Every TLS certificate issued is logged publicly. A board at `<slug>.personio.de` needed a
certificate, so it is in the logs **whether or not any crawler ever visited it**. This is
categorically better than crawl data for subdomain platforms: it is complete by
construction rather than by luck.

- Applies to: any platform whose extractor is `_subdomain_extractor` or `_host_extractor`
  — currently applicantpro, bamboohr, breezy, icims, jazzhr, personio, recruitee,
  teamtailor, oracle. Declared by shape, **not** by a hand-maintained platform list.
- Source: `crt.sh` (`?q=%.<domain>&output=json`), with the existing politeness and retry
  discipline. Expect large responses; page and stream.
- Reuse `_looks_like_company`, `_is_locale` and the junk filters already in the harvester —
  CT logs contain plenty of `www`, `staging`, `test`, and wildcard entries.
- Wildcards (`*.personio.de`) yield no slug and must be dropped, not normalised into one.

## Phase 3 — archive-feedback source (broad: all 18, zero network)

The bot already collects job links. Those links contain ATS URLs, some for boards not in
the fleet. Feeding them back through the *same* `Platform.extract` costs nothing and works
for every platform without a line of per-platform code — the cleanest demonstration that
the Phase 1 abstraction is real.

It also satisfies the governing rule (§4) by construction: the fleet grows as a side effect
of normal operation rather than by anyone running anything.

Wire it into the daily rollover branch beside the existing four hooks.

## Phase 4 — platform-native directories (the path-segment platforms)

CT logs do not help ashby/greenhouse/jobvite/lever/rippling/smartrecruiters/workable —
they share one host. For these the general source is the vendor's own customer directory
or public board-search endpoint, where one exists.

- **Investigate before implementing.** For each, find whether a public directory or search
  API exists, and record the finding either way — a written "no directory exists" is a
  result worth keeping so nobody re-investigates.
- Keep it one `DirectorySource` parameterised per platform, not seven bespoke scripts.
- Respect robots.txt and the measured worker ceilings.

## Phase 5 — new ATS families (coverage, not depth)

Dayforce (403, needs `browser_service`), UKG, SuccessFactors, ADP, Cornerstone, Bullhorn,
Avature. This is the largest *coverage* gap — missing employers entirely, not missing jobs
at known employers. Oracle/Taleo and Personio landed 2026-09-05 and are the template:
harvester entry, validator probe, scraper, workflow lists, worker ceiling, all four
registries in sync (`test_workflow_wiring.py` pins this).

Do this after Phases 1-3 so each new platform inherits every source instead of only CDX.

---

# 4. Governing rule (carried forward, non-negotiable)

**Nothing may depend on a one-time manual action.** Bootstrapping once into an automated
loop is fine; needing to be re-run by hand forever is a defect. Four instances were found
and fixed on 2026-09-05:

- `geo_priority.refresh` — the fleet-ordering index, never called by anything.
- `validate_ats_slugs` — the confirmed-live and dead stores existed only because someone
  typed the command.
- The confirmed-live/dead contradiction `ats_service` could write and nothing repaired.
- `check_platform_yield` — the guard against a silent platform, run by nobody.

Any new discovery source must land inside a loop that runs on its own, not as a script
someone remembers.

---

# 5. Hard constraints — read before touching anything

- **NEVER `git add`** these; they carry local-only work: `src/services/ats_service.py`,
  `src/services/job_service.py`, `sync_ats_companies.py`, `.gitignore`,
  `tests/test_dead_slug_ttl.py`. Edit freely; stage explicitly by path and run
  `git diff --cached --name-only` before every commit.
- **No Claude co-author trailer** on commits in this repo.
- **Never `git stash`** — ~38k uncommitted lines.
- **Do not push** (to `ats-harvest` or anywhere), **do not restart the bot** (pid 24480),
  and **never raise `PLATFORM_WORKERS`** — they are measured politeness ceilings, scale
  down only. Ask first.
- Commit style: `area: lowercase description of the behaviour change`. Body says what was
  wrong and what it cost, not what was added.
- Scrapers hit real employers' servers. Politeness ceilings, backoff and the refusal
  breaker are correctness, not decoration.

---

# 6. Verification discipline

- **Mutation-test new code.** Every mutant killed, the test strengthened, or the redundancy
  removed. **Verify a surviving mutant is a real mutation, not an equivalent one** — three
  "escapes" this session were malformed mutants that changed nothing.
- Full suite: `.venv/Scripts/python.exe -m pytest -q` — 2,895 tests, 4-9 minutes.
- **Clean-checkout verify in a detached worktree before committing any test**, and diff
  failures against the HEAD baseline rather than reading them raw: a clean checkout already
  has **8 failures + 1 collection error** in `test_job_match.py`, because callers of several
  things live in the never-commit files. Only *new* failures matter.
  `git worktree add --detach <tmp> HEAD`, copy only the files you intend to commit, run
  `pytest -q -p no:randomly --continue-on-collection-errors`, `comm -13` against baseline.
- Tests must not touch the network.
- **Heredocs in Bash mangle backslashes** (`\\n` becomes a real newline). Use Write/Edit for
  anything containing escapes. This cost three separate rounds on 2026-09-05.

---

# 7. The use-case guardian subagent — SPAWN THIS FIRST

**Do this before Phase 0, before the in-flight cleanup in §8, before anything.**

This bot has far more surface area than the ATS layer being worked on: Discord commands and
their aliases, the job and Reddit watchers, four daily rollover hooks, the resume and cover
letter pipeline, `.bestjobs` matching, the archive and its dedup index, quota, health.
A discovery refactor touches `harvest_ats.py`, which `validate_ats_slugs.py`,
`check_platform_yield.py`, `runner_probe_check.py`, `runner_volume_check.py` and
`check_probe_discrimination.py` all import. Nothing in §3 is worth regressing a command
someone uses, and the person who would notice is not reading the test output.

So: keep a subagent alive whose whole job is knowing what this bot does for its users.

## The protocol

- **Spawn it at the start of the session**, in the background, with the charter below. Let
  it build or refresh `docs/USE_CASES.md` while you do Phase 0's measurements.
- **Ask before you restructure.** "Does anything depend on X?" / "What breaks if I change
  the signature of Y?" / "Which tests cover the resume pipeline?" Ask it *before* the edit,
  not after the suite goes red.
- **Tell it after you land anything** that changes behaviour, adds a command, changes a
  data store's shape, or moves a module, so the inventory stays true rather than becoming
  another stale document.
- **It is read-only on product code.** It owns `docs/USE_CASES.md` and nothing else. It
  must not edit `src/`, `scripts/` or `tests/`, must not run git state-changing commands,
  must not restart the bot, and must not run scrapers or validators (they hit real
  employers).
- If it finishes and you still have work to do, message it again rather than spawning a
  second one — it keeps its context.

## Spawn prompt (reuse verbatim, adjust only the task at the end)

> You are the **use-case guardian** for a Discord job bot at
> `C:\Users\ernes\.vscode\discordbot\rebuilt_app` (branch `ats-commoncrawl-harvest`,
> Python 3.14, venv at `.venv/Scripts/python.exe`).
>
> Your standing job for this session: know every user-facing capability this bot has, keep
> that knowledge current, and be the thing that answers "would this change break something
> a user relies on?" The main session will ask you that repeatedly while it works on the
> ATS company-site discovery layer.
>
> **This task:** produce and maintain `docs/USE_CASES.md` — a complete, evidence-based
> inventory covering:
> 1. Every Discord command — `src/commands/handlers.py` defines them as `CMD_* = ".name"`
>    with `@command_handler(...)` decorators and a `COMMAND_ALIASES` map. Give the command,
>    aliases and slash forms, who may run it, what it does, what it reads and writes.
> 2. Every background loop — `src/watchers/manager.py`: the job watcher, the Reddit
>    watcher, the ATS scrape loop, and the daily rollover hooks inside
>    `_run_ats_scrape_loop` (`_refresh_company_lists`, `_validate_company_slugs`,
>    `_refresh_geo_index`, `_check_platform_yield`). Say what each does, how often, and
>    what a user would lose if it silently stopped.
> 3. The pipelines behind the commands: resumes (`src/services/resumes/`), job matching
>    (`src/services/job_match.py`), archive and dedup (`src/services/jba/`), the ATS
>    scrapers (`src/services/ats_service.py`), quota, health.
> 4. The data stores and what depends on them: `data/ats_harvest`, `data/ats_companies`,
>    `data/ats_checked`, `data/dead_slugs`, `data/jba/jobs`, `.bot_state.json` — who writes
>    each and who reads it.
> 5. UI surfaces in `src/ui/views.py`, including any no command can currently reach.
>
> For each use case: what the user does, what they get, which modules implement it, and
> **which tests cover it**. Flag any use case with no test coverage — those are where a
> refactor does damage silently.
>
> **Method:** read the code, never guess; every claim needs a file and line. Do not run the
> full suite (4-9 minutes) — run individual test files if you need to confirm something.
> Prefer Grep/Glob/Read over shell; heredocs on this machine mangle backslashes.
>
> **Hard rules:** read-only on product code — the only file you create or modify is
> `docs/USE_CASES.md`. No edits under `src/`, `scripts/`, `tests/`. No git commands that
> change state (reading is fine). Do not restart the bot (pid 24480). Do not run any
> scraper or validator script — several make live requests to real employers.
>
> **Context:** `src/services/ats_service.py`, `src/services/job_service.py`,
> `sync_ats_companies.py`, `.gitignore` and `tests/test_dead_slug_ttl.py` carry large
> uncommitted local changes and are never committed — read the working-tree versions, that
> is what runs. A clean checkout of HEAD fails 8 tests plus a collection error in
> `test_job_match.py` for that reason; expected, not a bug to report. 18 ATS platforms are
> scraped; `workable` answers 429 by design.
>
> Reply with: how many distinct use cases you catalogued, which have **no test coverage**,
> and any capability no user can currently reach. Under 20 lines — the document is the
> deliverable.

## Findings from the 2026-09-05 run (full detail in `docs/USE_CASES.md`, 413 lines)

**47 use cases catalogued** — 18 commands, 6 background loops, 7 pipelines, 13 data stores,
10 UI surfaces, 1 MCP surface.

**29 slash commands are declared but Discord never offers them.** `src/app.py` registers 15
with `client.tree`; `src/commands/handlers.py` declares 44 slash aliases. The 29-way gap
includes `/health`, `/quota`, `/reset`, `/jobtest`, `/best`, `/more`, `/watch`, `/hi`,
`/resume`, `/cover`. They are not broken — they simply do not exist as far as Discord is
concerned, so only the dot form works. Verified by comparing
`tree.command(name=...)` in `app.py` against the `"/name"` literals in `handlers.py`.
This is the largest user-facing gap the inventory found and it is nothing to do with the
ATS layer; treat it as its own piece of work, not a drive-by fix.

**Two capabilities no user can reach:**
- **Channel modes** — `ModeDropdownView` (`views.py:255-281`), `store.set_mode`,
  `format_mode_summary` are all intact but no command constructs the view, so
  `channel_modes` in `.bot_state.json` can only ever hold the default. Marked
  `# orphan-ok:` because the missing entry point is the defect, not the renderer.
- **`JobRoleFilterDropdown`** (`views.py:391-411`) — never attached; `JobSettingsView`'s
  five action rows are full, so `role_filters` is reachable only as free text in
  `JobTextModal`. `tests/test_job_region_control.py:14` asserts this deliberately.

**16 use cases have no test coverage**, ranked by blast radius. The top few matter for any
refactor that crosses these paths:
1. `.rclear`/`.rreset` — destructive overwrite of a channel's reddit seen file
2. `.scrape` — network egress plus a channel process lock; a leaked lock blocks that
   channel's job watcher
3. `.jobtest` — 175 lines re-implementing all three dedup layers, so drift makes it lie
4. `.health` handler (the embed builders are covered, the handler is not)
5. `dispatch_interaction_command` — every slash command flows through it
6. `_run_reddit_watcher` body, `should_skip_duplicate_message`, `_sync_geonames`, the
   reddit and scrape settings panels and their dropdowns/modals

Ask the guardian for the rest rather than re-deriving it.

---

# 8. In flight — finish before starting Phase 0

**Status 2026-09-06: the code is done and passing; only the commit is left.**
`149 passed` across `test_reset_command.py`, `test_platform_support.py`,
`test_spawn_settle.py`, `test_province_codes.py` and `test_geo_priority.py`. The three
`test_reset_command.py` stubs that briefly failed (they needed `**kwargs` once
`handle_reset` began passing `log_path=`) are fixed.

**Still owed before committing:** one full-suite run
(`.venv/Scripts/python.exe -m pytest -q`) and a clean-checkout verify in a detached
worktree, diffed against the HEAD baseline per §6. The last full suite was 2,895 tests
before the `geo_priority` fix landed, so it has not been run against the final state.

**Uncommitted, verified except for that full-suite run:**

- `src/services/platform_support.py` — `spawn_detached` gained `log_path` and a settle wait.
  It used to report success the instant `Popen` returned, so a restart that died on startup
  looked identical to one that worked *and* latched `_restart_spawned`, permanently
  disabling `.reset`; the child's output went to DEVNULL so the reason was recorded nowhere.
  17 tests in `tests/test_spawn_settle.py` (untracked), **12/12 mutants killed**.
- `src/commands/handlers.py` — `.reset` passes `log_path=.restart.log`. **Cannot be
  committed** (753-line unrelated local diff), so `test_spawn_settle.py` deliberately holds
  no pin on `handle_reset`.
- `src/services/jba/geo_priority.py` + `tests/test_province_codes.py` (untracked) —
  `country_of("Toronto, ON")` returned the *country* `"ON"`. The two-letter-tail branch
  checked US state codes and never Canadian province codes. Self-consistent, so `build()`
  filed those boards under `"ON"` too and the ordering looked correct: a channel scoped to
  `"Mississauga, ON"` prioritised a **146-board bucket instead of the 931 under CA**. Fixed
  with `_CA_ONLY_SUBDIVISION_CODES`, excluding nl/pe/sk/nu because those are also ISO
  country codes (Netherlands, Peru, Slovakia, Niue) — resolving them would make
  "Amsterdam, NL" a Canadian board. 26 tests, **7/7 mutants killed**, 59 existing geo tests
  still pass.
- `tests/test_platform_support.py` — two mocked-Popen stubs replaced with a `_StillRunning`
  stand-in whose `wait()` raises `TimeoutExpired`, which is what a live child does.

Commit together once the stubs are fixed and the suite is green:
`src/services/platform_support.py`, `src/services/jba/geo_priority.py`,
`tests/test_platform_support.py`, plus untracked `tests/test_spawn_settle.py`,
`tests/test_province_codes.py`, `tests/test_reset_command.py`.

**Known-wrong claim, already committed.** `check_platform_yield.py`'s
`FIELD_CONSEQUENCE["location"]` says a blank location means the job is *"DROPPED from every
location-scoped search"*. **False for the platforms it is applied to** — both gates keep an
unplaceable location (`job_match._matches_channel_region` returns `country is None`;
`job_service.filter_rows_by_region` keeps unknown regions for ATS-sourced rows, which icims
is). The 90% floor is still defensible on other grounds; the stated cost is not, and it was
repeated in commit `f57803b`'s message.

**Narrowness to revisit** (raised by the repo owner, and correct): the province fix removed
the case that was costing coverage but not the class. `country_of` still ends in
`return tail.upper()`, so *any* unrecognised two-letter tail becomes a country —
"Springfield, XX" yields the country "XX". That is how junk buckets enter the geo index and
how the "ON" bucket came to look legitimate. The four excluded province codes are also a
hand-maintained literal, pinned by a test — a tripwire, not self-maintenance.

---

# 9. Facts worth not rediscovering

- `index.commoncrawl.org` (query service) times out from this machine; `data.commoncrawl.org`
  and `web.archive.org` answer in 0.1s. Use `--index ccbulk` — also faster (1,418 Oracle
  tenants in 9.8s). Known flakiness documented in the harvester, not a local block.
- 11 of 18 platforms have **zero Wayback queries on purpose** (`d2e68f5`, measured: 45 min
  vs 7, and it bought 145 live boards against 1,114 dead). Lever is the one platform that
  needs Wayback — it blocks CCBot and left Common Crawl after CC-MAIN-2025-38.
- Crawl age predicts liveness far better than crawl count: 12 recent crawls gave +5,378 at
  90.7% live; 8 crawls from 2022-23 gave +1,259 at 11.7%. Reaching further back keeps
  finding companies and they keep getting deader.
- Descriptions are thin in the archive (19-26%) **by design** — ATS list endpoints return a
  title and a link. `job_match.enrich_descriptions` fetches real text at query time for the
  top `ENRICH_CANDIDATES = 28` within a 40s budget.
- `workable` answers 429 (quota-metered). Expected to yield nothing; the refusal breaker
  caps it at 25 requests per cycle instead of 1,200.
- 16,675 slugs exist only on this disk, so CI republishes a smaller fleet than the bot runs.
  Closing that needs a push decision from the owner.
- Two session-only cron jobs may be running: a sustainability audit (5 min) and a
  plan-and-work pass (15 min). They die with the session.

---

# 10. Landed 2026-09-05 (11 commits, `cd7f3e0`..`f57803b`)

- `cd7f3e0` orphan audit counted names with a regex, so **documenting why something was
  unwired made it stop being reported**. Rewritten to count AST references. 13 → 0.
- `26e137c` deleted duplicate counters in `check_platform_yield`.
- `3d1da45` `geo_priority.refresh` had never been called; the fleet-ordering index was
  whatever snapshot was on disk. Now daily.
- `dc67423` `# orphan-ok:` markers on `plan_cycle` and `format_mode_summary`.
- `69a46d3` nothing ever ran `validate_ats_slugs`. Now daily, bounded, and with no
  `--platform` so a new platform is validated because it exists.
- `0db0d1f` the `services/jba` docstring claimed vendored copies; six of seven modules are
  imported directly by the bot.
- `a20d748` a test timed a cycle with a flat 0.5s sleep and began failing as the fleet grew.
- `8bb76c8` 137 slugs were in both the confirmed-live and dead stores, so the monitor could
  sample a slug the scraper refuses and record the platform as silent.
- `11510ad` `check_platform_yield` now runs daily, last of four hooks.
- `f57803b` one advisory floor governed three fields with opposite consequences; ashby's
  22% `date_posted` cleared it by two points while 78% of its postings skipped the
  freshness filter.

Also: the bot was restarted onto current code (pid 24480, had been 28 commits behind), and
oracle/personio were bootstrapped into the confirmed-live store (933/1,418 and 537/736 live).
