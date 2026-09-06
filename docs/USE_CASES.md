# Use-case inventory

What this bot actually does for the people using it, with a file:line behind every
claim and the tests (if any) that would catch a regression.

Read this before changing anything under `src/`. The column that matters most is
**Tests** — a use case with `NONE` is one a refactor can break in total silence.

Scope note: several modules carry large uncommitted working-tree changes
(`src/services/ats_service.py`, `src/services/job_service.py`,
`sync_ats_companies.py`, `tests/test_dead_slug_ttl.py`). Everything below
describes the **working tree**, because that is what the running bot executes.

---

## 1. Command surface

### 1.1 How a command reaches a handler

| Piece | Where |
| --- | --- |
| Command constants `CMD_* = ".name"` | `src/commands/handlers.py:126-148` |
| Slash aliases `_COMMAND_ALIASES` | `src/commands/handlers.py:152-176` |
| Alias expansion | `src/commands/handlers.py:294-305` (`_expand_command_aliases`) |
| Prefix matcher (token-boundary; `.st123` does not match `.st`) | `src/commands/handlers.py:452-465` (`_command_matches`) |
| Decorator | `src/commands/handlers.py:483-497` (`command_handler`) |
| Handler tuple, dispatch order | `src/commands/handlers.py:534-552` |
| Text dispatch | `src/commands/handlers.py:2176-2180` (`dispatch`), called from `src/app.py:368-384` (`on_message`) |
| Slash dispatch (adapts an Interaction into a message-shaped object) | `src/commands/handlers.py:2182-2221` (`dispatch_interaction_command`), adapter at `:60-96` |

Every slash command in `src/app.py:209-321` re-dispatches as **text** through the
same router, so argument parsing has exactly one home. Consequence: a change to
`_extract_command_payload` changes both the `.` and `/` forms at once.

**Registered slash commands (15)** — `src/app.py:209-321`:
`commands, status, resumebuild, resumecoverbuild, resumecheck, continue,
hello2, jobsettings, jobsinit, redditsettings, clearredditseen,
resetredditseen, settings, scrape, bestjobs`.

**Gap: 30 unregistered slash forms** — `_COMMAND_ALIASES` declares 45 slash forms
total (`:152-176`), but only 15 are registered via `client.tree.command`. The 30
that Discord never offers users are: `/cmd`, `/st`, `/watch`, `/watcherstatus`,
`/resume`, `/ernestresume`, `/res`, `/coverbuild`, `/cover`, `/more`, `/hi`,
`/hello`, `/job`, `/jobs`, `/jobtest`, `/jobpipelinetest`, `/reddit`, `/rset`,
`/rclear`, `/rreset`, `/scrapecfg`, `/cfg`, `/health`, `/whealth`, `/bestmatches`,
`/best`, `/quota`, `/quotas`, `/reset`. They are reachable only by typing the dot
form. Not a bug to fix blind; just do not assume `/health` works.

### 1.2 Per-command inventory

Permission legend: **guild** = requires a server channel (fails in DM);
**owner** = guild owner only, via `_is_guild_owner`
(`src/commands/handlers.py:814-817`).

| Command | Aliases / slash | Who | What the user gets | Implementation | Reads / writes | Tests |
| --- | --- | --- | --- | --- | --- | --- |
| `.cmd` | `/cmd`, `/commands` | anyone | Embed cheat sheet of every command, in one of three flavours (general / job / reddit) | `handlers.py:574-577`, embed built at `:190-283` | none | Embed builder + auto-pin: `tests/test_cheatsheet_pinning.py`. Handler dispatch itself: **NONE** |
| `.st` / `.watch` | `/st`, `/status`, `/watch`, `/watcherstatus` | anyone | Per-channel line: job watcher enabled/running, reddit watcher enabled/running, which process holds the channel lock | `handlers.py:676-697` | reads `store.get_job_settings`, `get_reddit_settings`, `watcher_manager.channel_*_tasks`, `channel_active_process` | **NONE** |
| `.health` | `/health`, `/whealth` (not registered) | anyone | Scrape-health dashboard for this channel; `.health all` gives every channel. Includes ATS platform roster, fleet coverage, scheduler queue, browser/Reddit session state | `handlers.py:699-751`; embeds `src/services/health.py:561-` and `:696-` | reads `WatcherHealthTracker`, `scheduler.stats()`, `browser_service.session_valid()` | Embed builders: `tests/test_health_browser.py`, `test_health_coverage_field.py`, `test_health_platform_roster.py`, `test_health_reddit_session.py`. Handler: **NONE** |
| `.resumebuild` | `/resumebuild`, `/resume`, `/ernestresume`, `/res` | guild; self only unless owner | Reply to a bot job listing → tailored LaTeX resume compiled to PDF, attached | `handlers.py:1255-1470`; shared prep `:1143-1253` | reads `resumes_cache/<key>/{baseinfo,instructions,template}.tex/txt`; writes `resume_cache_dir/compile_logs/`, `failed_latex/`, Gemini explicit cache | `tests/test_resume_directive_wiring.py`, `test_resume_service.py`, `test_structured_resume.py`, `test_resume_attachment_fallback.py`, `test_latex_*`, `test_field_floors.py`, `test_grounding_audit.py`, `test_domain_fit_audit.py`, `test_skills_rewrite.py` |
| `.resumecoverbuild` | `/resumecoverbuild`, `/coverbuild`, `/cover` | guild; self only unless owner | Cover letter PDF covering what the resume left out | `handlers.py:1482-1569` | same profile dir; writes `*.coverbuild.log` | `tests/test_cover_letter.py`, `test_resume_directive_wiring.py` |
| `.resumecheck` | `/resumecheck` | guild; self only unless owner | Compiles the user's current `template.tex` as-is → PDF, or the LaTeX error | `handlers.py:1572-1678` | reads `template.tex`; writes `*.resumecheck.log` | **NONE for the handler** (compile path covered by `test_resume_service.py`, `test_latex_engine_family.py`) |
| `.more` | `/more`, `/continue` | anyone | Next chunk of a paginated reply | `handlers.py:2346-2360`; producer `set_continuation` `:2223-2229` | in-memory `channel_continuations` only — **lost on restart** | **NONE** |
| `.hi` | `/hi`, `/hello`, `/hello2` | anyone | "Hello!" liveness check | `handlers.py:2282-2285` | none | **NONE** |
| `.job` / `.jobs` | `/job`, `/jobsettings`, `/jobs`, `/jobsinit` | anyone (panel then gates by clicker) | Job-watcher settings panel: sources, date window, refresh rate, exclusions, thresholds, region toggle, start/stop, and the three resume-cache file editors | `handlers.py:2373-2411`; view `src/ui/views.py:823-1095` | reads/writes `store.channel_job_settings`; seeds `resumes_cache/<key>/` via `ensure_profile_cache` | `tests/test_jobsettings_modal.py`, `test_job_region_control.py`. Handler itself: **NONE** |
| `.jobtest` | `/jobtest`, `/jobpipelinetest` (not registered) | anyone | Full pipeline dry run with counts at each stage: retrieved per site → role filter → exclusions → semantic → URL dedup → FIFO dedup → Discord-history dedup → "would send N". All dedup read-only | `handlers.py:1998-2172` | reads job settings, `store.channel_job_seen`, dedup listing files, channel history. Writes nothing | **NONE** |
| `.bestjobs` / `.best` | `/bestjobs`, `/bestmatches`, `/best` | anyone; naming another profile is owner-only | Top archive matches for the caller's resume profile. `.best week 15 --fast` — window, count, profile, fast mode | `handlers.py:1689-1775`; parser `:411-448`; engine `src/services/job_match.py:1835-1966` | reads `data/jba/jobs` archive, `resumes_cache/<key>/`, this channel's job settings for scoping | Strong: `tests/test_job_match.py` (157 tests, incl. end-to-end router dispatch at `:793-`), `test_command_aliases.py` for the parser |
| `.reddit` / `.rset` | `/reddit`, `/redditsettings`, `/rset` | anyone (panel gates by clicker) | Reddit media-watcher panel: sort, time filter, media mode, refresh, subreddit/flair/NSFW modal, start/stop | `handlers.py:2413-2431`; view `views.py:1224-1259` | reads/writes `store.channel_reddit_settings` | **NONE** |
| `.rclear` / `.rreset` | `/rclear`, `/clearredditseen`, `/rreset`, `/resetredditseen` | anyone | Wipes this channel's seen-post list so the reddit watcher re-posts | `handlers.py:1681-1687` | **overwrites** `.reddit_seen_<channel_id>.json` | **NONE** — and this is a destructive write with no test |
| `.scrapecfg` / `.cfg` | `/scrapecfg`, `/settings`, `/cfg` | anyone (panel gates by clicker) | Generic-scrape settings panel: max items, timeout, AI cleanup on/off | `handlers.py:2362-2371`; view `views.py:346-355` | reads/writes `store.channel_scrape_settings` | **NONE** |
| `.scrape` | `/scrape` | anyone | One-off scrape of a URL (`.scrape <url> [\| css-selector]`). Job-board URLs route through JobSpy; everything else through the generic scraper, optionally cleaned by OpenRouter | `handlers.py:2287-2344` | takes the channel process lock; reads scrape settings; network egress | **NONE** |
| `.quota` | `/quota`, `/quotas` (not registered) | **owner**, guild | View or set per-member / per-role shares of the expensive AI commands. `.quota default 0.5`, `.quota role @R 0.7`, `.quota user @m 0.3`, `.quota clear user @m` | `handlers.py:1902-1996`; parser `:311-332`; policy `src/services/quota.py` | reads/writes `store.guild_quotas` → `.bot_state.json` | `tests/test_quota.py` (21) |
| `.reset` | `/reset` (not registered) | **owner** | Restarts the bot: spawns `run.py --forcerun` detached, which kills this process and starts a fresh interpreter | `handlers.py:1850-1900` | spawns a process; writes `.restart.log` | `tests/test_reset_command.py`, `test_spawn_settle.py`, `test_shutdown_signal.py` |

### 1.3 Cross-cutting command behaviours

These are easy to break from a distance because no single command owns them.

| Behaviour | Where | Tests |
| --- | --- | --- |
| `(...)` free-form LLM directive, depth-tracked, 600-char cap, stripped before target resolution | `handlers.py:365-406` | `tests/test_command_aliases.py` |
| `--aggressive` / `--strongaggressive` flag stripping, order-sensitive vs. the directive | `handlers.py:344-358`; ordering comment `:1257-1261` | `tests/test_command_aliases.py` |
| Resume target resolution (`@mention`, username, multi-word display name) and the owner-only rule for targeting someone else | `handlers.py:814-1041` | `tests/test_command_permissions.py` |
| Quota scaling of resume/bestjobs work (`allowance.scale`, `blocked`) | `handlers.py:1780-1806`; `services/quota.py:157-201` | `tests/test_quota.py` |
| Interactive work runs on the shared priority scheduler under `work_guard()` so shutdown drains it | `handlers.py:554-572` (`_run_interactive`) | `tests/test_priority_scheduler_integration.py` |
| Job-post text may arrive as a `.txt`/`.md` attachment instead of message body | `handlers.py:768-789` | `tests/test_resume_attachment_fallback.py` |
| Long output chunking + `.more` continuation | `handlers.py:2223-2229` | **NONE** |
| Settings panels are edited in place / re-sent per `panel_key` | `handlers.py:2231-2280` | **NONE** |
| Cheat sheet auto-pinned when a watcher starts | `handlers.py:579-674`; trigger `src/watchers/manager.py:1288, 1311` | `tests/test_cheatsheet_pinning.py` |

---

## 2. Background loops

All started from `src/watchers/manager.py`. If one dies, the watchdog
(`:1215-1273`) restarts it within 30 s — but only for the three it knows about.

### 2.1 Job watcher — per channel

- **Where:** `manager.py:514-625`, started by `start_job_watcher` `:1275-1289`.
- **Cadence:** `max(60, refresh_seconds)`, default 900 s (`state/store.py:30`).
- **Does:** scrapes the channel's configured sites via `job_service.scrape_job_postings`
  → role filter → exclusion terms → semantic filter (separate threshold for ATS
  sources, `:541-565`) → canonical-URL dedup against `store.channel_job_seen`
  → posts each survivor → logs sent items into the archive via
  `merge_data.log_jobs` (`:601-612`).
- **Writes:** Discord messages; `store.channel_job_seen` (capped at
  `DEDUP_SEEN_LINKS_CAP` = 80 000, `job_service.py:135`); the dedup listing
  files; `data/jba/jobs`.
- **If it silently stops:** the channel simply goes quiet. There is no "no new
  jobs" message, so a dead watcher and a slow job market look identical. `.st`
  and `.health` are the only tells.
- **Tests:** `tests/test_watcher_dedup_resilience.py`, `test_job_scrape_dedup_cache.py`,
  `test_log_jobs_dedup.py`, `test_job_region_control.py`.

### 2.2 Reddit watcher — per channel

- **Where:** `manager.py:627-676`, started by `start_reddit_watcher` `:1300-1312`.
- **Cadence:** `max(60, refresh_seconds)`, default 300 s.
- **Does:** polls each configured subreddit concurrently
  (`reddit_service.poll_subreddit_media`), posts new media, galleries via
  `send_reddit_gallery_message` (`:380-422`).
- **State:** `.reddit_seen_<channel>.json` when a single subreddit is configured
  (legacy filename preserved on purpose, `:652-657`), otherwise
  `.reddit_seen_<channel>_<subreddit>.json`.
- **If it silently stops:** channel goes quiet; seen-state stops advancing, so a
  restart may re-post a burst.
- **Tests:** `tests/test_reddit_service_circuit_breaker.py`,
  `test_reddit_service_concurrency.py`, `test_health_reddit_session.py`. The loop
  method `_run_reddit_watcher` itself: **NONE**.

### 2.3 ATS scrape loop — process-wide

- **Where:** `manager.py:1037-1207`, ensured by `_ensure_ats_scrape_loop` `:1209-1213`.
- **Cadence:** 4 cycles/day, one every 6 h (`:1041-1042`). Sleeps an hour when no
  channel has a job watcher enabled (`:1074-1080`).
- **Does:** fans 18 platforms (`ats_service.py:59-62`) out concurrently, each its own
  scheduler submission at BACKGROUND tier, each bounded by `ATS_PLATFORM_TIMEOUT_S`
  and the whole cycle by `ats_cycle_timeout` (`manager.py:98-122`). Slug order comes
  from `_ordered_slugs` (`:703-740`), which combines geo priority with a persisted
  rotation cursor so a cut-off cycle resumes where the last one stopped. Results
  go to `merge_data.log_jobs`; health is recorded per platform.
- **After each cycle:** `commit_archives_daily` (publishes archive zips when
  `JBA_ARCHIVE_GIT_COMMIT` is set) and `_sync_geonames`.
- **`bamboohr` is conditional** on `config.ats_bamboohr_enabled` (`:1083-1086`).
  `workable` answers 429 by design and is expected to look unhealthy.
- **If it silently stops:** `.bestjobs` slowly degrades — the archive stops
  growing, so "best matches for the last day" returns yesterday's data and then
  nothing. Watchdog restarts it only while at least one job watcher is enabled
  (`:1236-1240`).
- **Tests:** `tests/test_ats_scrape_scheduling.py`, `test_ats_scrape_ordering.py`,
  `test_ats_fleet_rotation.py`, `test_ats_cycle_budget.py`, `test_ats_silence.py`,
  `test_ats_refusal_breaker.py`, `test_bamboohr_enabled.py`, `test_health_platform_roster.py`.

### 2.4 Daily UTC-rollover hooks

Run in order, once per new UTC day, inside the ATS loop (`manager.py:1052-1063`).
The `_is_new_utc_day` guard (`:836-848`) means they do **not** fire on the loop's
first pass — `run.py` has just synced.

| Hook | Where | What it does | Silent-failure consequence | Tests |
| --- | --- | --- | --- | --- |
| `_refresh_company_lists` | `manager.py:850-894` | Runs `sync_ats_companies.py`, then `ats_service.reload_company_lists()` | The in-process company cache lives for the life of the process. Without this a long-uptime bot scrapes the fleet it booted with and never sees a newly-harvested employer | `tests/test_ats_company_refresh.py` |
| `_validate_company_slugs` | `manager.py:896-951` | Runs `scripts/validate_ats_slugs.py` under a wall-clock budget; writes dead marks and confirmed-live records | Dead boards keep costing one request each per cycle, forever. Measured live rates run 87% down to 11% for fresh slugs | `tests/test_slug_validation_loop.py`, `test_validate_ats_slugs.py`, `test_validator_outage_guard.py` |
| `_refresh_geo_index` | `manager.py:1001-1035` | `geo_priority.refresh()` — rebuilds the country index that orders slugs | Ordering silently degrades to no-op (`slugs_for` returns empty), so a cut-off cycle cancels exactly the boards most likely to yield relevant jobs | `tests/test_geo_index_refresh.py`, `test_geo_priority.py` |
| `_check_platform_yield` | `manager.py:953-999` | Runs `scripts/check_platform_yield.py --days 7`; **reports only, changes nothing** | You lose the only signal that distinguishes "platform has no scraper / is switched off" from "its companies have nothing open". This is how bamboohr and paylocity were found contributing zero against 13 755 and 9 325 live companies | `tests/test_platform_yield_loop.py`, `test_platform_yield.py` |

Each hook is wrapped so a failure prints and continues — losing an optimisation
must never cost the scrape that was about to run.

### 2.5 Watchdog

`manager.py:1215-1273`, every 30 s. Restarts any enabled-but-dead job watcher,
reddit watcher, or the ATS loop; logs scheduler stats and learned per-label
costs. **If the watchdog itself dies, nothing restarts it.**
Tests: `tests/test_priority_scheduler_integration.py:181`.

### 2.6 Startup restore

`restore_enabled_watchers` (`manager.py:1323-1341`), called from `on_ready`
(`app.py:324-327`). Restarts watchers marked enabled; purges job state for
channels that no longer exist. Tests: `tests/test_watcher_dedup_resilience.py:154-260`.

---

## 3. Pipelines behind the commands

### 3.1 Resume building — `src/services/resumes/`

| Stage | Module | Notes |
| --- | --- | --- |
| Profile cache seeding, key resolution, LaTeX compile | `resume.py` | `ensure_profile_cache`, `discord_profile_key`, `compile_latex_to_pdf`; required files `baseinfo.txt`, `instructions.txt`, `template.tex` (`resume.py:21`) |
| Structured (JSON) profile mode | `structured.py` | Bypasses the Gemini explicit cache entirely (`handlers.py:1288-1296`) |
| Job-context extraction + rewrite + LaTeX repair + condense | `listing.py` | `extract_job_context_from_message`, `generate_resume_rewrite`, `repair_latex_until_compiles` (2 rounds), `condense_latex_if_overflowing` |
| Cover letter | `cover.py` | `generate_cover_letter` |
| Gemini explicit context cache | `cache.py` | `ResumeExplicitCacheManager` |
| Provider settings / fallback order | `configkey.py`, `resume.py:25` | gemini → gemini-flash → openrouter → groq |
| ATS/PDF quality audit | `ats_check.py` | Type-3 bitmap-font detection |

Failure modes are all user-visible: missing key, missing profile files, generation
failure, compile failure (with the failing `.tex` dumped under
`resume_cache_dir/failed_latex/`).

Tests: `test_resume_service.py` (151), `test_structured_resume.py` (349),
`test_cover_letter.py`, `test_ats_check.py`, `test_latex_bitmap_font_salvage.py`,
`test_latex_engine_family.py`, `test_field_floors.py`, `test_grounding_audit.py`,
`test_domain_fit_audit.py`, `test_skills_rewrite.py`, `test_profile_bootstrap.py`.

### 3.2 `.bestjobs` ranking — `src/services/job_match.py`

Three stages (module docstring `:1-35`):
1. Lexical prefilter over the whole window → `LLM_CANDIDATES` (60, `:121`).
2. Description enrichment for the top `ENRICH_CANDIDATES` (28, `:140`) — real
   network fetches, cached in `description_cache`.
3. One batched LLM judgement.

Fail-open: with no provider or a total provider failure it returns the lexical
order and says so in the report header (`:1977-1986`). Channel scoping
(`role_filters`, `exclusion_terms`, `allow_north_america`) is applied *before*
ranking (`:1849-1862`).

Tests: `tests/test_job_match.py` (157), `test_description_cache.py`,
`test_role_filter_levels.py`, `test_job_level.py`.

### 3.3 Archive / dedup — `src/services/jba/`

| Module | Role |
| --- | --- |
| `merge_data.py` | `log_jobs` writes into `data/jba/jobs/jobs.db`; weekly rollover to `<month>/<week>.zip`; monthly consolidation; `commit_archives_daily` publishes zips via git when `JBA_ARCHIVE_GIT_COMMIT` (push additionally needs `JBA_ARCHIVE_GIT_PUSH`) |
| `archive_index.py` | SQLite index over the archive zips, used by ATS archive dedup (`ats_service.py:2868-2886`) and rebuilt on schema change — it **drops** its tables, which is why the description cache deliberately does not live there |
| `description_cache.py` | Job descriptions keyed on posting URL. Successes cached forever, failures for 3 days (`:52-57`) |
| `geo_priority.py` | Country→slug index learned from the archive; **orders, never filters** |
| `geo_db.py`, `geolocation.py` | `data/geo.db`; Glassdoor location-ID caching |
| `scraper.py` | The per-platform ATS scrapers plus its own dead-slug store at `data/dead_slugs` (`:732-750`) |

Tests: `test_archive_index.py`, `test_archive_metrics.py`, `test_archive_git_publish.py`,
`test_log_jobs_dedup.py`, `test_ats_dedup_identity.py`, `test_description_cache.py`,
`test_geo_priority.py`, `test_build_geo_db.py`, `test_geo_db_city_ranking.py`,
`test_province_codes.py`, `test_jba_package_docs.py`.

### 3.4 ATS scrapers — `src/services/ats_service.py`

18 platforms (`:30-62`). Dead-slug lifecycle: 90-day TTL (`:124`), 7-day recheck
window (`:131`), plus a per-slug 0-7 day spread (`:139`) so ~19 800 dead slugs do
not all re-probe on the same day and re-synchronise. All the shared dicts are
guarded by a reentrant lock (`:239`) because 30-50 threads per platform mutate
them while 6 platforms run concurrently.

`load_company_lists` (`:269-332`) unions three sources: upstream
`data/ats_companies`, local `data/ats_harvest`, published `data/ats_harvest_ci`.
It warns loudly when the **upstream** directory is missing rather than when the
merged result is empty — `data/ats_companies` is gitignored, so on a fresh
deployment it is absent and every platform would otherwise scrape zero companies
indistinguishably from "no jobs matched".

Tests: `test_new_ats_scrapers.py`, `test_oracle_personio_scrapers.py`,
`test_ats_scraper_hardening.py`, `test_ats_traversal.py`, `test_dead_slug_ttl.py`,
`test_ats_refusal_breaker.py`, `test_probe_scraper_alignment.py`,
`test_probe_discrimination.py`, `test_monitor_ats_scrapability.py`,
`test_ziprecruiter_scraper.py`, `test_harvest_ats.py`, `test_harvest_sync.py`,
`test_compare_upstream.py`.

### 3.5 Quota

`src/services/quota.py`. Shares are 0-1. Guild owner is always `FULL_SHARE`
(1.0, `:45`) and cannot be limited. A share below `BLOCKED_BELOW` (0.05, `:56`)
refuses the command outright. Precedence: user override → role → default.
Scaling degrades a command (fewer LLM/enrich candidates) rather than failing it.
Tests: `tests/test_quota.py`.

### 3.6 Health

`src/services/health.py`. `WatcherHealthTracker` (`:159-399`) accumulates
per-channel scrape events, per-ATS-platform results, fleet coverage, browser
events, watchdog restarts, bot uptime. Everything is **in-process** — a restart
zeroes it, which is exactly why `_check_platform_yield` reads a week of committed
archive instead. Tests: the four `test_health_*.py` files.

### 3.7 Scheduler

`src/services/priority_scheduler.py`. Two tiers; INTERACTIVE preempts BACKGROUND;
a background task waiting 20 s is promoted so it cannot starve (`:27`). Costs are
learned per label from a rolling median. Tests: `test_priority_scheduler.py`,
`test_priority_scheduler_integration.py`, `test_capacity.py`,
`test_capacity_integration.py`.

### 3.8 ATS harvest — `scripts/harvest_ats.py`

Rebuilds the company lists under `data/ats_harvest/` from the Common Crawl CDX API,
independent of upstream `data/ats_companies`. These are later unioned by
`ats_service.load_company_lists` (`:269-332`), so a bad harvest degrades to "extra
404 slugs" rather than a broken list.

**Public surface** (used by `validate_ats_slugs.py` and `compare_upstream.py`):
- `PLATFORM_BY_NAME: dict[str, Platform]` (`:623`) — maps platform names to their
  extractors and queries.
- `Platform(dataclass)` (`:234-240`) — immutable: `name: str`, `queries: tuple[tuple[str, str], ...]`,
  `extract: Callable[[str], str | None]`.
- `_INFRA_SUBDOMAIN_RE: re.Pattern[str]` (`:113-116`) — filters infrastructure
  subdomains before probing.
- `_current_identifier(platform: Platform, slug: str) -> str | None` (`:1680-1693`) —
  validates a stored identifier against the current extractor; returns None if the slug
  is no longer valid under today's rules.
- `output_lock(out_dir: Path) -> ContextManager` (`:1785-1820`) — refuses to run two
  harvests against the same directory; context manager.
- `HarvestLocked(RuntimeError)` (`:1744-1745`) — raised by `output_lock` when a harvest
  is already in progress.

Tests: `test_harvest_ats.py`, `test_harvest_sync.py`.

---

## 4. Data stores

| Store | Written by | Read by | If lost |
| --- | --- | --- | --- |
| `.bot_state.json` | `RuntimeStore.save()` — `src/state/store.py:176-193`, called on every settings/quota/seen change | `RuntimeStore.load()` `:195-236`, at boot (`app.py:457`) | Every channel's job/reddit/scrape settings, quotas, seen-URL sets and pinned-cheatsheet ids. Watchers do not restart; channels go silent |
| `data/ats_companies/` | `sync_ats_companies.py` (rewrites **wholesale**) | `ats_service.load_company_lists` `:269-332` | Gitignored (carries its own git history). Missing → loud warning, fleet falls back to harvest only |
| `data/ats_harvest/` | `scripts/harvest_ats.py` | `load_company_lists` (unioned) | Holds ~14 278 slugs the published set lacks — Wayback finds CI stopped sweeping for |
| `data/ats_harvest_ci/` | published by the harvest workflow, pulled by `sync_ats_companies.py` | `load_company_lists` (unioned) | Holds ~243 slugs the local set lacks. Neither directory is redundant |
| `data/dead_slugs/<platform>.json` | `ats_service._mark_dead` / `flush_dead_slugs` `:408-424`; `scripts/validate_ats_slugs.py`; also `jba/scraper.py:748` | `ats_service._is_dead` `:436-453`; `jba/scraper.py:565` | Every dead board is re-requested for a full TTL. This file is what stops ~19 800 wasted requests |
| `data/ats_checked/<platform>.json` | `scripts/validate_ats_slugs.py` only | `scripts/monitor_ats_scrapability.py:39`, `scripts/check_platform_yield.py:40` | Confirmed-live sampling and yield monitoring stop working. Before the daily hook existed, these files only existed where someone had typed the command by hand |
| `data/jba/jobs/jobs.db` | `merge_data.log_jobs` `:839`; `description_cache` side table | `job_match`, `geo_priority`, `mcp_server` | Current week of archive plus every cached description. `.bestjobs` degrades to titles-only against older zips |
| `data/jba/jobs/<month>/<week>.zip` | `merge_data._archive_old_weeks` `:179`, `_consolidate_old_months` `:248` | `archive_index`, `job_match`, `geo_priority`, `scripts/compute_metrics.py` | The whole historical archive `.bestjobs week` ranks over |
| `data/geo.db` | `scripts/build_geo_db.py`, `sync_geonames.py` | `jba/geo_db.py`, region inference | Region filtering and Glassdoor location lookup degrade |
| `.reddit_seen_<channel>[_<sub>].json` | reddit watcher; **wiped by `.rclear`** | reddit watcher | Re-posts a burst of already-seen media |
| `.message_listing_<channel>_<type>.json` + dedup dir | `job_service.record_message_for_dedup` | `is_message_duplicate` | FIFO title-signature dedup layer is lost; duplicates reappear |
| `data/ats_baseline.json`, `data/ats_discovery_state.json` | `scripts/validate_ats.py`, discovery/rotation state | validation harness, `_ats_rotation_state_path` `manager.py:742` | Rotation cursor resets — a partial cycle restarts from the top instead of resuming |

---

## 5. UI surfaces (`src/ui/views.py`)

### 5.1 Reachable

| Surface | Opened by | Controls | Tests |
| --- | --- | --- | --- |
| `ScrapeSettingsView` `:346` | `.scrapecfg` / `.cfg` | Max items, Timeout, AI cleanup | **NONE** |
| `JobSettingsView` `:823` | `.job` / `.jobs` | Rows 0-2: `JobSourceDropdown`, `JobDateDropdown`, `JobRefreshDropdown`. Row 3: Edit exclusions, Advanced (thresholds), Edit text/results, Region toggle. Row 4: Edit base info, Edit instructions, Edit template, Start/Stop watcher | `test_jobsettings_modal.py`, `test_job_region_control.py` |
| `JobTextModal` `:435` | "Edit text/results" | keywords, location, radius, results, role filters (free text) | `test_jobsettings_modal.py` |
| `JobExclusionTermsModal` `:536` | "Edit exclusions" | exclusion terms | **NONE** |
| `JobThresholdModal` `:581` | "Advanced" | semantic threshold + separate ATS threshold | **NONE** |
| `ResumeFileEditModal` `:651` / `TemplateEditModal` `:752` | the three green row-4 buttons | Edits `baseinfo.txt` / `instructions.txt` / `template.tex` in 3×4000-char parts; the template editor also compiles a preview | `test_jobsettings_modal.py` |
| `RedditSettingsView` `:1224` | `.reddit` / `.rset` | Sort, Time filter, Media mode, Refresh; modal for subreddits/flair/NSFW; Start/Stop | **NONE** |
| `RedditTextModal` `:1185` | "Edit subreddit/media/filter" | subreddits, flair, media mode, content filter, limit | **NONE** |

Permission model for panels: `JobSettingsView._is_allowed` (`:865-876`) accepts the
menu opener, the guild owner, or `config.main_user_id`. The Reddit and Scrape
panels are stricter — menu opener only (`:1239-1243` and each dropdown callback).

### 5.2 Built but unreachable

**`ModeDropdown` / `ModeDropdownView` (`views.py:255-281`) — no command opens it.**
The whole channel-mode feature is orphaned: `store.set_mode` / `get_mode`
(`state/store.py:131-135`), `MODE_DESCRIPTIONS` (`:7-12`) and
`format_mode_summary` (`views.py:59-`) all still work, but nothing in
`handlers.py` or `app.py` constructs the view, so no user can change a channel's
mode and `channel_modes` in `.bot_state.json` can only ever hold the default.
The code carries an explicit `# orphan-ok` note at `views.py:53-58` saying the
missing piece is the entry point, not the renderer. **No tests.**

**`JobRoleFilterDropdown` (`views.py:391-411`)** — has options, a callback and its
own row, and is never added to any view. `JobSettingsView.__init__` explains why
(`:852-858`, `:1036-1039`): the panel's five action rows are full, so
`role_filters` stays reachable only through the free-text `JobTextModal`, where
"internship" has to be typed exactly. `tests/test_job_region_control.py:14`
asserts this state deliberately.

---

## 6. Non-Discord surface

`scripts/mcp_server.py` exposes the archive read-only over MCP: `search_jobs`
(`:83`), `archive_stats` (`:121`), `archive_days` (`:150`), `ats_coverage`
(`:174`). All annotated read-only. Depends on `data/jba/jobs` and, for coverage,
on `ats_service.load_company_lists` + `_load_dead_slugs`. Tests: **NONE**
directly; it reuses tested primitives.

`run.py` boot sequence (`:128-160`): kills any stale bot, then runs
`sync_ats_companies.py` unless `--skip-ats-sync`, then launches `src/app.py`
forwarding signals. `install_shutdown_signal_handler` (`app.py:404-433`) turns
SIGTERM into the graceful-drain path. Tests: `test_shutdown_signal.py`,
`test_spawn_settle.py`, `test_container_config.py`, `test_port_wiring.py`.

---

## 7. Use cases with no test coverage

Ordered by how much damage a silent break would do.

1. **`.rclear` / `.rreset`** — destructive overwrite of a channel's reddit seen
   file (`handlers.py:1681-1687`). No test asserts it writes the right file or
   the right shape.
2. **`.scrape`** — takes the channel process lock, does network egress, and has a
   `finally` release. A leaked lock blocks the job watcher for that channel.
3. **`.jobtest`** — 175 lines reproducing all three dedup layers read-only
   (`handlers.py:1998-2172`). If it drifts from the real watcher it lies about
   what would be sent, which is its only purpose.
4. **`.health` handler** — the embed builders are well covered, the handler that
   assembles their inputs is not.
5. **`.st`** — the only quick way to see whether a watcher is alive.
6. **`.resumecheck` handler** — profile resolution and owner override are
   duplicated from `_prepare_resume_request` rather than shared.
7. **Reddit settings panel end to end** — `RedditSettingsView`, `RedditTextModal`
   and all four Reddit dropdowns.
8. **Scrape settings panel** — `ScrapeSettingsView` and its three dropdowns.
9. **`JobExclusionTermsModal` and `JobThresholdModal`** — both write settings the
   watcher and `.bestjobs` read.
10. **`.more` / `set_continuation`** — pagination, in-memory only.
11. **`.cmd` and `.hi` handlers** — trivial, but `.cmd` is the discovery surface.
12. **`_run_reddit_watcher`** — the loop body itself (the service under it is tested).
13. **`should_skip_duplicate_message`** — the Discord-history dedup layer.
14. **`dispatch_interaction_command`** — every slash command flows through it.
15. **`_sync_geonames`** — runs every ATS cycle, shells out, unverified.
16. **The channel-mode surface** — unreachable *and* untested (see 5.2).

---

## 8. Quick "would this break a user?" checklist

- Touching `handlers.py` command constants or `_COMMAND_ALIASES`? Both the dot
  form and the slash form change. `tests/test_command_aliases.py` guards the
  parse but nothing guards the handlers.
- Touching `_extract_llm_directive` or the flag strippers? Anything left in the
  message is read as a **target username** — that is the failure mode these
  guard against.
- Touching `RuntimeStore` keys or `save()`/`load()`? `.bot_state.json` is the
  only persistence for settings, quotas, and seen-URL sets.
- Removing an ATS platform from `ATS_PLATFORMS`? `.health`'s roster, the yield
  check, the validator and the harvest all enumerate from it.
- Changing `load_company_lists` merge behaviour? The union across three sources
  is deliberate; any merge that can *shrink* a list drops companies the bot
  already scrapes.
- Changing dead-slug TTL / recheck / spread? Removing the spread re-synchronises
  ~19 800 slugs into one cycle.
- Removing a daily rollover hook? See the consequence column in 2.4 — three of
  the four fail silently by construction.
- Adding a control to `JobSettingsView`? All five action rows are full.
