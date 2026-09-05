# Rebuilt Discord Scraper Bot

A Discord bot that continuously watches job boards and ATS platforms, mirrors
Reddit feeds, scrapes arbitrary web pages on demand, and generates LaTeX resumes
and cover letters tailored to a specific job posting. It is a modular rebuild of
an earlier single-file scraper bot.

The bot is designed to run unattended for weeks at a time on Windows or Linux
(see [Linux deployment](#linux-deployment)), so a large share of the code is about survival rather than features: process locks,
watchdogs, circuit breakers, proxy rotation, dead-slug caches, graceful drains,
and multi-provider LLM fallback. This README documents all of it.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Running the bot](#running-the-bot)
   - [Linux deployment](#linux-deployment)
5. [Configuration](#configuration)
6. [Commands](#commands)
7. [Architecture](#architecture)
   - [Startup and process lifecycle](#startup-and-process-lifecycle)
   - [The priority work scheduler](#the-priority-work-scheduler)
   - [Runtime state](#runtime-state)
   - [Watcher manager](#watcher-manager)
   - [Job scraping pipeline](#job-scraping-pipeline)
   - [ATS scraping](#ats-scraping)
   - [The JBA subsystem](#the-jba-subsystem)
   - [Best-match ranking](#best-match-ranking)
   - [Browser service](#browser-service)
   - [Reddit service](#reddit-service)
   - [Generic scraping](#generic-scraping)
   - [Health tracking](#health-tracking)
   - [Resume system](#resume-system)
8. [Deduplication](#deduplication)
9. [Data directory](#data-directory)
10. [Testing and CI](#testing-and-ci)
11. [Operational runbook](#operational-runbook)
12. [Repository layout](#repository-layout)
13. [Companion projects](#companion-projects)

---

## What it does

**Job watching.** Per Discord channel, you configure keywords, location, radius,
freshness window, role filters, exclusion terms, and which sites to hit. A
background loop scrapes on an interval, filters results (keyword and optionally
semantic), dedupes against a rolling history, and posts each surviving listing
as a message. Sources are JobSpy (Indeed, LinkedIn, Glassdoor, ZipRecruiter,
Google, Bayt, Naukri, and whatever else the installed version supports) plus
first-party scrapers for Glassdoor and ZipRecruiter and direct ATS scrapers for
sixteen job-board platforms: Greenhouse, Lever, Ashby, Workday, iCIMS, Workable,
Breezy, SmartRecruiters, Recruitee, TeamTailor, Rippling, JazzHR, Jobvite,
ApplicantPro, Paylocity, and (optionally) BambooHR.

**Reddit watching.** Per channel, mirror one or more subreddits with sort,
time-filter, flair, NSFW/spoiler, and media-mode controls. Galleries are
expanded and posted as image sets.

**On-demand scraping.** `.scrape <url>` fetches a page, optionally narrowed by a
CSS selector, optionally cleaned up by an LLM before posting.

**Resume and cover-letter generation.** Reply to a job post the bot published
with `.resumebuild` and it scrapes the posting, tailors your LaTeX resume to it
via an LLM, renders and compiles it, and uploads the PDF. `.resumecoverbuild`
then writes a cover letter out of what the resume could *not* fit.

**Health reporting.** `.health` renders a dashboard of scrape latency, success
and error counts, per-ATS-platform results, browser-service state, watchdog
restarts, scheduler queue depth, and uptime.

---

## Requirements

| Thing | Why |
| --- | --- |
| **Python 3.14** | What CI is pinned to and what the suite is verified against. Set `PYTHON_EXE` to your 3.14 interpreter before using `run.bat` — its no-`PYTHON_EXE` fallback is currently `py -3.11` (and `py -3` in one branch), which is stale. See the note under [Running the bot](#running-the-bot). |
| **A real TeX distribution** | Resume commands only. The `pdflatex` pip package is a wrapper, not an engine — you need actual `pdflatex` and `kpsewhich` on `PATH`, plus the template's packages: `XCharter`, `geometry`, `enumitem`, `hyperref`, `titlesec`, `comment`, `glyphtounicode`. |
| **The `playwright` package** | Drives the browser for the Reddit fallback, ZipRecruiter, and some ATS pages. The pip package is all you need — do **not** run `playwright install`; see the row below. |
| **A system Chrome or Chromium** | Required for every browser-backed path, and not interchangeable with a Playwright download. `browser_service` has one launch site and it passes `executable_path=find_chrome()`, which probes system install locations only; `_do_start` logs "Chrome not found -- skipping Chrome layer" and disables the layer when that comes back empty. A real Chrome plus a persistent profile is preferred, for Reddit session reuse. |
| **A Discord application** with the **Message Content Intent** enabled | Text commands (`.job`, `.scrape`, …) are read from message content. Without the intent the bot sees empty content and logs a warning on every message. |

Bot permissions needed in each channel: **View Channel**, **Read Message
History**, **Send Messages**, **Attach Files** (PDF upload), **Embed Links**,
and **Manage Messages** if you want the command cheatsheet auto-pinned.

---

## Installation

```bash
pip install -U -r requirements.txt
```

That pulls `discord.py`, `requests`, `curl_cffi`, `pysocks`, `playwright`,
`beautifulsoup4`, `python-jobspy>=1.1.82`, `ftfy`, `google-genai`, `pycountry`,
and `pdflatex`.

The browser itself is a *system* install, not a Playwright download — see the
table above, and the [ARM64 note](#arm64--raspberry-pi), which says the same
thing:

```bash
sudo apt install google-chrome-stable    # Linux; or: sudo apt install chromium
```

On Windows, install Chrome normally; `find_chrome()` probes the standard
`Program Files` locations.

Optional semantic matching:

```bash
python -m pip install -r requirements-semantic.txt
```

This installs `sentence-transformers`, which drags in torch (~2 GB). That is why
it is not in `requirements.txt`. Without it the bot runs fine and silently falls
back to keyword-only matching — check with `job_service.semantic_plugin_available()`.
Install it into the **same interpreter the bot actually runs under**, which under
`run.bat` is whatever `PYTHON_EXE` / `py -3` resolves to, not necessarily a
`.venv`.

`requirements-lock.txt` holds a pinned resolution of the full set.

One-time Reddit session setup (optional, improves Reddit reliability a lot):

```bash
python setup_reddit_browser.py
```

It logs into Reddit in the bot's Chrome profile, optionally copying an existing
logged-in Chrome profile's session first. **Do not run it while the bot is
running** — both processes would own the same runtime profile.

---

## Running the bot

Direct:

```bash
python src/app.py
```

Supervised, any OS (recommended):

```bash
python run.py
```

`run.py` is the cross-platform launcher (Windows and Linux). Flags: `--forcerun`
(stop an existing instance first), `--skip-ats-sync`.

It is thin on purpose: single-instance enforcement already lives in
`src/app.py`'s `acquire_runtime_lock()` (portable — `O_CREAT|O_EXCL` plus a
liveness check on the recorded pid), and the restart policy and log rotation
belong to the supervisor. See [Linux deployment](#linux-deployment).

Supervised (Windows, legacy):

```bash
run.bat
```

`run.bat` is retained so existing Task Scheduler entries keep working; it and
`run.py` are interchangeable on Windows. It does considerably more than launch
Python:

- **Elevation guard.** It detects Session 0 or Administrator context up front and
  warns, because Chrome profile locks created by an elevated or Session-0 owner
  cannot be cleared from a normal user session and cause recurring profile-lock
  failures. The usual cause is a Task Scheduler entry with "Run with highest
  privileges."
- **Start lock.** A `.start.lock` directory prevents two launchers from racing.
- **Single-instance check.** It scans WMI for an existing `src\app.py` process
  and skips the start; if the running instance is more than 24 h old it does a
  graceful stop, waits 5 s, then force-kills survivors and restarts.
- **ATS list sync.** Runs `sync_ats_companies.py` first (sparse, depth-1 git
  fetch of the company-slug lists in `data/ats_companies`).
- **Log rotation.** Rotates `logs/bot_console.log` to `.log.1` past 5 MB, so the
  daily restarts cannot grow it without bound.

**Interpreter selection.** `run.bat` uses `%PYTHON_EXE%` when set. Its fallback
is inconsistent today: `py -3.11` for the ATS sync, the scheduler-mode launch,
and the log line, but `py -3` for the detached launch. CI and the test suite
target 3.14, so **set `PYTHON_EXE` explicitly** rather than relying on the
fallback.

Flags:

| Flag | Effect |
| --- | --- |
| `forcerun` | Remove a stale start lock, kill any existing bot (via `.bot.pid`, `.bot.lock`, then WMI), then start. Aborts if a survivor cannot be killed — usually an elevation problem. |
| `scheduler` | Run the bot in the foreground so Task Scheduler can enforce single-instance itself. |
| `noelevated` | Refuse to start when elevated or in Session 0. Also available as `REDDIT_BOT_REFUSE_ELEVATED=1`. |

### Linux deployment

Every OS-specific decision lives in `src/services/platform_support.py` — Chrome
discovery, browser user-data locations, the presented user-agent, process lookup
and termination, profile lock artifacts. No other module branches on the
platform, and `tests/test_platform_support.py` drives *both* branches explicitly,
so the Linux paths stay verified even when the suite runs on Windows.

```bash
# System packages. texlive is only needed for the resume commands; without it
# .resumebuild reports "LaTeX compile environment is not ready" and the rest of
# the bot is unaffected. XCharter lives in texlive-fonts-extra.
sudo apt install google-chrome-stable          # or chromium
sudo apt install texlive-latex-recommended texlive-fonts-extra texlive-xetex

sudo useradd -r -m -d /opt/discordbot botuser  # needs a real home; see below
python -m pip install -r requirements.txt

# No `playwright install` step: browser_service never opens a downloaded
# browser. Its only launch site passes executable_path=find_chrome(), and
# _do_start skips the whole Chrome layer when that returns None. The apt
# install above IS the browser step.

# Neither of these is in git. Without them the bot starts but silently does
# nothing useful: no companies to scrape, and no city/region matching.
python sync_ats_companies.py                   # clones the company-slug lists
python sync_geonames.py                        # pulls the GeoNames export, builds data/geo.db (~101 MB)

# Secrets. config.py exits if discordtoken is missing, so create this first.
install -m 600 -o botuser -g botuser /dev/null /opt/discordbot/.env
echo 'discordtoken=YOUR_TOKEN' | sudo tee -a /opt/discordbot/.env

python setup_reddit_browser.py                 # optional; needs a graphical session

sudo cp deploy/discordbot.service /etc/systemd/system/   # EDIT the paths in it first
sudo systemctl daemon-reload
sudo systemctl enable --now discordbot
```

Verify the two generated artifacts before enabling the service:

```bash
python scripts/build_geo_db.py --check
```

### Docker

```bash
docker compose build && docker compose up -d
```

`Dockerfile` + `docker-compose.yml` carry the three things the bot shells out
to — Chromium, a LaTeX engine, and git — and bind-mount the repo at `/app` so
every writable path (state, logs, Chrome profile, job archives) stays on the
host. `.env` is mounted, never baked in. Full notes, including the two
generated artifacts you must build on first run and why the entrypoint clears
the runtime lock, are in [deploy/DOCKER.md](deploy/DOCKER.md).

`deploy/discordbot.service` covers what run.bat was doing as a supervisor:
`Restart=always` plus `RuntimeMaxSec=86400` gives the same daily restart, and
journald replaces the manual `bot_console.log` rotation.

Optional env overrides, honoured on both OSes:

| Variable | Effect |
| --- | --- |
| `CHROME_EXECUTABLE` | Chrome/Chromium binary; probed before every built-in location. |
| `CHROME_USER_DATA_DIR` | Browser user-data dir to harvest the Reddit session from. |
| `CHROME_NO_SANDBOX` | Set to `1` **only** in containers — Chrome's setuid sandbox cannot initialise as an unprivileged container PID 1. Dropping it on a normal host is a real security regression. |
| `PYTHON_EXE` | Interpreter `run.py` / `run.bat` launch the bot with. |
| `BOT_WORKER_SCALE` | Override the auto-detected hardware scale factor (see below). `>1` scales up. |
| `BOT_MAX_WORKERS` | Hard ceiling on every thread pool, applied after scaling. |
| `JBA_ARCHIVE_GIT_COMMIT` | Set to `1` to let the bot `git commit` its monthly job archive into the working tree. **Off by default** — a service account with a writable `.git` is a liability, and a local auto-commit collides with pull-based deploys. |
| `JBA_GIT_AUTHOR_NAME` / `JBA_GIT_AUTHOR_EMAIL` | Committer identity for the above. Passed with `git -c`, because a service account normally has no `~/.gitconfig` and `git commit` would exit 128. |

**Two things git does not carry**, both of which fail quietly rather than
loudly if you skip them:

- `data/ats_companies/` — gitignored (it has its own history). `sync_ats_companies.py`
  clones it on first run. Without it `load_company_lists()` returns nothing and
  ATS scraping finds zero jobs, which looks identical to "nothing matched".
- `data/geo.db` — ~101 MB, gitignored, built by `scripts/build_geo_db.py` from
  the GeoNames sources in `data/geonames_raw/`. Those are gitignored too, so on
  a fresh clone run `sync_geonames.py`, which fetches them and rebuilds in one
  step; `build_geo_db.py` alone has nothing to read. Without it city and
  region matching is disabled (the bot logs this and falls back to country-only
  matching rather than failing).

### Hardware scaling

Pool sizes were tuned on a 12-core / 16 GB machine. `src/services/capacity.py`
scales them to whatever the host actually is, so the same checkout runs on a
1-core VPS and a 32-core server without editing constants.

Detection is **container-aware**: `os.cpu_count()` reports the *host's* cores
inside a cgroup-limited container — precisely where scaling down matters most —
so cgroup v2/v1 CPU quota and memory limits are read first. The scale factor is
bound by whichever of CPU or memory is scarcer.

Scaling is asymmetric by design. Downward always applies. **Upward is opt-in per
call site**, because some pool sizes are politeness ceilings against scraping
targets rather than local resource limits — a bigger box must not mean more load
on Greenhouse or Workday. Pools that only cost local resources (the work
scheduler, the scrape fan-out) do grow.

Low-memory hosts also disable semantic matching automatically: the torch runtime
is the largest fixed allocation in the process, so below ~2 GB the bot falls back
to keyword matching rather than risking the OOM killer. The dedup link cache
(held per watched channel) shrinks on small hosts too.

**Timeouts scale in the opposite direction**, via `capacity.timeout()`. This is
the half that actually breaks on slow hardware: shrinking a pool makes each item
take *longer*, so a fixed budget tuned on a fast box becomes marginal. The
scrape-subprocess budget, its paired cross-channel cache TTL, the LaTeX compile
budgets, and Chromium cold-start all stretch (bounded at 3x) and never tighten on
fast hosts. Per-request *network* timeouts deliberately do not scale — a slow
local CPU does not make a remote server answer any slower, and stretching those
would only delay detection of a genuinely dead endpoint.

### ARM64 / Raspberry Pi

Works, with caveats. Use a 64-bit OS and an 8 GB board; 4 GB is workable, 2 GB is
not.

`playwright install chromium` is the wrong instruction everywhere, not just here
— `browser_service` launches via `executable_path` and never opens a downloaded
browser — but ARM adds two more reasons it cannot work: Google ships no
`google-chrome-stable` for ARM64 Linux, and Playwright's bundled Chromium
download does not reliably cover `linux-arm64`. Install the system browser:

```bash
sudo apt install chromium-browser
```

Two knobs are worth setting by hand. The scale factor counts *cores, not
per-core performance*, so a 4-core Cortex-A76 scores 0.33 while its real
throughput is nearer 0.15 — and torch on aarch64 has no AVX, making semantic
matching slower than the keyword path it is meant to improve on:

```bash
BOT_WORKER_SCALE=0.15
```

Set `semantic_enabled = false` in `settings.toml`. Also boot from SSD/NVMe rather
than microSD (the bot writes state JSON on every send plus dedup files and
sqlite — sustained small random writes), and use active cooling, since thermal
throttling compounds the timeout pressure above.

`/health` prints the detected values (`Hardware: cpu=… mem=… scale=…`) — the
quickest way to confirm a new host sized itself the way you expected.

`setup_reddit_browser.py` needs a real display and refuses early when
`DISPLAY`/`WAYLAND_DISPLAY` are unset — on a headless server, run it on a desktop
and copy the resulting `chrome_profile/` across.

---

## Configuration

Configuration is split deliberately across two files:

- **`.env`** (repo root) — **secrets and machine-specific paths only.** Values
  are also pushed into `os.environ` via `setdefault` at load, so service modules
  can reach them with plain `os.getenv`. Real environment variables win over
  `.env` entries.
- **`settings.toml`** (repo root, optional) — **behavioural defaults.** Never
  read for secrets. If the file is absent, every default below applies.

Anything a user can change from a Discord settings panel is stored per channel
in `.bot_state.json`, not in either config file. `settings.toml` only seeds the
defaults new channels start from.

### `.env` — secrets and paths

Several keys accept multiple spellings; the first one found wins, in the order
listed.

**Required**

| Key | Meaning |
| --- | --- |
| `discordtoken` | Discord bot token. The bot exits with code 1 if this is missing. |

**Identity / ownership**

| Key | Meaning |
| --- | --- |
| `MAIN_USER_ID` / `mainUserId` | Owner's Discord user id. |
| `MAIN_USER_PROFILE` / `mainUserProfile` | Owner's profile folder name under `src/services/resumes/resumes_cache/` (e.g. `xboxsignout._`). If unset and exactly one seeded folder exists, it is inferred. Falls back to the guild owner id. |

**LLM providers.** Resume generation walks a fallback chain in the order
`gemini → gemini-flash → openrouter → groq`, skipping providers with no key and
cooling a provider down for 120 s after it rate-limits.

| Key | Meaning |
| --- | --- |
| `GEMINI_API_KEY` / `GOOGLE_AI_STUDIO_API_KEY` / `geminiAPI` | Gemini key. Gemini is the only provider that supports the explicit context cache. |
| `GEMINI_MODEL` | Default `gemini-2.5-flash`. |
| `GEMINI_RESUME_CACHE_TTL_SECONDS` | Default `86400`. |
| `OPENROUTER_API_KEY` / `openRouter` | OpenRouter key. |
| `OPENROUTER_MODEL`, `OPENROUTER_RESUME_MODEL` | Model overrides for general and resume use. |
| `GROQ_API_KEY` / `groqAPI` | Groq key. |
| `GROQ_MODEL`, `GROQ_RESUME_MODEL` | Model overrides. |

Per-provider ceilings are compiled into `PROVIDER_CAPABILITIES` in
[resume.py:96](src/services/resumes/resume.py:96) — prompt-character limits of
800 k (Gemini), 120 k (OpenRouter), 24 k (Groq), with request timeouts of 60/45/30 s.
Prompts too long for a provider are truncated with an explicit marker rather
than dropping the provider. Per-model overrides keyed by model-id prefix let you
swap the model in a slot and re-tune its limits.

**Resume / LaTeX**

| Key | Meaning |
| --- | --- |
| `RESUME_MAX_PAGES` | Page cap for generated resumes. Default `1`, minimum 1. |
| `LATEX_ENGINE_TIMEOUT_SECONDS` | Per-engine compile timeout. Default `75`, clamped to 30–180. |
| `LATEX_TOTAL_TIMEOUT_SECONDS` | Total compile budget across engine fallback and rerun passes. Default `180`, clamped to 60–… |

**Job scraping**

| Key | Meaning |
| --- | --- |
| `JOBSPY_PYTHON_EXE` | Run JobSpy under a different interpreter. Install the same `python-jobspy` version there — the bot probes the target interpreter for JobSpy and its supported site list at runtime. |
| `JOBSPY_PROXIES` | Proxy pool, comma/semicolon/newline separated, rotated per request attempt. |
| `INDEED_API_KEY`, `INDEED_GRAPHQL_API_KEY`, `INDEED_GRAPHQL_BEARER` | Optional credentials for the direct Indeed GraphQL path used when scraping a single posting for the resume pipeline. |

**Reddit**

| Key | Meaning |
| --- | --- |
| `REDDIT_PROXIES` | Proxy pool, same format and rotation as above. |
| `REDDIT_CHROME_PROFILE` | Source Chrome profile directory (default `chrome_profile/`). |
| `REDDIT_CHROME_RUNTIME_PROFILE` | Runtime copy the browser actually drives (default `chrome_profile_runtime/`). |
| `REDDIT_PLAYWRIGHT_CONCURRENCY` | Concurrent Playwright Reddit fetches. Default `2`. |
| `REDDIT_PLAYWRIGHT_GATE_WAIT_SECONDS` | Seconds to wait for a gate slot. Default `10`. |
| `REDDIT_PLAYWRIGHT_BREAKER_THRESHOLD` | Consecutive failures before the circuit breaker opens. Default `3`. |
| `REDDIT_PLAYWRIGHT_BREAKER_COOLDOWN_SECONDS` | Breaker cooldown. Default `120`. |

**Discord slash-command sync**

| Key | Meaning |
| --- | --- |
| `DISCORD_SYNC_GUILD_ID` / `DISCORD_GUILD_ID` | Guild targeted by the one-time forced sync. |
| `DISCORD_FORCE_SYNC_ONCE=1` | On next start, clear that guild's guild-scoped commands and republish globals. A `.force_slash_sync_once` marker file does the same and is deleted after a successful run. |

### `settings.toml` — behaviour

```toml
[semantic]
enabled = true
model = "sentence-transformers/all-MiniLM-L6-v2"
threshold = 0.30
match_target = "description"   # "description" or "title"
description_char_limit = 2200

[dedup]
months_threshold = 2
max_fifo_files = 6
max_entries_per_file = 500

[network]
http_timeout_seconds = 20
subprocess_scrape_timeout_seconds = 90
site_concurrency_limit = 2
site_semaphore_timeout_seconds = 120

[watcher]
dedupe_seconds = 120
discord_history_check_limit = 5

[ats]
bamboohr_enabled = true

[scrape_defaults]
max_items = 20
timeout_seconds = 20
use_ai_cleanup = true

[job_defaults]
keywords = "python developer"
location = "Canada"
radius_miles = 25
hours_old = 72
results_wanted = 10
refresh_seconds = 900
country_indeed = "AUTO"
allow_north_america = false

[reddit_defaults]
subreddit = "wallpapers"
sort = "new"
time_filter = "day"
limit = 10
refresh_seconds = 300
```

Notes on individual keys:

- `semantic.match_target = "title"` scores the job title (chunked on `-`, `–`,
  `/` so long compound titles still match) instead of the description body.
  Description matching is preferred by default because title-only matching
  produces false positives.
- `network.site_concurrency_limit` is enforced per site with a semaphore, so one
  slow board cannot starve the others.
- `ats.bamboohr_enabled` is **on**. It was off because BambooHR was believed to
  be Cloudflare-gated and to need a headless browser. Re-measured 2026-09-05
  against 60 random non-dead boards with plain `requests`: 58 answered 200, two
  401, **zero** challenge pages, and 43 of them carried 347 live postings. Every
  response is served through Cloudflare's CDN — which is what the original claim
  saw — but sitting behind the CDN is not being challenged by it. BambooHR is
  the largest fleet of the sixteen (21,291 boards, 14,169 not dead-marked), so
  this was the biggest single source of coverage left switched off. Set it to
  `false` in `settings.toml` to restore the old behaviour.

Per-channel job settings additionally carry `sites`, `role_filters`,
`exclusion_terms`, `semantic_threshold` (0.30) and `ats_semantic_threshold`
(0.20), and `enabled`. Reddit settings carry `subreddits`, `flair_tags`,
`include_nsfw`, `include_spoiler`, `media_mode`, and `enabled`. Defaults for all
of these live in [store.py:15](src/state/store.py:15).

---

## Commands

Text commands use the `.` prefix. Every command also has `/`-style aliases, and
a subset is registered as real Discord slash commands in
[app.py:179](src/app.py:179). The source of truth for aliases is
`_COMMAND_ALIASES` in [handlers.py:144](src/commands/handlers.py:144); run `.cmd`
in Discord for the live cheatsheet, which the bot also keeps pinned per channel.

### General

| Command | Aliases | Description |
| --- | --- | --- |
| `.cmd` | `/commands` | Command cheatsheet. Comes in a general and a job-specific variant. |
| `.hi` | `/hello`, `/hello2` | Liveness check. |
| `.more` | `/continue` | Send the next chunk of pending long output. Long results are chunked to ~1900 characters and paged. |
| `.scrape <url>` | `/scrape` | One-off scrape. `.scrape <url> \| <css selector>` narrows to a selector. Job-board URLs are routed through the job scraper instead of the generic one. |
| `.cfg` / `.scrapecfg` | `/settings` | Scrape settings panel (max items, timeout, AI cleanup). |
| `.st` / `.watch` | `/status`, `/watcherstatus` | Watcher status for this channel. |
| `.health` | `/whealth` | Scrape health dashboard. `.health all` covers every channel. |

### Job watching

| Command | Aliases | Description |
| --- | --- | --- |
| `.job` / `.jobs` | `/jobsettings`, `/jobsinit` | Job watcher settings panel — keywords, location, radius, sites, hours-old, results wanted, refresh interval, role filters, exclusion terms, region policy, semantic thresholds, enable/disable. |
| `.jobtest` | `/jobpipelinetest` | Run the whole pipeline once, in the foreground, and report per-stage metrics. The way to debug a channel that stopped producing results. |
| `.bestjobs` / `.best` | `/bestmatches` | Best archive matches for a resume profile: `.best [day\|week] [count] [profile] [--fast]` (defaults day / 10 / the channel owner's own profile, tokens in any order; `--fast` skips the description fetch). `profile` accepts an @mention or a cache folder name. Ranks the jobs already logged in `data/jba/jobs` against the profile — see [Best-match ranking](#best-match-ranking). Only the guild owner may name another profile. |

### Reddit watching

| Command | Aliases | Description |
| --- | --- | --- |
| `.reddit` / `.rset` | `/redditsettings` | Reddit watcher settings panel. |
| `.rclear` | `/clearredditseen` | Clear this channel's seen-post IDs. |
| `.rreset` | `/resetredditseen` | Alias of the above. |

### Resume (owner-only)

These require a guild text channel — not a DM — and the caller must be the guild
owner. Each must be sent as a **reply to a job post message**, or given an
explicit `message_id` when invoked as a slash command.

| Command | Aliases | Description |
| --- | --- | --- |
| `.resumebuild` | `/resumebuild`, `/resume`, `/res` | Tailored resume PDF for the replied-to listing. Flags: `--aggressive`, `--strongaggressive` (progressively more willing to rewrite rather than re-select). |
| `.resumecoverbuild` | `/coverbuild`, `/cover` | Cover letter built from what the resume left out. |
| `.resumecheck` | `/resumecheck` | Compile the current cached template as a baseline sanity check, no LLM involved. |

---

## Architecture

```
                    Discord gateway
                          │
                  ┌───────┴───────┐
            on_message        on_interaction
                  │                │
            CommandRouter ──── ui/views.py (modals, buttons)
                  │
       ┌──────────┼─────────────────────────┐
       │          │                         │
  WatcherManager  │                  PriorityWorkScheduler
   ├ job loops    │                   (shared thread pool,
   ├ reddit loops │                    interactive > background,
   ├ ATS loop     │                    shortest-job-first)
   └ watchdog     │                         │
                  │                         │
            RuntimeStore              services/*
          (.bot_state.json)      job · ats · browser · reddit
                                 scrape · rss · health · jba
                                          resumes/
```

### Startup and process lifecycle

[`src/app.py`](src/app.py) is the entry point. In order:

1. `load_config()` reads `.env` and `settings.toml` into an `AppConfig`.
2. **Runtime lock.** `acquire_runtime_lock` does an `O_CREAT|O_EXCL` create of
   `.bot.lock` containing `{"pid": …}`. If the file exists, the owner PID is
   checked with `os.kill(pid, 0)`; a dead owner's lock is removed and the
   acquisition retried, a live owner causes a clean `SystemExit(0)`. This is
   what stops two bots from double-posting.
3. Health tracker is created and wired into the browser service as a hook; the
   browser service starts.
4. `configure_job_service(config)` and `init_store_defaults(config)` push
   `settings.toml` values into the service and store module globals.
5. `RuntimeStore` loads `.bot_state.json`.
6. A **single** `PriorityWorkScheduler` is created and handed to both the
   `WatcherManager` and the `CommandRouter` — see below for why that matters.
7. Slash commands are registered and events bound.
8. `client.close` is wrapped in a graceful drain: up to 120 s waiting for
   in-flight scrapes, then up to 30 s for in-flight watcher sends, then the real
   close. Both drains log whether they completed or timed out.
9. `.bot.pid` is written; on exit the browser service stops, the scheduler is
   shut down with `wait=True` as a bounded safety net, and the PID file and lock
   are released.

On `on_ready` the bot restores every channel whose watcher was `enabled` in
persisted state, starts the watchdog, syncs slash commands (with the optional
one-time forced guild refresh), then scans channel history to learn Discord
usernames and migrate legacy numeric-ID resume profile folders to username keys.

### The priority work scheduler

[`services/priority_scheduler.py`](src/services/priority_scheduler.py) is a
thread pool that dispatches by `(tier, cost)` rather than FIFO.

- **Two tiers.** `INTERACTIVE` (0) is work a user is actively waiting on — a
  `.resumebuild` compile, a `.jobtest` run. `BACKGROUND` (1) is watcher scraping,
  ATS scraping, semantic filtering. Interactive always dispatches ahead of
  background.
- **Shortest-job-first within a tier.** Every submission carries a label (see
  [`scheduler_labels.py`](src/services/scheduler_labels.py) for the full
  vocabulary: `job_scrape`, `semantic_filter`, `ats_scrape`, `reddit_scrape`,
  `resume_rewrite`, `resume_compile_latex`, …). The scheduler keeps a rolling
  48-hour duration history per label and uses the **median** as the cost
  estimate, so ordering adapts to the machine it is on instead of a hardcoded
  guess. Unmeasured labels default to a cost of 1.0.
- **Aging.** A sweep every second promotes any task that has waited more than
  20 s, so background work cannot be starved indefinitely by a steady interactive
  stream. Promotions are counted and surfaced in `.health`.
- **Size.** `max(4, os.cpu_count())` worker threads plus one aging thread, all
  daemon.

The single-instance rule matters: arbitration only works if everything competing
for the same CPU goes through the same queue. That is why `app.py` constructs one
scheduler and injects it into both the watcher manager and the command router.

### Runtime state

[`state/store.py`](src/state/store.py) is a JSON-backed store persisted to
`.bot_state.json`. It holds, keyed by channel id: mode, scrape settings, job
settings, reddit settings, and the message id of the pinned cheatsheet per sheet
kind. `init_store_defaults` overwrites the module-level default dicts from
`AppConfig` at startup, so a channel created after a `settings.toml` edit picks
up the new defaults while existing channels keep their saved values.

### Watcher manager

[`watchers/manager.py`](src/watchers/manager.py) owns every background loop.

- **`_run_job_watcher(channel_id)`** — per channel: scrape, filter, dedupe,
  format, send, sleep `refresh_seconds`.
- **`_run_reddit_watcher(channel_id)`** — per channel, same shape, with gallery
  expansion.
- **`_run_ats_scrape_loop()`** — one process-wide loop, not per channel, since
  ATS results are shared across channels.
- **`_run_watchdog(interval_seconds=30)`** — every 30 s, checks that each loop
  that should be alive still is, and restarts dead ones. Restarts are recorded
  in the health tracker so `.health` shows whether a channel is quietly
  crash-looping.

Supporting machinery:

- **Work guard.** `work_guard()` and `_tracked_to_thread()` count in-flight
  blocking work so `drain_active_work()` can block shutdown until scrapes finish.
  `_enter_send_cycle` / `_exit_send_cycle` do the same for Discord sends via
  `drain_active_sends()`.
- **Per-channel process locks.** `acquire_channel_process` /
  `release_channel_process` stop two loops of the same kind from running for one
  channel.
- **Per-channel send locks.** An `asyncio.Lock` per (channel, watcher type)
  serialises sends so the duplicate check and the send cannot interleave.
- **Deleted-channel purge.** `_channel_exists` plus `_purge_deleted_job_channel`
  clean up settings and dedup state for channels that no longer exist rather
  than looping on 404s forever.

### Job scraping pipeline

[`services/job_service.py`](src/services/job_service.py) is the largest module.
A single scrape pass does roughly this:

1. **Resolve sites.** `normalize_requested_sites` maps the channel's `sites`
   setting (including `all`) against what the target interpreter's installed
   JobSpy actually supports, discovered at runtime by
   `jobspy_runtime_metadata()`. Sites in `CUSTOM_SCRAPER_SITES` — Glassdoor,
   ZipRecruiter, and the five ATS platforms — are handled by first-party code
   instead of JobSpy.
2. **Build per-site scrape code and run it out of process.**
   `build_site_scrape_code` emits a small program that
   `run_site_scrape_subprocess` executes under `JOBSPY_PYTHON_EXE` (or
   `sys.executable`) with a `subprocess_scrape_timeout_seconds` budget. Running
   JobSpy out of process means a segfault or a hung HTTP client in a scraper
   cannot take the bot down.
3. **Per-site concurrency gate.** `_site_scrape_slot` wraps each site in a
   semaphore limited to `site_concurrency_limit`, with a
   `site_semaphore_timeout_seconds` acquisition timeout.
4. **First-party scrapers.** `scrape_glassdoor_postings` resolves locations via
   the geo DB, reads card ages off the listing page, and can open a detail page
   for a precise `addressLocality`/`addressRegion`. `scrape_ziprecruiter_postings`
   goes through Playwright behind the browser gate.
5. **Short-TTL cache.** `scrape_job_postings` memoises identical parameter sets
   for `JOB_SCRAPE_CACHE_TTL_SECONDS` (55 s) so several channels watching the
   same query in the same tick share one network pass.
6. **Normalisation.** `shape_job_item` produces a uniform row.
   `fix_text_encoding` runs mojibake repair through `ftfy`.
   `canonicalize_job_link` strips `utm_*` and other tracking parameters so the
   same job from two sources dedupes to one link.
   `normalize_canadian_province` and `infer_job_region` canonicalise locations.
7. **Cross-source dedupe.** `dedupe_job_rows` merges rows sharing any of the keys
   from `job_row_dedupe_keys`, and `collect_job_sites` records every source a
   merged row came from so the posted message can say "Indeed · LinkedIn".
8. **Filtering.** Region filter (`filter_rows_by_region`, gated on
   `allow_north_america`), freshness (`_posting_age_ok` against `hours_old`),
   `matches_role_filters` on the title, `matches_exclusion_terms` across the row,
   and keyword variants from `build_keyword_variants`.
9. **Semantic filtering** (optional). If `sentence-transformers` is installed and
   `[semantic] enabled` is true, listings are embedded with the configured model
   and cosine-compared to a search text built from keywords, location, and role
   filters. The description body is preferred; when absent, the title is chunked
   on common delimiters (each chunk keeping company and location context) and
   the maximum score is taken. Encoding is batched in one pass per listing set.
   If the model fails to load or scoring raises, the function returns 1.0 — it
   fails **open**, so a broken semantic layer never silently suppresses every
   job.
10. **Dedupe against history**, format, and send (see [Deduplication](#deduplication)).

### ATS scraping

[`services/ats_service.py`](src/services/ats_service.py) scrapes sixteen ATS
platforms directly rather than through an aggregator, which gets fresher results
and full descriptions. In registration order: Greenhouse, Lever, Ashby, Workday,
iCIMS, BambooHR, Workable, Breezy, SmartRecruiters, Recruitee, TeamTailor,
Rippling, JazzHR, Jobvite, ApplicantPro and Paylocity. Every one of them is
harvested by `scripts/harvest_ats.py`; nine (Workable, Breezy, JazzHR,
Recruitee, TeamTailor, ApplicantPro, Rippling, SmartRecruiters, Jobvite) exist
only in the local harvest and have no upstream company list.

- **Company slug lists** live in `data/ats_companies/*.json`, one file per
  platform, synced by `sync_ats_companies.py` from a separate git repo using a
  sparse, blob-filtered, depth-1 fetch.
- **Dead-slug caching.** A slug that 404s or 410s is written to
  `data/dead_slugs/<platform>.json` with a timestamp and skipped on future runs;
  entries expire so a company that comes back is eventually retried.
  `clear_dead_slugs()` forces a full re-probe.
- **Per-platform strategies.** Greenhouse, Lever, Workable, Breezy, Recruitee,
  TeamTailor and Rippling have clean JSON boards. Ashby uses its public posting
  API (the GraphQL endpoint was a dead end -- asking for fields errors the whole
  query, and it does not 404 properly, which is what dead-marking needs).
  Workday needs its slug parsed into tenant/host/path parts and a per-job detail
  fetch to recover a posting date. iCIMS is walked via sitemap XML, fetched with
  `in_iframe=1` because the plain job URL stopped serving JSON-LD. JazzHR,
  Jobvite, ApplicantPro and Paylocity have no API and are read from the rendered
  page.
- **Rate limiting is per platform and measured, not guessed.** `PLATFORM_WORKERS`
  is a politeness ceiling rather than a resource limit, so it never scales up on
  bigger hardware. Workable and Recruitee sit at 4 workers because they meter by
  quota: a workable validation pass once came back 91% HTTP 429. A 403 or 429 is
  never treated as a dead board -- the host declining to answer says nothing
  about whether the company exists -- so a metered platform returns nothing
  rather than condemning its own fleet.
- **Location matching.** `_matches_location` sits on top of a geo lookup
  (`data/geo_lookup.json.gz`, lazily loaded) that resolves cities, subdivision
  codes and names, and countries, with remote-work detection via several regex
  patterns so "Remote (Canada)", "Toronto, ON – Remote", and "Remote" all behave.
- Results feed into the same `shape_job_item` / dedupe path as everything else.

### The JBA subsystem

[`services/jba/`](src/services/jba) is an isolated, lightly adapted copy of the
[job-board-aggregator](https://github.com/Feashliaa/job-board-aggregator)
scripts, kept separate so upstream changes stay easy to merge.

- **`geo_db.py`** — read-only SQLite access to `data/geo.db` plus a cache of
  Glassdoor location-ID lookups, so the autocomplete endpoint is hit once per
  city rather than once per scrape.
- **`geolocation.py`** — parses free-text location strings out of postings and
  resolves them to coordinates.
- **`merge_data.py`** — date-based job log storage. The current week lives in
  `data/jba/jobs/jobs.db` for fast lookups; at a week boundary the previous
  week is exported to a weekly zip and purged from the DB; at a month boundary
  the weekly zips are consolidated into `YYYY-MM.zip` holding per-day JSON plus
  `seen_urls.json`, and the result is git-committed.
- **`scraper.py`** — the standalone aggregator with its own fetchers per
  platform, recruiter-company filtering, job-tier classification, and user-agent
  rotation.

### Best-match ranking

[`services/job_match.py`](src/services/job_match.py) answers "what were the
best-fitting jobs for my profile today, or this week?" over the archive
`merge_data.py` already keeps. It is what `.bestjobs` / `.best` runs.

The archive is large — thousands of records a week, and the source data is
mostly a title, a company and a location, since fewer than 2% of stored records
carry a description. So the ranking runs in three stages:

1. **Lexical prefilter** over every record in the window. The profile's
   `baseinfo.txt` is parsed into a `ProfileSignal` — skill anchors from the
   `== SKILL ANCHORS ==` section, role terms from entry headers and their
   `[category]` tags, the cities the candidate has actually worked in, and
   whether they are still a student (an in-progress degree end-year). Each job
   gets a weighted blend of `skills` / `role` / `level` / `location`, all in
   0..1. This is regex over short strings — a week costs about a tenth of a
   second — and exists only to cut the window down to `LLM_CANDIDATES` (60).
2. **Description enrichment** for the strongest `ENRICH_CANDIDATES` (18) of
   those. Without it the model is scoring job titles, so this fetches each
   posting's real text through the same ladder the resume commands use
   (`listing.scrape_job_posting`: per-platform APIs first). Concurrent, with a
   per-fetch timeout and a hard wall-clock budget; whatever has come back when
   the budget expires is what the judge gets, and a posting that 404s or blocks
   the client simply keeps its title. The browser rung is disabled for this
   pass — there is one shared browser context and the Reddit watcher needs it,
   so a bulk fetch must not queue up behind it.
3. **One batched LLM call** over the whole shortlist, through the same provider
   fallback chain as the resume system
   (`listing.generate_validated_with_providers`) but with
   `lowest_priority_first`, which reverses that chain. Ranking an archive is
   bulk work nobody is blocked on the way they are blocked on `.resumebuild`,
   so it deliberately spends the last-resort provider's quota rather than the
   primary one's; the chain still falls back upward if that provider is
   unconfigured or failing. The model
   returns a 0–100 score and a one-line reason per posting, and that is what
   the user sees. This is where the actual judgement happens: the prefilter
   cannot tell that "Avionics Firmware Co-op" suits an embedded student who
   has never written the word "avionics". Enriched and title-only postings are
   judged together, and the prompt says so, so a posting is never ranked down
   merely for having less text attached.

Enrichment roughly doubles the command's latency (a day window goes from ~30s
to ~70s) in exchange for markedly sharper scores — with descriptions the model
separates a genuine match from a title that merely sounds right, instead of
returning a flat band of 85s. `--fast` skips it.

Jobs are deduped on the way in by two rules: the same URL seen on several days
(a posting is re-logged every day it stays live), and the same title+company
under different URLs (boards mint one listing per city).

**Fail-open.** If no provider is configured or every one fails, the prefilter
ordering is returned and the report says `keyword ranking only (<reason>)`
rather than presenting weaker scores as if a model had ranked them.

This replaced an earlier local `sentence-transformers` pass. Embedding a week
of postings took minutes and more RSS than the rest of the bot combined, for a
weaker signal than one cheap model call.

### Browser service

[`services/browser_service.py`](src/services/browser_service.py) owns a single
persistent Playwright/Chrome context and hands out `fetch_json` / `fetch_html`.

- **Two profiles.** A source profile (`chrome_profile/`) holds the real logged-in
  session; a runtime profile (`chrome_profile_runtime/`) is what the browser
  actually drives. `_sync_profile_snapshot` copies `Cookies`, `Preferences`,
  `Login Data`, and the `Network`, `Local Storage`, `Session Storage`, and
  `IndexedDB` directories across. This keeps the browser from corrupting the
  profile you logged in with.
- **Session harvesting.** `_find_chrome_profile_with_reddit_session` can locate
  an existing Chrome profile that already has a valid Reddit session and harvest
  it, converting Chrome's epoch offsets to check cookie expiry first.
- **Lock clearing.** `_clear_profile_locks` and `_kill_chrome_using_profile`
  remove stale locks and kill the owning Chrome, checking Windows session IDs so
  it does not try (and fail) to kill a process in another session.
- **Session upkeep.** Validity is re-checked every 1800 s and re-synced every
  600 s; a 120 s cooldown prevents start thrashing.
- **Priority dispatch gate.** `_PriorityDispatchGate` bounds concurrent browser
  fetches and lets priority (interactive) callers jump the queue, with
  configurable queue-wait and dispatch-timeout padding.
- Every state change is emitted through a health hook, which is how `.health`
  reports browser readiness.

### Reddit service

[`services/reddit_service.py`](src/services/reddit_service.py) fetches subreddit
media through a ladder of increasingly expensive strategies, because Reddit
aggressively blocks plain requests.

1. **`curl_cffi` with browser impersonation** — a TLS fingerprint chosen from a
   rotating set, with a matching User-Agent and headers, optionally through a
   rotating proxy from `REDDIT_PROXIES` (8 s probe timeout per proxy).
2. **RSS/Atom** — `_scrape_via_rss` parses the subreddit feed, with
   `_supplement_rss_galleries` filling in gallery images the feed omits.
3. **Playwright** — the browser service, behind a concurrency gate and a
   **circuit breaker**: three consecutive failures open it for 120 s, so a
   Reddit-side block does not turn into a tight loop of expensive browser
   launches.

Rate-limit handling is shared process-wide — a `Retry-After` seen by one
subreddit's fetch backs off the others. Gallery extraction reads both
`media_metadata` and preview images, normalising URLs. Seen state is stored per
channel as both post IDs and normalised URLs (URLs catch crossposts and reposts
of the same image), capped at 2000 entries.

### Generic scraping

[`services/scrape_service.py`](src/services/scrape_service.py) handles `.scrape`.
It parses `url | selector`, fetches with a timeout, extracts items, optionally
runs an LLM cleanup pass through OpenRouter when `use_ai_cleanup` is on, formats
the result, and chunks it to Discord's message limit (`chunk_text_for_discord`,
~1900 chars) for the `.more` pager. Job-board URLs are detected by
`JOB_BOARD_HOST_PATTERNS` and routed to the job scraper instead.

[`services/rss_service.py`](src/services/rss_service.py) is a small Atom parser
with an HTML link extractor, used by the Reddit RSS rung.

### Health tracking

[`services/health.py`](src/services/health.py) accumulates:

- **Per-channel job health** — last success, last error, rolling event window
  with durations, success/error counts.
- **ATS health** — per-platform result counts and whether the shared loop is
  alive.
- **Browser health** — readiness, session validity, event log.
- **Watchdog health** — every restart, with target and timestamp.
- **Scheduler stats** — queue depth by tier, active workers, completed count,
  aging promotions.
- **Uptime**, formatted human-readably.

`build_channel_health_embed` and `build_all_health_embed` render these into
Discord embeds for `.health` and `.health all`.

### Resume system

The most involved subsystem. The design principle is **"the LLM proposes, Python
disposes"**: the model returns structured JSON decisions, and Python
deterministically renders the LaTeX. This is what keeps output compilable and
prevents the model from inventing experience.

**Flow**

1. **Authorization** — guild channel only, caller must equal `guild.owner_id`.
2. **Job context** ([`listing.py`](src/services/resumes/listing.py)) — parse the
   replied-to message for title and URL, including an optional `Apply:` line.
3. **Posting scrape** — a ladder of source-specific extractors before any
   generic path: Indeed GraphQL, LinkedIn guest API, Greenhouse API, Lever API,
   Ashby API, Workday API, iCIMS LD-JSON, then a `job_service` fallback, then a
   browser-service fetch. Each rung's outcome (rung name, elapsed, error) is
   appended to a scrape telemetry log so degradation is visible after the fact.
4. **Profile resolution** ([`resume.py`](src/services/resumes/resume.py)) —
   profile folders live at `src/services/resumes/resumes_cache/<profile_key>/`
   and are scaffolded with `baseinfo.txt`, `instructions.txt`, `template.tex`,
   and `template.log` when missing. Anything else in a profile folder is purged.
   New profiles are seeded from owner or global seed content. Legacy numeric-ID
   folders are migrated to username keys at startup.
5. **Gemini explicit cache** ([`cache.py`](src/services/resumes/cache.py)) —
   stable candidate background from `resumes/*.md|*.txt` is uploaded once as a
   Gemini explicit cache; the local mapping lives in
   `.resume_cache/resume_explicit_cache.json` and is only rebuilt when content,
   model, or TTL changes. **This is not the same thing as the local resume
   folder** — the folder is the source, the cache is a remote object.
6. **Structured tailoring** ([`structured.py`](src/services/resumes/structured.py),
   ~4000 lines) — the core. `parse_template_catalog` reads `template.tex` into a
   `TemplateCatalog` of tagged entries (`% [category]` markers);
   `parse_baseinfo_blocks` and `parse_skill_anchors` read the profile facts;
   `build_structured_prompt` asks the model for a ranking, exclusions,
   inclusions, and tailored bullets **as JSON**. Response handling includes
   `extract_json_object`, `_salvage_truncated_json`, and a repair-state machine
   for models that truncate mid-object.
7. **Validation and audits** — `validate_tailored_bullet` rejects bullets that
   are not grounded in baseinfo. A **domain-fit audit** gates specialist-tagged
   entries unless the job description actually names their domain (both via a
   curated term list and via an LLM audit covering domains the profile never
   configured). A **grounding audit** re-checks claims against source facts.
   `enforce_cross_bullet_consistency` catches contradictory numbers across
   bullets via Jaccard and number-set comparison.
8. **Deterministic render** — `render_structured_resume` walks entries in the
   model's ranked order, filling a page budget. Exclusions are honoured only
   while the remaining entries can still meet a bullet minimum, and are rolled
   back in ranked order when they cannot — so an over-aggressive model cannot
   produce an empty resume. Emphasis is restored to canonical forms, JD tools
   are bolded within a cap, and skills are reordered and reconciled against the
   bullets actually rendered. `lint_render_fidelity` reports findings.
9. **Compile** — checks the local LaTeX environment, compiles in a temp
   workspace with engine fallback and a rerun pass inside the total timeout, and
   writes the log to `.resume_cache/compile_logs/<profile>.log`. On overflow the
   pipeline can re-enter with a condense pass (`resume_condense_latex`); on
   failure, a repair pass (`resume_repair_latex`).
10. **Upload** — the rewritten `.tex` is attached, then the PDF.

**Degradation ladder.** Provider fallback → cached selection → deterministic
assembly. If every provider fails, the cover letter is still assembled from the
omitted content and extracted listing keywords, and the resume still renders
from the last good selection. The pipeline is built not to hard-fail.

**Cover letters** ([`cover.py`](src/services/resumes/cover.py)) deliberately use
what the resume dropped — hidden entries, honoured exclusions, inclusions that
lost the budget race — as primary material, so the two documents complement
rather than repeat each other. The model returns only JSON paragraphs; the LaTeX
letter is rendered from a fixed, dependency-free preamble so it always compiles.

**Bootstrap** ([`bootstrap.py`](src/services/resumes/bootstrap.py)) converts a
freeform `template.tex` + `baseinfo.txt` pair into the structured format. The LLM
only *points at* existing template lines and proposes category slugs and skill
anchors; Python inserts the tags. Nothing is accepted until the candidate files
round-trip through the real `load_structured_profile` and a baseline render
completes with zero fidelity findings.

Further detail lives in [src/resumes/resume-system.md](src/resumes/resume-system.md)
(note: it still refers to the older `$resume` command name).

---

## Deduplication

Two independent layers, because they solve different problems.

**Row-level dedupe** happens inside a single scrape pass. `dedupe_job_rows`
merges rows that share a canonical link, or a normalised title+company, so one
job that appears on four boards becomes one message listing four sources.

**History dedupe** happens across passes. Per channel and watcher type, the bot
keeps a FIFO of listing files under `dedup_listings/`:

- Each message is reduced to an ASCII signature (`message_to_ascii_signature`)
  and stored with a compressed timestamp.
- `max_entries_per_file` (500) rows per file, `max_fifo_files` (6) files, oldest
  rotated out — a bounded history rather than an ever-growing one.
- A signature older than `months_threshold` (2) no longer suppresses a repost,
  so a genuinely re-opened role can come through again.
- Signatures are namespaced per listing file so channels do not share dedup
  state, with a legacy-signature path for entries written before namespacing.
- `watcher.dedupe_seconds` (120) additionally suppresses near-simultaneous
  duplicates within one send cycle, and `discord_history_check_limit` (5)
  re-checks the last few channel messages so a restart mid-send does not
  double-post.

---

## Data directory

| Path | Contents |
| --- | --- |
| `data/ats_companies/` | Per-platform company slug lists, git-synced by `sync_ats_companies.py`. |
| `data/dead_slugs/` | Slugs that returned 404/410, with expiry timestamps. |
| `data/geo.db` | SQLite geo database used by the JBA geo layer and Glassdoor location resolution. Gitignored; built by `scripts/build_geo_db.py`. |
| `data/geo_lookup.json.gz` | Compressed city/subdivision/country lookup for ATS location matching. |
| `data/geonames_raw/` | GeoNames export the geo DB is built from. Gitignored; pulled by `sync_geonames.py`. |
| `data/locations.json` | Location enrichment map for the JBA scraper. |
| `data/jba/jobs/` | `jobs.db` (current week) plus weekly and monthly zip archives. |
| `data/ats_discovery_state.json` | ATS discovery progress. |

Runtime files in the repo root: `.bot_state.json` (per-channel settings),
`.bot.pid`, `.bot.lock`, `.start.lock`, `.last_start.txt`, `.resume_cache/`,
`dedup_listings/`, `chrome_profile/`, `chrome_profile_runtime/`,
`.browser_service.log`, `.interaction_trace.log`, `logs/bot_console.log`.

---

## Testing and CI

```bash
pytest -q
```

`pytest.ini` scopes collection to `tests/` and excludes `chrome_profile`,
`dedup_listings`, `.venv`, and `.resume_cache`. The suite covers the pieces most
likely to break silently:

| Area | Tests |
| --- | --- |
| Commands | `test_command_aliases.py`, `test_command_permissions.py`, `test_cheatsheet_pinning.py`, `test_jobsettings_modal.py` |
| Scheduling | `test_priority_scheduler.py`, `test_priority_scheduler_integration.py`, `test_ats_scrape_scheduling.py` |
| Browser | `test_browser_service_dispatch.py`, `test_browser_service_ensure_ready.py`, `test_browser_service_harvest.py`, `test_health_browser.py` |
| Job pipeline | `test_job_service_descriptions.py`, `test_job_service_keyword_variants.py`, `test_job_scrape_dedup_cache.py`, `test_watcher_dedup_resilience.py`, `test_ziprecruiter_scraper.py` |
| Best-match ranking | `test_job_match.py` |
| Reddit | `test_reddit_service_circuit_breaker.py`, `test_reddit_service_concurrency.py` |
| Resume | `test_resume_service.py`, `test_structured_resume.py`, `test_structured_domain_agnostic.py`, `test_domain_fit_audit.py`, `test_grounding_audit.py`, `test_cover_letter.py`, `test_profile_bootstrap.py`, `test_provider_capabilities.py` |
| Container | `test_container_config.py` (invariants of `Dockerfile`/`docker-compose.yml` that were each a real bug: a healthcheck that matched its own shell and reported healthy forever, a restart policy that looped on config errors, personal profiles baked into an image layer, and an `env_file` that made every compose command fail on a fresh clone) |
| Repo hygiene | `test_requirements_declared.py` (every third-party import in `src/` is declared -- `pypdf` was not, and its absence silently disabled the ATS audit), `test_state_isolation.py` (the autouse fixtures really do redirect real-state paths, so tests cannot write into the live archive or `.interaction_events.log`) |

CI (`.github/workflows/ci.yml`) runs `pytest -q` on Python 3.14 for pushes to
`main`/`master` and all pull requests. The version is pinned to what the bot
runs under so a red run means a real regression, not an environment mismatch.
CI installs only `requirements.txt` — semantic matching degrades gracefully, so
torch is not worth the CI minutes.

Pushing a `v*` tag triggers `cd-release.yml`, which runs the tests, zips the
commit, and publishes a GitHub Release. It can also be run manually from the
Actions tab. No custom secrets are needed — it uses the built-in `GITHUB_TOKEN`,
but the repo must have **Settings → Actions → General → Workflow permissions**
set to *Read and write*. Full notes in [docs/ci-cd.md](docs/ci-cd.md).

---

## Operational runbook

**A job channel stopped posting.** Run `.jobtest` in that channel — it runs the
full pipeline in the foreground with per-stage metrics and shows which stage is
dropping everything. Then `.health` for that channel's error history.

**Everything is being filtered out.** Lower `semantic_threshold` in the channel's
job settings, or set `[semantic] match_target = "title"`, or set
`enabled = false` to fall back to keyword matching.

**Reddit returns nothing.** Run `python verify_reddit.py`, which reports which
bypass layers are active (SOCKS5 proxy / curl_cffi) and what each rung returns.
If the browser rung is the problem, re-run `setup_reddit_browser.py` with the bot
stopped.

**Chrome profile lock errors that will not clear.** The lock owner is almost
certainly elevated or in Session 0. Check for a Task Scheduler entry with "Run
with highest privileges" and disable it; start the bot from a normal shell.
`run.bat noelevated` makes this fail loudly instead of silently.

**"Another runtime owner is active."** A live process holds `.bot.lock`. Use
`run.bat forcerun`, which kills by PID file, lock file, and then WMI scan, and
reports survivors it could not kill.

**Slash commands are stale or duplicated.** Set `DISCORD_FORCE_SYNC_ONCE=1` (or
create a `.force_slash_sync_once` file) and restart once. It clears guild-scoped
commands for `DISCORD_SYNC_GUILD_ID` and republishes the globals, then deletes
the marker.

**Commands are typed but nothing happens.** The log line about empty message
content means the Message Content Intent is not enabled on the Discord
application.

**Resume compile fails.** Read `.resume_cache/compile_logs/<profile>.log`. Run
`.resumecheck` to compile the template with no LLM in the loop and isolate
whether the problem is the template/TeX install or the generated content.

**Debug scripts** live in `scripts/`: `bootstrap_profile.py`,
`debug_glassdoor_trace.py`, `migrate_to_sqlite.py`, `run_resume.py`, and several
targeted `test_*.py` probes. `.e2e_pipeline.py` in the root runs a full
channel-settings → scrape → resume → PDF pass outside Discord.

---

## Repository layout

```
src/
  app.py                    Discord client, slash-command registration, PID/lock,
                            graceful drain, startup wiring
  config.py                 .env + settings.toml -> AppConfig
  commands/handlers.py      command routing, aliases, cheatsheets, permissions,
                            resume/cover/check handlers, pagination
  ui/views.py               Discord modals and buttons for every settings panel
  state/store.py            per-channel runtime state -> .bot_state.json
  watchers/manager.py       job/reddit/ATS loops, watchdog, send + dedup guards
  services/
    job_service.py          JobSpy orchestration, first-party Glassdoor and
                            ZipRecruiter scrapers, normalisation, dedupe,
                            keyword + semantic filtering, dedup FIFO
    ats_service.py          Greenhouse, Lever, Ashby, Workday, iCIMS, BambooHR
    job_match.py            profile-vs-archive ranking behind .bestjobs:
                            lexical prefilter, then one batched LLM judge call
    browser_service.py      Playwright/Chrome context, profile sync and session
                            harvest, lock clearing, priority dispatch gate
    reddit_service.py       curl_cffi -> RSS -> Playwright ladder, breaker,
                            proxy rotation, gallery extraction, seen state
    scrape_service.py       generic .scrape handler and Discord chunking
    rss_service.py          Atom parsing and HTML link extraction
    priority_scheduler.py   two-tier, shortest-job-first thread pool with aging
    health.py               health accumulation and Discord embeds
    net_util.py             retry backoff, proxy-pool parsing and rotation
    scheduler_labels.py     the scheduler's cost-history label vocabulary
    jba/                    isolated job-board-aggregator copy:
      geo_db.py             SQLite geo access + Glassdoor location-ID cache
      geolocation.py        location-string parsing and coordinate lookup
      merge_data.py         weekly SQLite -> weekly zip -> monthly zip archiving
      scraper.py            standalone multi-platform aggregator
    resumes/
      structured.py         template catalog, structured prompt, JSON salvage,
                            validation, domain-fit and grounding audits,
                            deterministic render, fidelity lint
      resume.py             provider order and capabilities, profile management,
                            LaTeX environment checks and compilation
      listing.py            job-context parsing, per-source posting scrapers,
                            provider dispatch and fallback, telemetry
      cover.py              cover letters from omitted resume content
      bootstrap.py          freeform profile -> structured profile
      cache.py              Gemini explicit cache management
      configkey.py          render-config keys
      resumes_cache/<key>/  baseinfo.txt, instructions.txt, template.tex,
                            template.log per profile
resumes/                    your real resume source material (.md / .txt)
data/                       geo DB, ATS lists, dead slugs, job archives
scripts/                    debug, probe, and migration one-offs
tests/                      pytest suite
docs/ci-cd.md               workflow documentation
autofill_extension/         companion Chrome extension (see below)
run.py                      cross-platform launcher (Windows/Linux)
src/services/platform_support.py  all OS-specific decisions live here
src/services/capacity.py    hardware-aware pool scaling
src/services/jba/archive_index.py  archive dedup index (4-month window)
scripts/build_geo_db.py     rebuilds data/geo.db from data/geonames_raw/ (sources come from sync_geonames.py)
scripts/compute_metrics.py  operating metrics from the job archives
run.bat                     supervised Windows launcher (legacy)
deploy/discordbot.service   systemd unit for Linux deployment
Dockerfile                  container image (Chromium + TeX + git)
docker-compose.yml          container deployment; see deploy/DOCKER.md
deploy/docker-entrypoint.sh container preflight (lock clear, token check)
deploy/docker_smoke.py      proves the container can launch Chromium and compile a PDF
.github/workflows/docker-image.yml  builds the image and runs the smoke test in CI
sync_ats_companies.py       sparse git sync of ATS company lists
sync_geonames.py            pulls the GeoNames export and rebuilds data/geo.db
setup_reddit_browser.py     one-time Reddit login into the bot's Chrome profile
verify_reddit.py            diagnostic for the Reddit bypass ladder
.e2e_pipeline.py            end-to-end scrape -> resume -> PDF outside Discord
```

---

## Companion projects

**`autofill_extension/`** — a Manifest V3 Chrome extension, "Application
Autofill", that fills job application forms from a profile stored only on the
local device. No account, no server. It has its own `background/`, `content/`,
`popup/`, `options/`, `shared/`, and `tests/` directories.

---

## License

See [LICENSE](LICENSE).
