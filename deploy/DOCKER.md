# Running the bot in Docker

The bot is not a pure Python process: it drives Chromium through Playwright,
shells out to `pdflatex` for resume PDFs, and runs `git` against its own working
tree to publish job archives. The image carries all three. Everything writable
is resolved from the install directory (`.bot.lock`, `.bot_state.json`, `data/`,
`logs/`, `dedup_listings/`, `chrome_profile/`), which is why
`docker-compose.yml` bind-mounts the repo at `/app` instead of using named
volumes — state stays on the host and survives every rebuild. The one
deliberate exception is Chromium's *runtime* profile, which is pushed onto a
named volume; see below for why.

## Before this can work anywhere but this machine

The container depends on source that is currently **untracked**. A `git clone`
of the current HEAD is missing all of these:

```
run.py                                  <- the image's CMD
src/services/platform_support.py        <- Chrome discovery
src/services/capacity.py                <- pool sizing
src/services/quota.py
src/services/job_match.py
src/services/jba/description_cache.py
src/services/resumes/ats_check.py       <- the Type 3 / ATS gate
deploy/docker_smoke.py                  <- the verification script
deploy/docker-entrypoint.sh             <- the entrypoint itself
```

That list is generated, not guessed: it is every module reachable by walking
imports from `run.py`, `src/app.py` and `deploy/docker_smoke.py` that git does
not track. A hand-written version of it missed three.

Locally this is invisible, because the bind mount supplies the working tree. It
bites on a fresh clone, in CI, and in any deployment that pulls rather than
copies: the image builds fine and *then* fails with `can't open file
'/app/run.py'` or an ImportError, neither of which points at the cause.

`.github/workflows/docker-image.yml` has a preflight step that checks for these
and fails naming the file. Commit them alongside the Docker files.

## Getting a Docker daemon (without Docker Desktop)

Docker Desktop is not required, and on Windows its installer needs elevation.
This works with no UAC prompt once the WSL 2 platform is enabled:

```bash
wsl --install -d Ubuntu --no-launch
```

```bash
wsl -d Ubuntu -u root -- bash -lc "apt-get update && apt-get install -y docker.io docker-compose-v2"
```

Then enable systemd so the daemon starts itself, rather than needing `dockerd`
run by hand every time:

```bash
wsl -d Ubuntu -u root -- bash -lc "printf '[boot]
systemd=true
' > /etc/wsl.conf"
```

```bash
wsl --shutdown
```

```bash
wsl -d Ubuntu -u root -- bash -lc "systemctl enable --now docker && docker info --format 'Server: {{.ServerVersion}}'"
```

Verified: after a full `wsl --shutdown`, a cold `wsl -d Ubuntu` brings the
daemon back on its own (`systemctl is-active docker` reports `active`). Without
the `wsl.conf` step you get no init system, and `dockerd` has to be started
manually after every restart.

One thing that bites here: **`docker.io` does not ship Compose.** `docker
compose` is a separate package (`docker-compose-v2`); Docker Desktop bundles
it, a plain engine install does not. Every command below assumes it.

The repo lives on the Windows filesystem, so from inside WSL the build context
is under `/mnt/c/...`. That works, but it is a 9p mount and noticeably slower
than a checkout inside the distro — see the SQLite note near the end.

## Quick start

```bash
docker compose build
```

```bash
docker compose up -d
```

```bash
docker compose logs -f
```

`.env` is mounted, never baked into the image. It must exist before the first
`up`: the entrypoint exits 1 with a clear message when both `/app/.env` and
`discordtoken` are absent.

That check applies only when the command is the bot (`run.py` / `src/app.py`).
The one-off commands below need no Discord token and are meant to run *before*
the first start — when `.env` may not exist yet — so they are let through.

## Do not start the container while a bot already runs on this checkout

`docker compose up` reads the same `.env`, so it logs in as the *same* Discord
bot, and it bind-mounts the same repo, so it shares `.bot_state.json`, the dedup
listings and the job archive with whatever is already running on the host. Its
entrypoint also matches `run.py`, so it will clear the host instance's
`.bot.lock` on the way in.

Stop the host bot first (`python run.py` exits on SIGTERM and drains), or point
the container at a separate checkout. One-off `docker compose run` commands are
safe alongside a live bot -- that is exactly why the entrypoint leaves the lock
and browser profile alone for them.

## First run: the two generated artifacts

Neither is in git, and without them the bot starts but does nothing useful — no
companies to scrape, and no city/region matching. The entrypoint warns about
each on every start.

`data/ats_companies` repairs itself: `run.py` shells out to
`sync_ats_companies.py` on every start, which clones it when absent. You only
need this to seed it before the first `up`, or to see the clone fail loudly
rather than buried in the startup log:

```bash
docker compose run --rm bot python sync_ats_companies.py
```

`data/geo.db` does not repair itself — nothing rebuilds it automatically, and
it is excluded from the image, so run this once:

```bash
docker compose run --rm bot python scripts/build_geo_db.py
```

Verify with `docker compose run --rm bot python scripts/build_geo_db.py --check`.

## Verify the image actually works

```bash
docker compose run --rm bot python deploy/docker_smoke.py
```

Verified: on a real build of this image every check passes — Chromium launches
headless and renders, all 23 LaTeX files resolve, chktex is present, and a real
`pdflatex` compile audits as `ATS audit: ok` (which also proves `pypdf` is
installed, since without it the audit degrades to `unknown`).

This also runs in CI: `.github/workflows/docker-image.yml` builds the image and
runs the same script on a runner, so the apt layer and the Chromium launch get
verified without a local Docker install. It is scoped to the files that define
the image (plus `workflow_dispatch`), because the TeX trees make the build slow
and nothing about the image changes when only `src/` does.

Import checks would pass on an image where Chromium cannot launch and pdflatex
silently emits bitmaps — the two failures that actually matter — so every check
runs the real thing: it launches headless Chromium and renders a page, probes
each LaTeX style file the resume templates use with `kpsewhich` (naming the
missing ones rather than failing on whichever the compiler hit first), compiles
a real PDF and judges it with the pipeline's own `audit_pdf_ats`, resolves the
jobspy interpreter, and writes to the state paths. Non-zero exit on any
required failure.

Reported as optional rather than fatal: the jobspy interpreter, the LaTeX style
set, `chktex`, and git-on-the-working-tree. That last one is optional because
archive publishing is opt-in — `JBA_ARCHIVE_GIT_COMMIT` is off by default, so a
deployment with no `.git` is legitimate. When `.git` *is* present the check still
runs, because `git status` failing there is the "dubious ownership" symptom that
would break the archive commit later.

## Build arguments

| Arg | Default | Effect |
| --- | --- | --- |
| `WITH_LATEX` | `1` | The TeX trees, plus `chktex`. Measured: **5.71 GB** built, versus **2.14 GB** at `0` — so this is ~3.6 GB. At `0`, `.resumebuild` reports "LaTeX compile environment is not ready" and nothing else changes. |
| `WITH_SEMANTIC` | `0` | At `1`, installs `sentence-transformers` (pulls torch, ~2GB). Without it job matching silently falls back to keyword-only. Safe alongside jobspy — see below for why that needed arranging. |
| `UID` / `GID` | `1000` | The container user's ids. They must match the host user owning this checkout, or the container cannot write into the bind mount — no state file, no lock, no archives. Docker Desktop maps ownership for you; on a Linux host whose uid is not 1000, build with `UID=$(id -u) GID=$(id -g) docker compose build`. |

```bash
docker compose build --build-arg WITH_LATEX=0
```

## What has and has not been verified

Everything in this document is measured on **one** environment: Linux 6.18
(WSL2), x86_64, Docker 29.1.3, overlay2, cgroup **v2**, no SELinux. Within that
it has been exercised hard -- `cap-drop=ALL`, a read-only root filesystem, pid
and CPU/memory limits, UID 1000 and 1500, four timezones, Python 3.12 and 3.14,
and a fresh clone.

Outside it, be sceptical:

| | Status |
| --- | --- |
| Modern x86_64 Linux, cgroup v2, no SELinux | verified |
| SELinux hosts | **will fail without `:z`** -- see below |
| arm64 / Raspberry Pi | unbuilt, unrun |
| cgroup v1 hosts | `capacity.py`'s v1 path is unit-tested (`tests/test_capacity.py`), never run in a real v1 container |
| Rootless Docker / Podman | untested; uid mapping differs, so the `UID`/`GID` args mean something else |
| Older daemons with restrictive seccomp | untested -- the reason `CHROME_NO_SANDBOX=1` is kept |
| Sustained operation as a live bot | **never done anywhere.** Verified only up to the Discord login boundary |

## SELinux hosts (RHEL, Fedora, CentOS, Rocky, Alma)

**The compose file as written will fail there**, and this is the most likely
real-world portability failure in the whole setup.

`volumes: - .:/app` has no SELinux label option. Under SELinux in enforcing
mode a container cannot read a bind mount it has not been granted, so the bot
gets permission denied on the entire working tree -- not a degradation, a hard
stop. The symptom is misleading: Python reports missing files rather than
anything mentioning SELinux.

Fix by labelling the mount:

```yaml
    volumes:
      - .:/app:z
```

`:z` relabels the content as shared between containers; `:Z` labels it private
to this one. Either works here; `:z` is the safer default if you ever run a
one-off container against the same checkout.

This is **not** the default because `:z` relabels recursively, and this repo can
carry a multi-GB `chrome_profile/` -- on a Windows or plain-Linux host that cost
buys nothing, since the option is ignored where SELinux is not enforcing. Add it
when you deploy to a Red Hat-family host.

To be explicit about what is and is not verified here:

- **`:z` and `:Z` are harmless where SELinux is not enforcing** -- verified. The
  full smoke test passes with `-v "$PWD:/app:z"` and with `:Z`, and
  `docker compose config` accepts the long form. So adding it costs nothing on a
  Debian/Ubuntu host, and you can leave it in place.
- **Whether it *fixes* an enforcing host is not verified.** This machine's
  kernel has SELinux compiled in (`/sys/fs/selinux` exists) but never
  initialised, so nothing here exercises a real denial. The remedy is the
  standard one for this failure, but treat it as documented, not proven.

## Running on a hardened host

The image needs no Linux capabilities and no writable root filesystem. Measured,
each running the full smoke test:

| Constraint | Result |
| --- | --- |
| `--cap-drop=ALL` | passes |
| `--cap-drop=ALL --security-opt no-new-privileges:true` | passes |
| `--read-only` root filesystem | passes, with the tmpfs caveat below |
| `--pids-limit=200` | passes |
| `--memory=1g --cpus=1` | passes (pools scale down; see sizing above) |

So it is deployable under a restrictive orchestrator policy, not just on a
permissive dev box.

The read-only case has one trap worth knowing: `--tmpfs /home/bot:rw` mounts
**root-owned** (`drwx------ root root`), and the container runs as uid 1000, so
Chromium fails to launch with a `TargetClosedError` that says nothing about
permissions. Give the tmpfs an owner:

```bash
docker run --rm --read-only --shm-size=1g -v "$PWD:/app"   --tmpfs /tmp:rw,exec,uid=1000,gid=1000   --tmpfs /home/bot:rw,exec,uid=1000,gid=1000   discordbot:latest python deploy/docker_smoke.py
```

Match `uid=`/`gid=` to the `UID`/`GID` the image was built with.

## ARM64 / Raspberry Pi

**Untested.** Nothing in the Dockerfile or compose file pins an architecture and
the base image publishes `linux/arm64/v8` (verified via `docker manifest
inspect`), so there is no *known* blocker -- but this image has only ever been
built and run on x86_64. Debian's `chromium` and the TeX trees on arm64, and the
build time on a Pi, are all unverified. Treat the below as reasoning, not
evidence.

The design does line up with the README's ARM note: the image
installs the *distro* `chromium` rather than downloading one, which is exactly
what that section says to do — Playwright's bundled Chromium does not reliably
cover `linux-arm64`, and Google ships no `google-chrome-stable` for it.

Expect the TeX layer to be slow to install on a Pi; `WITH_LATEX=0` is the
obvious build there unless you need the resume commands.

## What the image costs

Measured, not estimated (`docker history`). The default build
(`WITH_LATEX=1`, `WITH_SEMANTIC=0`) is **5.71 GB**; the same build with
`WITH_LATEX=0` is **2.14 GB**, and breaks down like this:

| Layer | Size | Note |
| --- | --- | --- |
| apt (chromium, fonts, git, procps) | 832 MB | dominated by Chromium; unavoidable, the browser is the point |
| pip main environment | 296 MB | 47 packages (11 declared + transitive) |
| jobspy virtualenv | 290 MB | **the price of the numpy split** — it carries its own numpy 1.26.3 and pandas rather than sharing |
| source tree | 22.7 MB | after `.dockerignore`; was 34.5 MB before the `**/` pattern fix, and would be 640 MB+ with `data/` |
| user creation | 78 kB | `COPY --chown`, not `chown -R`, which would have duplicated the 22.7 MB |

`WITH_LATEX=1` adds ~3.6 GB of TeX on top, of which `texlive-fonts-extra` alone
is a 624 MB download. If you do not use the resume commands, `WITH_LATEX=0`
is by far the biggest saving available.

So the jobspy isolation is not free. It is still the right trade — the
alternative is an environment where installing the semantic extras silently
breaks jobspy's pinned numpy — but if you never use the jobspy-backed sites,
dropping `python-jobspy` from `requirements.txt` reclaims 290 MB and the venv
step skips itself.

## Which apt package provides which LaTeX style

Verified against Debian trixie's package contents, because the templates pull
fonts that are not in the base TeX install. Drop a package here and the
templates using its styles stop compiling — that is the real cost of
`WITH_LATEX=0`, and of any attempt to trim this list further.

| Package | Provides (among others) |
| --- | --- |
| `texlive-fonts-extra` | `fontawesome5`, `XCharter`, `CormorantGaramond`, `sourcesanspro`, `noto-sans`, `roboto`, `FiraSans` |
| `texlive-fonts-recommended` | `marvosym`, `charter` |
| `texlive-latex-extra` | `titlesec`, `fullpage`, `enumitem` |
| `texlive-latex-recommended` | the base layout set — `geometry`, `hyperref`, `fancyhdr`, `tabularx`, `parskip` |
| `texlive-base` | `glyphtounicode.tex` — not a `.sty`; the templates `\input` it unconditionally, so without it every resume compile dies on that line. Named explicitly rather than left to arrive transitively. |
| `lmodern` | `lmodern` — scalable Type 1 defaults. Measured in this image on the smoke test's own document: **with** it, `pdflatex` exits 0 and the audit says `ok` at 0% Type 3; **without** it, `pdflatex` still exits 0 and the audit says `unreadable` at **100% Type 3**. The exit code tells you nothing; this package is the difference between a readable resume and a bitmap. |
| `chktex` | the `chktex` linter. Optional in code (`if chktex_path:`), so its absence does not break compiles — it just quietly thins the lint findings fed back into the resume rebuild loop. |

`docker_smoke.py` re-checks all of these with `kpsewhich` inside the image, and
a real build reports **all 23 present**. Note
that running that script on a Windows host proves nothing about this: MiKTeX
installs missing packages on demand, so everything "passes" there regardless.

## What the container settings are actually for

- **`CHROME_NO_SANDBOX=1`** (set in the Dockerfile) makes
  `platform_support.chrome_sandbox_args()` return `--no-sandbox
  --disable-dev-shm-usage`.

  **It is probably no longer necessary, and it costs you the Chromium sandbox.**
  Measured in this image on Docker 29.1.3: Chromium launches and renders a page
  with the flag, *without* it, and without it under
  `security_opt: no-new-privileges:true`. The advice it came from predates
  daemons whose default seccomp profile permits the unprivileged user namespace
  that Chromium's namespace sandbox needs. This bot renders untrusted pages
  (Reddit, job boards), so the sandbox is a real boundary worth keeping.

  It is still set here because the evidence is from one host: where unprivileged
  user namespaces are restricted (some hardened kernels, older daemons),
  Chromium falls back to the setuid helper, which `no-new-privileges` then
  neuters — and the browser layer simply fails to start. To drop it, remove the
  variable from the `ENV` block in the Dockerfile and rebuild; if Chromium then
  fails to launch, put it back via `.env` and nothing else changes.

  One coupling to know: the same flag also carries `--disable-dev-shm-usage`.
  Compose sets `shm_size: 1gb`, which addresses that properly, but a bare
  `docker run` without `--shm-size` would get Chromium's 64MB default.

  `docker compose run --rm bot python deploy/docker_smoke.py` reports which
  posture is in effect and, more importantly, whether Chromium actually
  launches.
- **No `playwright install` in the image, deliberately.** The pip package ships
  the driver; `playwright install` downloads *browsers*, and this codebase never
  opens one. `browser_service` has a single launch site and it passes
  `executable_path=find_chrome()`, which probes system locations only, and
  `_do_start` disables the whole Chrome layer when that returns None rather than
  falling back to a bundled Chromium. The distro `chromium` package is therefore
  not a convenience here — it is the browser. (`reddit_service` does not launch
  its own: it goes through `browser_service.ensure_ready()`.)
- **The healthcheck pattern is `"[s]rc/app.py"`, and the brackets matter.**
  Docker runs the CMD through `sh -c`, so that shell's own command line contains
  the pattern and `pgrep -f` scans it. A plain `"src/app.py"` matches itself and
  reports healthy forever -- a container running only `pip` came back
  `(healthy)`, with `pgrep -af` naming `1 sh -c pgrep -af "src/app.py"` as the
  match. The bracketed form matches the bot's real cmdline and not the shell's.
- **No equivalent of the unit's `RuntimeMaxSec=86400`.** systemd restarts the
  bot once a day to shed accumulated Chrome/Playwright state; Docker has no
  max-runtime setting, and `stop_grace_period` only bounds shutdown. If you want
  that behaviour, schedule it:

  ```bash
  docker compose restart bot
  ```

  It goes through SIGTERM, so the 300s drain still applies. Without it the
  container simply runs until something stops it -- which is a behaviour
  difference from the systemd deployment, not an oversight.
- **`security_opt: no-new-privileges:true`** — the unit sets this too but warns
  it breaks Chrome's SUID sandbox helper on some hosts. That trade-off does not
  exist here: `CHROME_NO_SANDBOX` means Chromium launches with `--no-sandbox`,
  so there is no setuid helper involved. Verified by running the smoke test with
  it set — Chromium still launches and renders.
- **`shm_size: 1gb`** — Chromium's default `/dev/shm` in a container is 64MB.
  Verified rather than assumed: a bare `docker run` of this image reports
  `/dev/shm` at **64M** and `nofile` soft **1024**; under compose they are
  **1.0G** and **65536**, and PID 1 is `docker-init` rather than the command
  itself. The `ulimits` entry raises only the soft limit -- the hard limit is
  left at the daemon's default (1048576), since nothing here raises its own
  limit and pulling the ceiling down would only remove headroom.
- **`restart: on-failure:5`, not `unless-stopped`.** `app.py` exits 0 when
  another instance already holds the runtime lock and 1 on a missing token --
  neither is worth retrying forever. Measured: under `unless-stopped` both cases
  restart endlessly (8 restarts in 14 seconds, container stuck "restarting",
  nothing indicating a config error). Under `on-failure:5`, exit 0 does not
  restart at all and a config error is bounded, then visible. This mirrors the
  `Restart=on-failure` + `RestartPreventExitStatus=1` reasoning in
  `deploy/discordbot.service`; Docker cannot express the second half, so a
  bounded count stands in for it. The cost is that the container is not
  resurrected when the daemon restarts -- run `docker compose up -d` after a
  host or daemon restart.
- **`init: true`** — Chromium leaves zombies when a page crashes, and PID 1 has
  to reap them. Measured in this image: with a Python PID 1 and an orphaned
  grandchild, `ps` shows **1 process in `Z` state**; with `--init` it shows
  **0**. `docker-init` becomes PID 1 and reaps them.
- **`stop_grace_period: 300s`** — `app.py` drains scrapes (120s), then watcher
  sends (30s), then joins the scheduler pool. This matches `TimeoutStopSec` in
  the systemd unit. A shorter grace period means SIGKILL mid-drain, and that is
  not hypothetical: a container trapping SIGTERM and draining for 20s exits
  **137 after 5s** under `docker stop -t 5`, and **0 after 22s** under
  `-t 60`. Docker's default is 10s, so without this setting the bot would be
  killed partway through a drain that can legitimately take 150s.
- **Chromium's runtime profile lives off the bind mount**
  (`REDDIT_CHROME_RUNTIME_PROFILE=/home/bot/chrome_profile_runtime`, set in the
  **image** so it holds for a bare `docker run` too, and backed by a named
  volume under compose). It is
  LevelDB, a cookie SQLite and `SingletonLock` — file locking that Docker
  Desktop's virtiofs/9p only emulates. It is throwaway state, re-primed from
  `chrome_profile/` on every start. The *source* profile stays bind-mounted:
  that is the Reddit session you log into on the host.
- **The entrypoint clears `.bot.lock` / `.bot.pid` — but only when this
  container is about to be the bot.** `app.py` treats the lock as held when the
  recorded pid is alive: sound on a host, wrong here, because pids restart from
  1 in a fresh PID namespace while the file survives in the bind-mounted repo,
  so a stale `"pid": 1` reads as a live owner and the bot exits 0 forever.

  The condition matters. `docker compose run --rm bot python …` runs the same
  entrypoint against the same bind mount and the same `bot-home` volume as the
  already-running bot, so an unconditional clear would delete a *live* owner's
  lock (letting a second bot start) and yank the running Chromium's
  `SingletonLock`. Every one-off command in this document does exactly that, so
  the entrypoint matches `run.py` / `src/app.py` as whole argv tokens and leaves
  both alone otherwise.

  **Corollary: never run two bot containers over one repo checkout.** The PID
  namespace is what makes the clear safe, and it only holds one bot.

## Why jobspy gets its own virtualenv

`python-jobspy` pins `NUMPY==1.26.3` *exactly*. `sentence-transformers` (the
`WITH_SEMANTIC=1` extra) resolves numpy to 2.x. One environment cannot hold
both — installing the semantic extras on top of `requirements.txt` silently
upgrades numpy out from under jobspy.

So the image installs `requirements.txt` minus jobspy into the main
environment, and jobspy alone into `/opt/jobspy`, with
`JOBSPY_PYTHON_EXE=/opt/jobspy/bin/python`. This is not a workaround bolted on
from outside: `job_service.jobspy_python_executable()` checks that variable
first, and jobspy is only ever run as a *subprocess*, never imported in the hot
path. Nothing under `src/` imports pandas or numpy, so the main environment
genuinely does not need them — resolving `requirements.txt` without jobspy
yields 45 packages and no numpy at all.

The base is **3.12** because the jobspy venv inherits the base interpreter and
numpy 1.26.3 ships no cp313 wheel. Verify either resolution without building:

```bash
pip install --dry-run --report /dev/null --target /tmp/x --platform manylinux_2_17_x86_64 --python-version 3.12 --only-binary=:all: -r requirements.txt
```

On `--python-version 3.13` that same command fails with `Could not find a
version that satisfies the requirement NUMPY==1.26.3 (from python-jobspy)` —
the lowest cp313 candidate is 2.1.0.

Nothing in `src/` uses 3.13-only syntax; `tomllib` in `config.py` sets the real
floor at 3.11. Note the host runs 3.14 with no `python-jobspy` installed at
all, so the jobspy-backed sites are quiet there and work here.

## Sizing: the container sees the container's limits, not the host's

`capacity.py` reads cgroup v2 directly, and it works — verified in real
containers:

| Run | Detected | `workers(8)` |
| --- | --- | --- |
| no limits | 12 cores, 7.67 GiB | 4 |
| `--cpus=2 --memory=1g` | 2.0 cores, 1.0 GiB | 1 |
| `--cpus=0.5 --memory=512m` | 0.5 cores, 0.5 GiB | 1 |

Two consequences worth knowing:

- **You can constrain the bot** with `cpus:`/`mem_limit:` in compose and every
  thread pool scales down accordingly. `docker compose` sets no limits by
  default, so it gets whatever the daemon's host has.
- **On WSL that host is the WSL VM, not Windows.** WSL defaults to roughly half
  the machine's RAM, so the container can be sized smaller than the same bot
  running natively on Windows -- the run above saw 7.67 GiB inside a machine
  with more than that. If the scale factor looks low, check `.wslconfig` before
  suspecting the bot. `BOT_WORKER_SCALE` and `BOT_MAX_WORKERS` still override.

## One thing the bind mount does not hot-reload

Source under `src/` comes from the bind mount, so a `docker compose restart`
picks up code edits without rebuilding. The entrypoint does not: it is copied to
`/usr/local/bin/docker-entrypoint.sh` at build time and `chmod +x`-ed there, and
`ENTRYPOINT` points at that path, not at `/app/deploy/`.

So editing `deploy/docker-entrypoint.sh` and restarting silently changes
nothing. Rebuild:

```bash
docker compose build && docker compose up -d
```

It is baked deliberately rather than run from `/app/deploy/`: the image copy is
guaranteed to exist and to be executable, where a bind-mounted copy depends on
the host preserving the executable bit — which Windows checkouts do not.

## What is deliberately not in the image

`.dockerignore` keeps `src/services/resumes/resumes_cache/` out. Those are real
resume profiles — named people, their history, their contact details — and
`.gitignore` already treats them as private. An image layer is easier to hand
around than a repo, so baking them in would be a worse leak than committing
them. The bind mount supplies them at run time; a fresh clone does not have them
either, so the image behaves like a checkout rather than like one laptop.

The practical consequence: **resume builds need the bind mount.** A container run
without it has no templates. That is also why `docker_smoke.py` probes the TeX
binaries directly rather than calling `check_template_compile_environment()`,
which reads a template out of that directory.

`.env`, `chrome_profile/`, `vpn/`, and `data/ats_companies/` (a nested checkout
`run.py` re-syncs on every start) are excluded for the same reasons.

**If you edit `.dockerignore`, know that a bare pattern only matches at the
context root.** Docker matches each pattern against the file's full relative
path, so `*.log` does not match `src/foo.log` and `__pycache__/` does not match
`src/services/__pycache__/`. That is not a hypothetical: the image shipped 237
nested `.pyc` files and a nested `src/.panel_send_events.log` while this file
appeared to exclude both, and the same hole would have shipped a nested `*.pem`
or `*.key`. Anything that should be excluded at any depth needs a `**/` prefix,
and `tests/test_container_config.py` asserts that for the patterns where it
matters.

## Reddit login

`setup_reddit_browser.py` needs a graphical session, so run it on the host and
let the container pick up the resulting `chrome_profile/` through the bind
mount. The Reddit bypass ladder degrades on its own without it.

## Archive publishing will NOT work out of the box

`JBA_ARCHIVE_GIT_COMMIT` / `JBA_ARCHIVE_GIT_PUSH` are both `1` in this repo's
`.env`, and the commit half works in the container: `.git` is inside the bind
mount and the image runs `git config --system --add safe.directory '*'`, which
is required because the mounted tree is owned by the host user, and because the
setting does not apply to nested repos (`run.py` syncs `data/ats_companies`, a
checkout of its own, on every start).

The push half does not. Checked inside the container:

```
credential.helper : <none>
~/.git-credentials: absent
~/.ssh            : 0 key files
GIT_ASKPASS       : <unset>
remote            : https://github.com/ErChoi-com/DiscordBot.git
```

An https remote with no helper and no TTY cannot authenticate. The bot would
commit archives daily and never publish them — failing quietly, or hanging until
the 180s timeout and logging a rejected push. Pick one before switching a
publishing bot to the container:

- mount a credential store read-only, e.g.
  `- ~/.git-credentials:/home/bot/.git-credentials:ro`, and set
  `git config --global credential.helper store` in the image; or
- switch the remote to SSH and mount an unencrypted key at `/home/bot/.ssh`; or
- set `JBA_ARCHIVE_GIT_PUSH=0` for the container and keep publishing from the
  host.

## The cost of the Windows bind mount, measured

Running the same workloads inside the container against three backends:

| Backend | SQLite, 10k rows / 20 commits | 400 small file writes |
| --- | --- | --- |
| `/mnt/c` bind mount | 0.48s | **0.68s** |
| WSL-native ext4 | 0.38s | **0.03s** |
| container overlay (no mount) | 0.29s | 0.02s |

The interesting part is which number is bad. SQLite is only ~1.3x slower over
9p — noticeable, not painful. **Small-file writes are ~23x slower**, and that is
the workload that matters here: `dedup_listings/` writes many small JSON files,
and a Chrome profile is thousands of them. That is the real reason the Chromium
runtime profile is pushed onto a named volume rather than left on the mount.

`data/jba/jobs/archive_index.db` is hard-pinned in `archive_index.py:41` with no
env override, so it stays on the bind mount. On these numbers that is a modest
slowdown rather than a problem. The lock-emulation concern is separate and
remains theoretical — nothing here corrupted, and the database is single-writer
(guarded by `_build_lock`) and self-healing (a schema mismatch drops and
rebuilds). If you ever do see `database is locked`, moving the checkout onto
Linux-native storage is the fix — and it also buys the 23x on small files.

## VPN

`REDDIT_PROXIES` (SOCKS5) works as-is. Routing the whole container through
OpenVPN instead needs `cap_add: [NET_ADMIN]`, `devices: [/dev/net/tun]` and the
client running inside — deliberately not wired up here; the proxy path is the
supported one.
