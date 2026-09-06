# Discord job bot -- container image.
#
# Everything the bot shells out to has to be in here: Chromium (browser_service
# drives it through Playwright's executable_path), a LaTeX engine (resume PDF
# builds), and git (the archive commit/push loop). The bot resolves every
# writable path from its install directory, so /app is the state root -- see
# docker-compose.yml, which bind-mounts the repo over it.
# 3.12, not 3.13+: python-jobspy pins NUMPY==1.26.3 exactly, and numpy 1.26.3
# ships no cp313 wheel -- pip would fall back to the sdist and fail the build.
# Nothing in src/ uses 3.13-only syntax or stdlib (tomllib in config.py needs
# 3.11+, which is the real floor), so this costs nothing and buys jobspy.
FROM python:3.12-slim-trixie

# CHROME_NO_SANDBOX makes platform_support.chrome_sandbox_args() return
# --no-sandbox --disable-dev-shm-usage.
#
# Measured on Docker 29: Chromium launches and renders here with the flag,
# WITHOUT it, and without it under no-new-privileges -- so the old "Chromium
# cannot sandbox in a container" rationale no longer holds, and this costs a
# real security boundary for a bot that renders untrusted pages. It is kept for
# compatibility with hosts that restrict unprivileged user namespaces, where
# Chromium falls back to the setuid helper. To drop it, delete the line and
# rebuild; if Chromium then fails to launch, set it in .env instead. See
# deploy/DOCKER.md.
# REDDIT_CHROME_RUNTIME_PROFILE is set HERE, not only in docker-compose.yml.
# browser_service derives it from the repo path otherwise, which puts Chromium's
# working profile on the bind mount: its LevelDB and lock files then live on a
# filesystem that only emulates locking, and -- worse -- are shared with any bot
# already running on that same checkout. A `docker run` without compose used to
# inherit that default and clobber a live host bot's browser profile. Compose
# still overrides this; the point is that the safe value no longer depends on
# which launcher you happen to use.
ENV REDDIT_CHROME_RUNTIME_PROFILE=/home/bot/chrome_profile_runtime \
    CHROME_NO_SANDBOX=1 \
    CHROME_EXECUTABLE=/usr/bin/chromium \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    HOME=/home/bot

# Skip the TeX trees when you do not need resume PDFs. Measured: 5.71GB built
# at WITH_LATEX=1 versus 2.14GB at 0, so this is ~3.6GB:
#   docker compose build --build-arg WITH_LATEX=0
ARG WITH_LATEX=1

# No hand-listed X/NSS/ALSA libraries: the distro `chromium` package pulls its
# own dependency closure, and naming them by hand invites the Debian t64
# renames. fonts-liberation + fonts-noto-color-emoji are separate -- Chromium
# renders tofu without them, which breaks the screenshot paths. lmodern is not
# optional -- without a scalable default the PDF comes out as Type 3 bitmaps that ATS parsers
# cannot read, even though pdflatex exits 0. chktex is the LaTeX linter
# resume.py feeds back to the model (guarded by `if chktex_path:`), so leaving
# it out does not break compiles -- it just quietly makes the container's resume
# linting weaker than the developer's MiKTeX box, which is the kind of parity
# gap nothing ever reports. texlive-base is named explicitly for one file:
# glyphtounicode.tex, which the real templates \input unconditionally -- absent,
# every resume compile fails on that line. It would arrive transitively today;
# depending on that silently is how it goes missing later.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        chromium \
        ca-certificates \
        git \
        curl \
        procps \
        tzdata \
        fonts-liberation \
        fonts-noto-color-emoji; \
    if [ "$WITH_LATEX" = "1" ]; then \
        apt-get install -y --no-install-recommends \
            lmodern \
            chktex \
            texlive-base \
            texlive-latex-recommended \
            texlive-latex-extra \
            texlive-fonts-recommended \
            texlive-fonts-extra \
            texlive-xetex; \
    fi; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so an edit to src/ does not re-resolve the world.
COPY requirements.txt requirements-semantic.txt ./

# python-jobspy gets its OWN virtualenv, and it is not a stylistic choice:
# jobspy pins NUMPY==1.26.3 exactly while sentence-transformers resolves numpy
# to 2.x, so a single environment cannot satisfy both -- installing the
# semantic extras on top would silently upgrade numpy out from under jobspy.
#
# The codebase already has the seam. jobspy is never imported in the hot path;
# job_service runs it as a subprocess through the interpreter that
# jobspy_python_executable() resolves, and JOBSPY_PYTHON_EXE is the first
# override it checks. Nothing under src/ imports pandas or numpy directly, so
# the main environment does not need either.
RUN set -eux; \
    grep -viE '^[[:space:]]*python-jobspy' requirements.txt > /tmp/req-main.txt; \
    pip install --no-cache-dir -r /tmp/req-main.txt; \
    rm -f /tmp/req-main.txt

RUN set -eux; \
    grep -iE '^[[:space:]]*python-jobspy' requirements.txt > /tmp/req-jobspy.txt || true; \
    if [ -s /tmp/req-jobspy.txt ]; then \
        python -m venv /opt/jobspy; \
        /opt/jobspy/bin/pip install --no-cache-dir -r /tmp/req-jobspy.txt; \
    fi; \
    rm -f /tmp/req-jobspy.txt
ENV JOBSPY_PYTHON_EXE=/opt/jobspy/bin/python

# Optional and heavy (torch, ~2GB): semantic job matching. Safe to put in the
# main environment now that jobspy's numpy pin lives elsewhere. Without it the
# bot falls back to keyword matching, silently.
ARG WITH_SEMANTIC=0
RUN if [ "$WITH_SEMANTIC" = "1" ]; then pip install --no-cache-dir -r requirements-semantic.txt; fi

# The user is created BEFORE the source is copied so that COPY --chown can set
# ownership as it writes. `COPY . .` followed by `chown -R` would rewrite every
# file in a second layer, duplicating the whole tree in the image.
#
# `chown bot:bot /app` is separate and necessary: WORKDIR created that directory
# as root, and COPY --chown only owns the files it writes, not their parent. Without
# it the bot cannot create .bot.lock and dies with EACCES on startup -- which a
# bind mount hides completely, because the mount replaces /app with a writable
# host directory. It costs one inode, not a copy of the tree.
#
# UID/GID are build args because the bind mount is the point: files in it are
# owned by the *host* user, and a container user whose uid does not match cannot
# write .bot_state.json, the lock files, or the archives. On a Linux host whose
# uid is not 1000, build with:
#   docker compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)
# Docker Desktop (macOS/Windows) maps ownership for you, so the default is fine.
ARG UID=1000
ARG GID=1000

# safe.directory '*' rather than '/app': the bind-mounted tree is owned by the
# host user, not by `bot`, and the setting does not apply to nested repos --
# run.py syncs data/ats_companies (its own checkout) on every start, and the
# archive push touches /app/.git. A wildcard in a single-purpose container is
# the honest trade; the alternative is enumerating repos that move.
RUN set -eux; \
    groupadd --gid "$GID" bot || true; \
    useradd --create-home --uid "$UID" --gid "$GID" --shell /bin/bash bot; \
    git config --system --add safe.directory '*'; \
    chown bot:bot /app

COPY --chown=bot:bot . .

COPY --chown=bot:bot deploy/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER bot

# app.py's own single-instance lock is the real guard; this only tells the
# supervisor the process is still up.
#
# It deliberately looks for src/app.py -- the bot itself -- not run.py, so a
# supervisor that hangs without ever launching the bot reads as unhealthy. That
# makes the start period the thing that has to be generous: run.py shells out to
# sync_ats_companies.py first, and on a fresh checkout that is a network
# `git clone` of the company lists. 90s was not enough for a cold first run, and
# a container reporting unhealthy while it legitimately starts breaks anything
# gating on `condition: service_healthy`.
#
# The brackets are load-bearing, do not "tidy" them away. Docker runs the CMD
# through `sh -c`, so that shell's own command line contains the pattern, and
# `pgrep -f` scans every process including its parent. With a plain
# "src/app.py" the check matches itself and reports healthy forever -- verified:
# a container running only `pip` reported (healthy), and `pgrep -af` named the
# culprit as `1 sh -c pgrep -af "src/app.py"`. "[s]rc/app.py" is a regex that
# matches the literal src/app.py in the bot's cmdline but not the bracketed
# text in the shell's own.
HEALTHCHECK --interval=60s --timeout=10s --start-period=300s --retries=3 \
    CMD pgrep -f "[s]rc/app.py" > /dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "run.py"]
