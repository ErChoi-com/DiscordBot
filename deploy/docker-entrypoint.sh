#!/usr/bin/env bash
# Container preflight, then exec the real command.
#
# Everything here is a container-specific correction; none of it belongs in the
# app, which is why it lives at the boundary. In order:
#
#   1. Decide whether this container is about to BE the bot, or is a one-off
#      `docker compose run`. Almost everything below depends on that.
#   2. For the bot only: refuse to start without a token.
#   3. Always: drop Windows-shaped interpreter overrides a host-authored .env
#      leaks in.
#   4. For the bot only: clear the runtime lock and Chrome singleton left by a
#      previous container.
#   5. Always: warn about the two generated artifacts that are not in git.
set -euo pipefail

cd /app

# Matched as whole argv tokens, not as a substring of the joined command: a
# `*run.py*` glob would also fire on something like scripts/rerun.py.
_is_bot_command() {
    for arg in "$@"; do
        case "$arg" in
            run.py | ./run.py | /app/run.py | src/app.py | ./src/app.py | /app/src/app.py)
                return 0 ;;
        esac
    done
    return 1
}

if _is_bot_command "$@"; then
    IS_BOT=1
else
    IS_BOT=0
    echo "[entrypoint] One-off command; leaving the runtime lock and browser profile alone."
fi

# Gated on the bot, because the point of this check is to fail loudly instead of
# restart-looping -- which only applies to something that restarts. The one-off
# commands this image is documented for (sync_ats_companies.py,
# scripts/build_geo_db.py, deploy/docker_smoke.py) need no Discord token, and
# they are run *before* the first start, exactly when .env may not exist yet.
if [ "$IS_BOT" = 1 ]; then
    if [ ! -f /app/.env ] && [ -z "${discordtoken:-}" ] && [ -z "${DISCORD_TOKEN:-}" ]; then
        echo "[entrypoint] No /app/.env and no discordtoken in the environment." >&2
        echo "[entrypoint] Mount your .env (docker-compose.yml does) or pass discordtoken=..." >&2
        exit 1
    fi
fi

# Not gated: a Windows path in either variable breaks a one-off jobspy call just
# as thoroughly as it breaks the bot.
#
# .env is written for Windows and mounted verbatim, so any interpreter override
# in it arrives as a path this container cannot exec. run.py's _interpreter()
# trusts PYTHON_EXE over sys.executable, and job_service checks
# JOBSPY_PYTHON_EXE before anything else -- a stray "C:\..." in either is a
# start-time crash or a silently disabled scraper. Drop the Windows-shaped ones
# and let the defaults (sys.executable, and the venv the image built) win.
for var in PYTHON_EXE JOBSPY_PYTHON_EXE; do
    eval "value=\${$var:-}"
    [ -n "$value" ] || continue
    # A drive letter, or any backslash: neither can appear in a path this
    # container could exec.
    if printf '%s' "$value" | grep -qE '^[A-Za-z]:|\\'; then
        echo "[entrypoint] Ignoring Windows path in $var: $value"
        unset "$var"
    fi
done

# Dropping the override must not also drop the image default: without this,
# a Windows JOBSPY_PYTHON_EXE in .env would leave job_service falling back to
# sys.executable, which has no jobspy, and the jobspy sites go quiet instead
# of using the venv the image built for exactly this purpose.
if [ -z "${JOBSPY_PYTHON_EXE:-}" ] && [ -x /opt/jobspy/bin/python ]; then
    export JOBSPY_PYTHON_EXE=/opt/jobspy/bin/python
fi

# Gated on the bot. `docker compose run --rm bot python ...` runs this same
# entrypoint against the same bind mount and the same bot-home volume as the
# already-running bot: clearing the lock there would delete a *live* owner's
# .bot.lock and let a second bot start, and removing SingletonLock would pull
# the running Chromium's profile lock out from under it.
if [ "$IS_BOT" = 1 ]; then
    # app.py treats the lock as held when the recorded pid is alive. That check
    # is sound on a host and wrong here: the file survives in the bind-mounted
    # repo while pids restart from 1 in a fresh PID namespace, so a stale
    # "pid": 1 reads as a live owner and the bot exits 0 forever. The namespace
    # is the guarantee that nothing else in it holds the lock, which is why you
    # must not run two bot containers over one repo.
    for stale in /app/.bot.lock /app/.bot.pid; do
        if [ -e "$stale" ]; then
            echo "[entrypoint] Clearing stale $stale from a previous container."
            rm -f "$stale"
        fi
    done

    # A leftover SingletonLock from a killed container blocks the next launch.
    # Follow REDDIT_CHROME_RUNTIME_PROFILE, which the image points off the bind
    # mount (compose may override it). The fallback is the in-repo path
    # browser_service.start() would derive -- reachable only if someone unsets
    # the variable, and it is shared with a host bot, so nothing here should
    # ever rely on it.
    RUNTIME_PROFILE="${REDDIT_CHROME_RUNTIME_PROFILE:-/app/chrome_profile_runtime}"
    rm -f "$RUNTIME_PROFILE"/Singleton* 2>/dev/null || true
fi

# Neither is in git and neither is fatal, but without them the bot starts and
# silently does nothing useful: no companies to scrape, no city/region matching.
[ -d /app/data/ats_companies ] || echo "[entrypoint] WARNING: data/ats_companies missing -- run: docker compose run --rm bot python sync_ats_companies.py"
[ -f /app/data/geo.db ] || echo "[entrypoint] WARNING: data/geo.db missing -- run: docker compose run --rm bot python sync_geonames.py --force"

exec "$@"
