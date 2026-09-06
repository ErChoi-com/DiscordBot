"""Invariants of the container setup that were each a real bug at some point.

Dockerfile and docker-compose.yml have no other test coverage, and every
assertion here corresponds to something that was actually broken and diagnosed
by running it -- not to a style preference:

  * the healthcheck matched its own wrapper shell and reported healthy forever,
    including with a dead bot;
  * `restart: unless-stopped` turned a missing token into an endless restart
    loop, the exact failure deploy/discordbot.service is written to avoid;
  * the Chrome runtime profile default lived only in compose, so a bare
    `docker run` put it back on the bind mount and clobbered a live bot's
    browser profile;
  * `.dockerignore` did not exclude resumes_cache/, so real resume profiles
    were baked into an image layer;
  * `env_file: - .env` made every compose command fail on a checkout that has
    no .env yet -- which is every fresh clone, and exactly when the documented
    first-run commands are supposed to run.

These are cheap static checks: no Docker daemon, no build.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
COMPOSE = ROOT / "docker-compose.yml"
DOCKERIGNORE = ROOT / ".dockerignore"
ENTRYPOINT = ROOT / "deploy" / "docker-entrypoint.sh"

pytestmark = pytest.mark.skipif(
    not DOCKERFILE.exists(), reason="container setup not present in this checkout"
)


def _dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def _compose() -> str:
    return COMPOSE.read_text(encoding="utf-8")


# ── healthcheck ──────────────────────────────────────────────────────────────

def test_healthcheck_pattern_cannot_match_its_own_shell():
    """`pgrep -f "src/app.py"` matches the sh -c wrapper running it.

    Docker runs HEALTHCHECK CMD through a shell whose command line contains the
    pattern, and pgrep -f scans every process. With the plain string the check
    reported healthy from a container running only `pip`. The bracket form is a
    regex that matches the bot's real cmdline but not the bracketed literal in
    the shell's own.
    """
    # Inspect the HEALTHCHECK instruction itself, not the whole file: the
    # comment above it necessarily mentions both forms, and an earlier version
    # of this test passed on that comment alone while the instruction was wrong.
    lines = [
        ln for ln in _dockerfile().splitlines()
        if ln.strip().startswith("HEALTHCHECK") or ln.strip().startswith("CMD pgrep")
    ]
    assert lines, "no HEALTHCHECK instruction found"
    instruction = " ".join(lines)
    assert "[s]rc/app.py" in instruction, "healthcheck must use the bracketed pattern"
    assert 'pgrep -f "src/app.py"' not in instruction, (
        "plain pattern self-matches its own sh -c wrapper: the healthcheck would "
        "report healthy forever, including with a dead bot"
    )


# ── restart policy ───────────────────────────────────────────────────────────

def test_restart_policy_does_not_loop_on_config_errors():
    """app.py exits 0 when another instance holds the lock and 1 on a missing
    token. Measured under `unless-stopped`: 8 restarts in 14 seconds, forever."""
    compose = _compose()
    assert "restart: on-failure" in compose
    for bad in ("restart: unless-stopped", "restart: always"):
        assert bad not in compose, f"{bad} restarts on exit 0 and on config errors"


# ── Chrome profile placement ─────────────────────────────────────────────────

def test_chrome_runtime_profile_default_is_in_the_image():
    """Not only in compose: a bare `docker run` inherits the image default, and
    the in-repo fallback is shared with any bot already running on that
    checkout."""
    assert "REDDIT_CHROME_RUNTIME_PROFILE=/home/bot/" in _dockerfile(), (
        "the safe default must not depend on which launcher is used"
    )


# ── privacy ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "pattern",
    [
        ".env",
        "src/services/resumes/resumes_cache/",
        "chrome_profile/",
        "vpn/",
    ],
)
def test_dockerignore_keeps_secrets_and_personal_data_out_of_layers(pattern: str):
    """An image layer is easier to pass around than a repo; anything gitignored
    as private must not be baked in."""
    lines = {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    # Either form is acceptable: `**/x` matches zero or more leading directories,
    # so it covers the root-level file too. Depth-independence is asserted
    # separately by test_dockerignore_patterns_are_depth_independent.
    assert pattern in lines or f"**/{pattern}" in lines, (
        f".dockerignore must exclude {pattern} (bare or **/-prefixed)"
    )


@pytest.mark.parametrize(
    "pattern",
    ["**/__pycache__/", "**/*.py[cod]", "**/*.log", "**/*.pem", "**/*.key", "**/.env"],
)
def test_dockerignore_patterns_are_depth_independent(pattern: str):
    """Docker matches a pattern against the file's FULL relative path.

    So a bare `*.log` or `__pycache__/` only ever matches at the context root
    and silently ships everything nested. That is not theoretical: before the
    `**/` prefixes were added this image carried 237 nested .pyc files and a
    nested src/.panel_send_events.log, while .dockerignore appeared to exclude
    both. The same hole would have shipped a nested *.pem or *.key.
    """
    lines = {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    bare = pattern[len("**/"):]
    assert pattern in lines, (
        f"{pattern} must carry the **/ prefix; bare {bare!r} matches only at the "
        "context root"
    )
    assert bare not in lines, (
        f"bare {bare!r} is still present and matches only the root -- remove it "
        f"in favour of {pattern}"
    )


# ── compose usability on a fresh clone ───────────────────────────────────────

def test_env_file_is_optional():
    """.env is gitignored. With the short form, a missing file makes Compose
    refuse everything -- including `docker compose build` -- so the documented
    first-run commands could not be run on a fresh clone.

    Parsed rather than grepped: the comment above the setting explains the fix
    and contains the words "required: false", which is enough to satisfy a
    substring check while the setting itself is back to the broken short form.
    """
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load(_compose())
    entries = spec["services"]["bot"]["env_file"]
    assert isinstance(entries, list) and entries, "env_file must be a non-empty list"
    for entry in entries:
        assert isinstance(entry, dict), (
            f"short form {entry!r} makes a missing .env fatal to every compose command"
        )
        assert entry.get("required") is False, f"{entry} must set required: false"


# ── entrypoint gating ────────────────────────────────────────────────────────

def test_entrypoint_only_touches_shared_state_for_the_bot():
    """`docker compose run` shares the bind mount and bot-home volume with a
    running bot. Clearing the lock or SingletonLock for a one-off command
    deletes a live owner's state."""
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "_is_bot_command" in text
    # Whole-token matching, so scripts/rerun.py does not count as run.py.
    # Comments are stripped first: the file *documents* why the glob is wrong,
    # and matching that text was a false positive when this test was written.
    code = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    assert "*run.py*" not in code, "substring matching would fire on e.g. rerun.py"


def test_entrypoint_runs_as_non_root():
    assert "USER bot" in _dockerfile()


# ── documentation drift ──────────────────────────────────────────────────────

DOC = ROOT / "deploy" / "DOCKER.md"


@pytest.mark.skipif(not DOC.exists(), reason="DOCKER.md not present")
@pytest.mark.parametrize(
    "claim,key",
    [
        ("restart: on-failure:5", "restart"),
        ("shm_size: 1gb", "shm_size"),
        ("stop_grace_period: 300s", "stop_grace_period"),
        ("init: true", "init"),
    ],
)
def test_docker_md_describes_the_settings_that_are_actually_set(claim: str, key: str):
    """DOCKER.md explains each setting and its measured justification.

    Rationale is duplicated across the Dockerfile, compose file, entrypoint and
    this doc, which is how two claims survived after being measured false --
    "~2GB TeX" (it is ~3.6GB) and "Chromium cannot sandbox in a container" (it
    can). Numbers are not asserted here, since they legitimately move; the
    settings are, since a doc describing a setting that is not there is worse
    than no doc.
    """
    yaml = pytest.importorskip("yaml")
    doc = DOC.read_text(encoding="utf-8")
    if claim not in doc:
        pytest.skip(f"DOCKER.md does not state {claim!r}")
    bot = yaml.safe_load(_compose())["services"]["bot"]
    actual = bot.get(key)
    expected = claim.split(": ", 1)[1]
    normalised = {
        "1gb": {"1gb", "1073741824"},
        "300s": {"300s", "5m0s"},
        "true": {"True", "true"},
    }.get(expected, {expected})
    assert str(actual) in normalised, (
        f"DOCKER.md documents `{claim}` but docker-compose.yml has {key}={actual!r}"
    )


# ── CI preflight completeness ────────────────────────────────────────────────

WORKFLOW = ROOT / ".github" / "workflows" / "docker-image.yml"


@pytest.mark.skipif(not WORKFLOW.exists(), reason="docker workflow not present")
def test_ci_preflight_names_every_untracked_module_the_container_needs():
    """The preflight list must not be hand-maintained into staleness.

    A checkout missing one of these builds a fine image whose CMD or imports do
    not exist, and the failure surfaces as "can't open file '/app/run.py'" or an
    ImportError -- neither pointing at the cause. The first hand-written version
    of that list missed three modules, which is why this test exists.

    Walks imports from the real entry points and asserts anything reachable but
    untracked is named in the workflow.
    """
    import ast
    import subprocess

    src = ROOT / "src"
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True
        ).stdout.split()
    )
    if not tracked:
        pytest.skip("not a git checkout")

    mod_file = {
        ".".join(p.relative_to(src).with_suffix("").parts): p
        for p in src.rglob("*.py")
    }

    def imports_of(path):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return set()
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                out.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                out.add(node.module)
                out.update(f"{node.module}.{a.name}" for a in node.names)
        return out

    seen: set = set()
    queue = [ROOT / "run.py", src / "app.py", ROOT / "deploy" / "docker_smoke.py"]
    while queue:
        f = queue.pop()
        if f in seen or not f.exists():
            continue
        seen.add(f)
        queue.extend(
            mod_file[n] for n in imports_of(f) if n in mod_file and mod_file[n] not in seen
        )

    workflow = WORKFLOW.read_text(encoding="utf-8")
    missing = [
        f.relative_to(ROOT).as_posix()
        for f in sorted(seen)
        if f.relative_to(ROOT).as_posix() not in tracked
        and f.relative_to(ROOT).as_posix() not in workflow
    ]
    assert not missing, (
        "reachable from the container's entry points, absent from a fresh clone, "
        f"and not named in the CI preflight: {missing}"
    )
