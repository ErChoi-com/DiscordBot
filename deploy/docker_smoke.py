"""Prove the container can actually do the things it shells out to.

    docker compose run --rm bot python deploy/docker_smoke.py

Import checks alone would pass on an image where Chromium cannot launch and
pdflatex silently emits bitmaps, which are the two failures that matter here.
So every check below runs the real thing and inspects the real output.

Exits non-zero if any REQUIRED check fails; optional checks report separately.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FAILED: list[str] = []
WARNED: list[str] = []


def check(name: str, required: bool = True):
    def decorator(fn):
        print(f"\n=== {name} ===", flush=True)
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - a smoke test reports, never raises
            (FAILED if required else WARNED).append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  {'FAIL' if required else 'WARN'}: {type(exc).__name__}: {exc}")
            return fn
        print(f"  OK{': ' + detail if detail else ''}")
        return fn
    return decorator


# -- interpreter and third-party imports -------------------------------------

@check("python environment")
def _python() -> str:
    return f"{sys.version.split()[0]} at {sys.executable}"


@check("third-party imports")
def _imports() -> str:
    # Exactly the third-party top-level imports under src/, minus the two
    # optional ones (jobspy lives in its own venv, sentence_transformers is the
    # WITH_SEMANTIC extra).
    mods = ["discord", "requests", "curl_cffi", "bs4", "ftfy", "pycountry",
            "playwright", "pypdf", "google.genai"]
    missing = []
    for m in mods:
        try:
            __import__(m)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{m} ({type(exc).__name__})")
    if missing:
        raise RuntimeError("cannot import: " + ", ".join(missing))
    return f"{len(mods)} modules"


@check("bot modules import")
def _bot_modules() -> str:
    import config  # noqa: F401
    from services import platform_support
    from services.resumes import ats_check  # noqa: F401
    return f"platform={platform_support.current_system()}"


# -- the things the container exists to provide ------------------------------

@check("chromium discovery")
def _chrome_found() -> str:
    from services import platform_support
    path = platform_support.find_chrome()
    if not path:
        raise RuntimeError("platform_support.find_chrome() returned None")
    # Report the sandbox posture; do not require it to be disabled.
    #
    # This used to fail when CHROME_NO_SANDBOX was unset, on the old assumption
    # that Chromium cannot sandbox in a container. Measured in this image on
    # Docker 29: it launches and renders with the flag, without it, and without
    # it under no-new-privileges. The daemon's default seccomp profile permits
    # the unprivileged user namespace that the namespace sandbox needs, so the
    # flag is a compatibility escape hatch rather than a requirement -- and
    # requiring it here would have enforced an unnecessary security regression.
    #
    # The real requirement is the next check: that Chromium actually launches.
    args = platform_support.chrome_sandbox_args()
    posture = "sandbox DISABLED (--no-sandbox)" if "--no-sandbox" in args else "sandboxed"
    return f"{path}, {posture}, args {args}"


@check("chromium actually launches")
def _chrome_launch() -> str:
    from playwright.sync_api import sync_playwright

    from services import platform_support
    with tempfile.TemporaryDirectory() as profile:
        with sync_playwright() as pw:
            ctx = pw.chromium.launch_persistent_context(
                user_data_dir=profile,
                executable_path=platform_support.find_chrome(),
                headless=True,
                args=["--no-first-run", "--no-default-browser-check",
                      *platform_support.chrome_sandbox_args()],
            )
            try:
                page = ctx.new_page()
                # Inline content, so this proves the browser renders without
                # depending on the network or on any site being up.
                page.set_content("<title>smoke</title><h1>ok</h1>")
                title = page.title()
                text = page.inner_text("h1")
            finally:
                ctx.close()
    if title != "smoke" or text != "ok":
        raise RuntimeError(f"rendered wrong content: title={title!r} text={text!r}")
    return "launched headless and rendered a page"


@check("latex engine present")
def _latex_engine() -> str:
    # Deliberately not check_template_compile_environment(): that reads the
    # resume template out of the gitignored profile directory, which a fresh
    # container does not have, so it would fail for a reason unrelated to
    # whether TeX is installed. The binaries are the thing being tested here.
    found = {name: shutil.which(name)
             for name in ("pdflatex", "xelatex", "lualatex", "kpsewhich")}
    if not found["pdflatex"]:
        raise RuntimeError(
            "pdflatex not on PATH -- built with WITH_LATEX=0? Resume builds "
            "will report 'LaTeX compile environment is not ready'."
        )
    have = [n for n, p in found.items() if p]
    return f"{found['pdflatex']} (also: {', '.join(n for n in have if n != 'pdflatex')})"


@check("latex packages the templates use", required=False)
def _latex_packages() -> str:
    # kpsewhich names exactly what is missing, where a failed compile would
    # only report whichever file it happened to hit first.
    #
    # Only meaningful inside the image. Run on a Windows host this passes
    # unconditionally, because MiKTeX installs missing packages on demand --
    # Debian's TeX Live is static, which is the whole reason to check.
    styles = [
        "lmodern", "parskip", "geometry", "hyperref", "titlesec", "enumitem",
        "fancyhdr", "tabularx", "multicol", "fullpage", "comment", "latexsym",
        "marvosym", "fontawesome5", "XCharter", "charter", "sourcesanspro",
        "roboto", "FiraSans", "CormorantGaramond", "noto-sans", "babel",
    ]
    kpsewhich = shutil.which("kpsewhich")
    if not kpsewhich:
        raise RuntimeError("kpsewhich not found (texlive-binaries missing)")
    missing = []
    for sty in styles:
        proc = subprocess.run([kpsewhich, f"{sty}.sty"],
                              capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not proc.stdout.strip():
            missing.append(sty)
    # glyphtounicode is NOT a .sty -- the templates pull it in with
    # \input{glyphtounicode}, so a .sty probe would report it missing even when
    # it is installed.
    #
    # It earns a check because the templates \input it unconditionally: if the
    # file is absent the compile fails outright, whatever else is installed.
    # Note it is not, on its own, what makes the PDF extractable -- measured on
    # an lmodern/T1 document, output is byte-identical with and without it and
    # carries a ToUnicode map either way, because pdfTeX already derives one for
    # those fonts. It supplies the glyph-name -> Unicode table \pdfgentounicode
    # falls back on where pdfTeX cannot.
    for tex in ("glyphtounicode.tex",):
        proc = subprocess.run([kpsewhich, tex], capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not proc.stdout.strip():
            missing.append(tex)

    if missing:
        raise RuntimeError(
            f"{len(missing)}/{len(styles) + 1} missing: {', '.join(missing)} "
            "(resume templates using these will fail to compile)"
        )
    return f"all {len(styles) + 1} present"


@check("chktex linter", required=False)
def _chktex() -> str:
    # Optional in the code too (`if chktex_path:` in resume.py), so a warning
    # rather than a failure -- but its absence silently weakens the lint
    # findings fed back into the resume rebuild loop.
    path = shutil.which("chktex")
    if not path:
        raise RuntimeError("chktex not installed; resume lint findings will be thinner")
    return path


@check("pdflatex produces an ATS-readable PDF")
def _latex_compile() -> str:
    # The whole point: pdflatex exits 0 while emitting Type 3 bitmaps whose
    # text extracts as glyph names. Compile for real, then judge the bytes with
    # the same audit the resume pipeline uses.
    #
    # Verified to actually fire, by compiling this same document without
    # lmodern in this image: pdflatex still exited 0, and the audit returned
    # status="unreadable" at 100% Type 3. With lmodern: "ok" at 0%.
    from services.resumes.ats_check import TYPE3_DEGRADED_RATIO, audit_pdf_ats

    # Mirrors what resumes_cache/*/template.tex actually does: lmodern for
    # scalable Type 1 text, plus glyphtounicode + \pdfgentounicode=1 for the
    # ToUnicode maps. Compiling a plainer document than the real templates
    # would test a configuration nothing ships.
    body = "\n".join([
        r"\documentclass[11pt]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{geometry}",
        r"\input{glyphtounicode}",
        r"\pdfgentounicode=1",
        r"\begin{document}",
        r"Jane Doe \\ jane@example.com \\ +1 416 555 0123",
        r"\section*{Experience}",
        ("Built and shipped container tooling. " * 40),
        r"\section*{Education}",
        ("Toronto Metropolitan University, Computer Engineering. " * 20),
        r"\section*{Skills}",
        ("Python, Docker, LaTeX, distributed systems. " * 20),
        r"\end{document}",
    ])
    with tempfile.TemporaryDirectory() as tmp:
        tex = Path(tmp) / "smoke.tex"
        tex.write_text(body, encoding="utf-8")
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "smoke.tex"],
            cwd=tmp, capture_output=True, text=True, timeout=180, check=False,
        )
        pdf = Path(tmp) / "smoke.pdf"
        if proc.returncode != 0 or not pdf.exists():
            tail = "\n".join((proc.stdout or "").splitlines()[-15:])
            raise RuntimeError(f"pdflatex exited {proc.returncode}:\n{tail}")
        pdf_bytes = pdf.read_bytes()

    audit = audit_pdf_ats(pdf_bytes, expect_text=["Jane Doe", "jane@example.com"])
    if audit.status == "unknown":
        raise RuntimeError(
            "audit returned 'unknown' -- pypdf is missing, so the Type 3 gate "
            "is not running at all"
        )
    if audit.status != "ok":
        raise RuntimeError(f"PDF is not ATS-readable: {audit.summary()}")
    if audit.type3_char_ratio >= TYPE3_DEGRADED_RATIO:
        raise RuntimeError(
            f"{audit.type3_char_ratio:.0%} of text is Type 3 bitmaps -- "
            "lmodern did not take effect"
        )
    return f"{len(pdf_bytes)} bytes, {audit.summary()}"


@check("jobspy interpreter", required=False)
def _jobspy() -> str:
    from services import job_service
    exe = job_service.jobspy_python_executable()
    if exe is None:
        raise RuntimeError(
            "no interpreter with jobspy (JOBSPY_PYTHON_EXE="
            f"{os.getenv('JOBSPY_PYTHON_EXE')!r}); jobspy-backed sites stay quiet"
        )
    sites = job_service.supported_jobspy_sites()
    if not sites:
        raise RuntimeError(f"{exe} reported no sites")
    return f"{exe} -> {len(sites)} sites"


@check("git usable on the working tree", required=False)
def _git() -> str:
    """Optional, because archive publishing is opt-in.

    JBA_ARCHIVE_GIT_COMMIT is off by default -- the systemd unit calls a service
    account with a writable .git "a liability" -- so a deployment with no .git is
    a legitimate one, not a broken image. This was a required check until a
    fresh checkout without .git failed the whole run for a feature it was never
    going to use.

    When .git *is* present the check still matters: `git status` failing there is
    the "dubious ownership" symptom, which would break the archive commit at the
    worst possible moment.
    """
    if not (ROOT / ".git").exists():
        raise RuntimeError(
            f"{ROOT}/.git absent -- fine unless you set JBA_ARCHIVE_GIT_COMMIT, "
            "which needs a git working tree to commit into"
        )
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        # "dubious ownership" lands here when safe.directory is not set.
        raise RuntimeError(f"git status failed: {(proc.stderr or '').strip()}")
    return f"{len(proc.stdout.splitlines())} changed paths"


@check("writable state paths")
def _writable() -> str:
    # The paths the *app* writes: base_dir holds .bot.lock/.bot.pid/
    # .bot_state.json, data/ takes the job archives, dedup_listings/ the
    # per-channel message listings, .resume_cache/ the resume telemetry.
    #
    # Deliberately not logs/: nothing under src/ or run.py writes there --
    # bot_console.log is produced by run.bat, the Windows launcher. Probing it
    # would create a directory the container never uses.
    import config
    cfg = config.load_config()
    targets = [
        cfg.base_dir,
        cfg.base_dir / "data",
        cfg.base_dir / "dedup_listings",
        cfg.base_dir / ".resume_cache",
    ]
    bad = []
    for t in targets:
        try:
            t.mkdir(parents=True, exist_ok=True)
            probe = t / ".smoke_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{t} ({type(exc).__name__})")
    if bad:
        raise RuntimeError("not writable: " + ", ".join(bad))
    return f"{len(targets)} paths writable"


print("\n" + "=" * 60)
if WARNED:
    print(f"{len(WARNED)} optional check(s) degraded:")
    for w in WARNED:
        print(f"  - {w}")
if FAILED:
    print(f"SMOKE FAILED: {len(FAILED)} required check(s)")
    for f in FAILED:
        print(f"  - {f}")
    raise SystemExit(1)
print("SMOKE PASSED" + (f" ({len(WARNED)} optional degraded)" if WARNED else ""))
