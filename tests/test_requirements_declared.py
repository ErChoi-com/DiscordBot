"""Every third-party module ``src/`` imports must be declared in a requirements file.

This exists because ``pypdf`` was not. It is imported by
``resumes/ats_check.py`` and had been working purely because it was installed
by hand on the developer's machine -- and its absence is silent by design:
``audit_pdf_ats`` catches the ImportError and returns ``status="unknown"``,
which callers are documented to treat as pass-through. A clean install
therefore lost the Type 3 bitmap gate and the lmodern salvage it drives,
with nothing in the logs to say so.

An import that happens to be installed locally is invisible until something
builds from scratch -- a container, CI, or a new checkout. This test is that
scratch build, minus the build.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
REQUIREMENTS = (ROOT / "requirements.txt", ROOT / "requirements-semantic.txt")

# Import name -> distribution name, where they differ. Anything not listed is
# assumed to import under its own distribution name.
IMPORT_TO_DISTRIBUTION = {
    "bs4": "beautifulsoup4",
    "discord": "discord.py",
    "google": "google-genai",
    "jobspy": "python-jobspy",
    "sentence_transformers": "sentence-transformers",
}

# Imported by the generated scrape script rather than by src/ itself, or
# resolved through a sibling path insert rather than as a distribution.
NOT_DISTRIBUTIONS = {"geo_db", "geolocation"}


def _declared_distributions() -> set[str]:
    names: set[str] = set()
    for path in REQUIREMENTS:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            # Strip version specifiers and extras: "python-jobspy>=1.1.82".
            for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "["):
                line = line.split(sep, 1)[0]
            names.add(line.strip().lower())
    return names


def _local_module_names() -> set[str]:
    names = {p.stem for p in SRC.glob("*.py")}
    names |= {d.name for d in SRC.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))}
    return names


def _third_party_imports() -> dict[str, set[str]]:
    """Top-level module name -> the files importing it."""
    stdlib = set(sys.stdlib_module_names)
    local = _local_module_names()
    found: dict[str, set[str]] = {}
    for file in SRC.rglob("*.py"):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a parse failure is its own bug
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import: always local.
                modules = [node.module] if (node.module and not node.level) else []
            else:
                continue
            for module in modules:
                top = module.split(".", 1)[0]
                if top in stdlib or top in local or top.startswith("_"):
                    continue
                found.setdefault(top, set()).add(str(file.relative_to(ROOT)))
    return found


def test_src_imports_are_all_declared():
    declared = _declared_distributions()
    assert declared, "no requirements parsed -- the parser, not the deps, is broken"

    undeclared: list[str] = []
    for module, files in sorted(_third_party_imports().items()):
        if module in NOT_DISTRIBUTIONS:
            continue
        distribution = IMPORT_TO_DISTRIBUTION.get(module, module).lower()
        if distribution not in declared:
            sample = sorted(files)[0]
            undeclared.append(f"{module} (as {distribution}), imported by {sample}")

    assert not undeclared, (
        "third-party imports missing from requirements*.txt:\n  "
        + "\n  ".join(undeclared)
        + "\nThese work only where they happen to be installed already."
    )


def test_pypdf_specifically_is_declared():
    """The regression this file was written for.

    Called out separately because its failure mode is silent rather than an
    ImportError: without pypdf the ATS audit degrades to "unknown" and every
    caller treats that as a pass.
    """
    assert "pypdf" in _declared_distributions(), (
        "pypdf is undeclared; audit_pdf_ats will return 'unknown' and the "
        "Type 3 bitmap gate silently stops running"
    )


@pytest.mark.parametrize("path", REQUIREMENTS, ids=lambda p: p.name)
def test_requirements_files_exist(path: Path):
    assert path.is_file(), f"{path} is referenced by the Dockerfile and by this test"
