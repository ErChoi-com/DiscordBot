"""The jba package's description has to match what the package actually is.

It began as a copy of an external aggregator and said so: "verbatim copies from
the external repo", integrated "via ats_service.py, not by importing from here
directly". Six of the seven modules are now imported directly by the bot -- 14
importers for merge_data, 8 for geo_db -- and have been rewritten around its
needs. Today's fleet-ordering work added another dependency, on
geo_priority.refresh.

That made the description worse than absent. It told the next reader that
modules the bot depends on are untouchable vendored copies, which is advice
that protects the wrong files and leaves the one genuinely vendored module
(scraper.py, whose stale "TODO - Add Workable" is upstream's) looking like
live code with an open task.

So the two lists are pinned. A module moving from vendored to adopted without
the description following is exactly how the last one stopped being true.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "services" / "jba"
SEARCH_DIRS = (REPO / "src", REPO / "scripts")

sys.path.insert(0, str(REPO / "src"))


def _doc() -> str:
    return ast.get_docstring(ast.parse((PKG / "__init__.py").read_text(encoding="utf-8"))) or ""


def _section(name: str) -> set[str]:
    """Module names listed under one heading of the package docstring."""
    doc = _doc()
    start = doc.index(f"{name} --")
    rest = doc[start + len(name):]
    other = min((i for i in (rest.find("\nADOPTED"), rest.find("\nVENDORED"),
                             rest.find("\ntests/")) if i > 0), default=len(rest))
    body = rest[:other]
    return {m for m in _modules() if re.search(rf"^\s{{4}}{m}\b", body, re.M)}


def _modules() -> set[str]:
    return {p.stem for p in PKG.glob("*.py") if p.stem != "__init__"}


def _importers(module: str, *, outside_only: bool = True) -> set[str]:
    """Files that import this module, by any of the spellings in use."""
    pattern = re.compile(
        rf"jba import {module}\b|jba\.{module}\b|^from \.{module} import", re.M
    )
    found = set()
    for root in SEARCH_DIRS:
        for path in root.rglob("*.py"):
            inside = PKG in path.parents or path == PKG
            if (inside and outside_only) or path.stem == module:
                continue
            if path.name == "__init__.py":
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if pattern.search(text):
                found.add(path.relative_to(REPO).as_posix())
    return found


def _external_importers(module: str) -> set[str]:
    return _importers(module)


def _any_importers(module: str) -> set[str]:
    """Importers anywhere, package-internal ones included.

    "Adopted" is not the same as "imported from outside the package".
    archive_index is the dedup index behind merge_data and is reached only
    through it -- as owned by this repo as merge_data is, with a schema version
    that has been bumped here three times. Requiring a direct external import
    would file it as vendored and tell the next reader to leave alone a module
    this repo has been editing all along.

    It also has to hold on a checkout that does not carry every local change:
    several importers of these modules live in files this repo does not commit,
    so a rule resting only on those flips with the checkout.
    """
    return _importers(module, outside_only=False)


# ── the two lists cover the package exactly ─────────────────────────────────

def _listed_names() -> set[str]:
    """Every name the docstring lists, whether or not such a module exists.

    _section() intersects with the real files, which is what makes the other
    tests readable -- and would also let the docstring name a module that was
    deleted years ago without anything noticing. A list that describes files
    which are not there is the same defect as one that describes them wrongly.
    """
    return set(re.findall(r"^    ([a-z_]+) {2,}\S", _doc(), re.M))


def test_every_listed_name_is_a_real_module():
    assert _listed_names() <= _modules(), _listed_names() - _modules()


def test_every_module_is_classified():
    """A module in neither list is one nobody decided about."""
    assert _section("ADOPTED") | _section("VENDORED") == _modules()


def test_no_module_is_in_both_lists():
    assert not (_section("ADOPTED") & _section("VENDORED"))


# ── and each list is true ───────────────────────────────────────────────────

def test_every_adopted_module_really_is_imported():
    """"Adopted" claims something depends on it. If nothing imports it at all,
    the file is either vendored or dead, and both need saying."""
    for module in sorted(_section("ADOPTED")):
        assert _any_importers(module), (
            f"{module} is listed as adopted but nothing imports it")


def test_all_but_one_adopted_module_is_imported_by_the_bot_directly():
    """archive_index is the one reached only through merge_data. Pinned so a
    second indirect entry is a decision someone makes, not a drift."""
    indirect = {m for m in _section("ADOPTED") if not _external_importers(m)}
    assert indirect <= {"archive_index"}, indirect


def test_no_vendored_module_is_imported_by_the_bot():
    """The failure being fixed, in the direction it actually happened: a file
    described as an untouched copy that the bot has quietly come to depend on.
    """
    for module in sorted(_section("VENDORED")):
        importers = _external_importers(module)
        assert not importers, (
            f"{module} is described as vendored but is imported by "
            f"{sorted(importers)} -- adopt it in the docstring or stop importing it")


def test_geo_priority_is_adopted_because_the_scrape_loop_depends_on_it():
    """Named specifically: the fleet ordering reads it every cycle and the
    daily rebuild writes it, so treating it as an untouchable copy would strand
    the one module whose staleness silently costs coverage.
    """
    assert "geo_priority" in _section("ADOPTED")
    assert any("manager" in f for f in _external_importers("geo_priority"))


def test_scraper_is_the_vendored_one():
    assert _section("VENDORED") == {"scraper"}


# ── the stale upstream TODO is explained, not inherited ─────────────────────

def test_the_upstream_todo_is_accounted_for():
    """`# TODO - Add Workable` sits in scraper.py and reads as an open item in
    this repo. It is not: ats_service has scraped Workable throughout. The
    docstring has to say so, or the audit trips over it again every pass.
    """
    assert "TODO - Add Workable" in (PKG / "scraper.py").read_text(encoding="utf-8")
    doc = _doc()
    assert "TODO - Add Workable" in doc
    assert "upstream" in doc


def test_workable_really_is_scraped_here():
    """The claim above is only worth making if it is true.

    Read from the platform registry under scripts/, not from ats_service: that
    module carries local work this repo does not commit, so a checkout of it
    can hold an older and shorter platform list. A test that calls the
    docstring wrong because the checkout is partial is a test that gets
    switched off.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import validate_ats_slugs

    names = {p if isinstance(p, str) else getattr(p, "name", p)
             for p in validate_ats_slugs.PLATFORMS}
    assert "workable" in names


# ── the description no longer says the thing that was wrong ─────────────────

def test_the_docstring_does_not_still_claim_the_package_is_verbatim():
    doc = _doc().lower()
    assert "files in this package are verbatim copies" not in doc


def test_the_docstring_still_credits_where_it_came_from():
    """Correcting what the package became must not drop the attribution."""
    assert "github.com/Feashliaa/job-board-aggregator" in _doc()
