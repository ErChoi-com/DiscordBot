"""The autouse fixtures in conftest.py must actually redirect real-state paths.

This guards a bug that has landed in this repo twice, and both times silently:

  * synthetic postings (``example.invalid``, ``Digy4``) written into the real job
    archive -- still present in ``data/jba/jobs/jobs.db`` and in four committed,
    pushed archive zips;
  * 7,442 synthetic ``channel_id=123`` records in ``.interaction_events.log``,
    93% of the file the live bot writes.

The shape is always the same: a module-level absolute path, a writer that
swallows ``OSError``/``Exception``, and no seam to patch. Nothing fails, nothing
is logged, and the damage is only visible if someone goes looking.

So this asserts the redirection itself rather than looking for damage after the
fact. It fails the moment a fixture is dropped, renamed, or a new writer is
added without one -- which is the only point at which the problem is cheap.

Note ``merge_data.log_jobs`` is protected differently: test_watcher_dedup_
resilience stubs the *function* rather than redirecting ``_JOBS_DIR``, so its
path constant legitimately still points at the real archive here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _assert_redirected(value: Path, label: str) -> None:
    resolved = Path(value).resolve()
    assert not resolved.is_relative_to(REPO_ROOT), (
        f"{label} still points inside the repo ({resolved}). A test writing "
        "through it would pollute real state, and the writer swallows errors, "
        "so nothing would report it. Add it to the autouse fixture in "
        "tests/conftest.py."
    )


def test_interaction_event_log_is_redirected():
    from ui import views

    _assert_redirected(views.INTERACTION_EVENTS_PATH, "views.INTERACTION_EVENTS_PATH")


@pytest.mark.parametrize(
    "attr",
    [
        "STRUCTURED_CACHE_ROOT",
        "STRUCTURED_SELECTION_CACHE_DIR",
        "STRUCTURED_TELEMETRY_PATH",
        "SCRAPE_TELEMETRY_PATH",
    ],
)
def test_listing_state_paths_are_redirected(attr: str):
    from services.resumes import listing

    _assert_redirected(getattr(listing, attr), f"listing.{attr}")


def test_cover_telemetry_path_is_redirected():
    from services.resumes import cover

    _assert_redirected(cover.COVER_TELEMETRY_PATH, "cover.COVER_TELEMETRY_PATH")


def test_writers_actually_honour_the_redirect():
    """Asserting the constants is not enough -- a writer that re-derives its own
    path from ``__file__`` would pass the checks above and still write for real.
    So drive the real writers and confirm the real files do not move."""
    from services.resumes import listing
    from ui import views

    real_scrape = REPO_ROOT / ".resume_cache" / "scrape_events.jsonl"
    real_events = REPO_ROOT / ".interaction_events.log"
    before = (
        real_scrape.stat().st_size if real_scrape.exists() else 0,
        real_events.stat().st_size if real_events.exists() else 0,
    )

    listing._record_scrape_outcome("https://example.invalid/isolation-probe", "probe", 0.0)
    views.log_interaction_event("isolation.probe", channel_id=999999)

    after = (
        real_scrape.stat().st_size if real_scrape.exists() else 0,
        real_events.stat().st_size if real_events.exists() else 0,
    )
    assert after == before, "a real state file grew while the fixtures were active"

    # ...and confirm the writes landed somewhere, so this cannot pass by the
    # writers having silently become no-ops.
    assert listing.SCRAPE_TELEMETRY_PATH.exists()
    assert views.INTERACTION_EVENTS_PATH.exists()
