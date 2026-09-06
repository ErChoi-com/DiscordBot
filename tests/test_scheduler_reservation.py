"""Reserved interactive capacity, and the observability that proves it.

These cover the failure that motivated the reservation: a pool fully occupied
by long background scrapes left a user's .resumebuild queued indefinitely,
because priority decides who goes NEXT, not who gets a thread.
"""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.priority_scheduler import BACKGROUND, INTERACTIVE, PriorityWorkScheduler


def _await_active(sched: PriorityWorkScheduler, expected: int, timeout: float = 5.0) -> int:
    """Block until `active` reaches `expected`, then return it (or the last
    value seen, so a failing assertion can report what actually happened)."""
    deadline = time.monotonic() + timeout
    active = -1
    while time.monotonic() < deadline:
        active = sched.stats()["active"]
        if active == expected:
            return active
        time.sleep(0.01)
    return active


def _await_stat(sched: PriorityWorkScheduler, key: str, minimum: int, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    value = -1
    while time.monotonic() < deadline:
        value = sched.stats()[key]
        if value >= minimum:
            return value
        time.sleep(0.01)
    return value


@pytest.fixture
def blocking_work():
    """A callable that parks on a shared event, plus the release switch.

    Released in teardown so a failed assertion can never leave worker threads
    parked for the full 10s timeout.
    """
    release = threading.Event()

    def work() -> str:
        release.wait(timeout=10)
        return "done"

    yield work, release
    release.set()


# ── Reserved interactive capacity ───────────────────────────────────────────


def test_interactive_work_runs_while_background_saturates_every_other_worker(blocking_work):
    work, release = blocking_work
    # 3 workers, 1 reserved. Six background tasks cannot fill more than the 2
    # general workers, so the reserved one stays free for the command the
    # user is actually waiting on.
    sched = PriorityWorkScheduler(max_workers=3, reserved_interactive=1)
    try:
        for _ in range(6):
            sched.submit(work, tier=BACKGROUND, cost=1.0)

        assert _await_active(sched, 2) == 2

        stats = sched.stats()
        assert stats["queued"] == 4, f"other four must still be queued: {stats}"
        assert stats["reserved_interactive"] == 1

        # The reserved worker is idle and refuses background work, so this
        # runs immediately instead of waiting out the blocked tasks.
        interactive = sched.submit(lambda: "interactive", tier=INTERACTIVE, cost=1.0)
        assert interactive.result(timeout=2) == "interactive"

        # Background is still parked; the command did not need it to finish.
        assert not release.is_set()
        assert sched.stats()["queued"] == 4
    finally:
        release.set()
        sched.shutdown()


def test_without_a_reservation_interactive_work_starves_behind_background(blocking_work):
    """Negative control for the test above, and a reproduction of the outage.

    Same pool, same submissions, reservation switched off. Priority ordering
    is untouched and still correct -- the interactive task sits at the head of
    the queue the entire time -- and it still never runs, because every worker
    is inside a background task that has no deadline it can be made to honour.
    That is the whole bug: being next in line is worthless when nothing ever
    yields a thread.
    """
    work, release = blocking_work
    sched = PriorityWorkScheduler(max_workers=2, reserved_interactive=0)
    try:
        for _ in range(4):
            sched.submit(work, tier=BACKGROUND, cost=1.0)

        assert _await_active(sched, 2) == 2

        interactive = sched.submit(lambda: "interactive", tier=INTERACTIVE, cost=1.0)
        with pytest.raises(FuturesTimeoutError):
            interactive.result(timeout=0.5)

        # It is not mis-ranked, merely unreachable: it is first in the queue.
        stats = sched.stats()
        assert stats["queued_interactive"] == 1, stats
        assert stats["active"] == 2, stats

        # Releasing the background work is the only thing that frees it.
        release.set()
        assert interactive.result(timeout=5) == "interactive"
    finally:
        release.set()
        sched.shutdown()


def test_reserved_worker_refuses_background_work_even_when_queue_is_backed_up(blocking_work):
    work, release = blocking_work
    # The reservation only means something if the reserved worker declines to
    # help with background work; otherwise it is just another general worker.
    sched = PriorityWorkScheduler(max_workers=2, reserved_interactive=1)
    try:
        for _ in range(4):
            sched.submit(work, tier=BACKGROUND, cost=1.0)

        assert _await_active(sched, 1) == 1
        time.sleep(0.15)  # give a misbehaving reserved worker time to grab one

        stats = sched.stats()
        assert stats["active"] == 1, f"reserved worker took background work: {stats}"
        assert stats["queued"] == 3
    finally:
        release.set()
        sched.shutdown()


def test_interactive_work_still_spills_into_idle_general_workers():
    """The reservation is a floor, not a partition. More interactive work than
    the floor holds must still use the rest of the pool rather than queue."""
    sched = PriorityWorkScheduler(max_workers=3, reserved_interactive=1)
    try:
        release = threading.Event()

        def work() -> str:
            release.wait(timeout=10)
            return "done"

        for _ in range(3):
            sched.submit(work, tier=INTERACTIVE, cost=1.0)

        # All three workers -- reserved and general -- take interactive work.
        assert _await_active(sched, 3) == 3
        assert sched.stats()["queued"] == 0
    finally:
        release.set()
        sched.shutdown()


def test_aged_background_work_cannot_occupy_the_interactive_reservation(blocking_work):
    """The regression that made the reservation necessary AND defeated it.

    Aging used to promote starved background tasks straight to INTERACTIVE.
    With a flooded queue every background task crossed the threshold, so the
    whole backlog became "interactive" and would have been eligible for the
    reserved workers -- restoring the original starvation with extra steps.
    Aging now targets PROMOTED, which outranks BACKGROUND but is not
    reservation-eligible.
    """
    from services import priority_scheduler as ps

    work, release = blocking_work
    original_threshold = ps._AGING_THRESHOLD_SECONDS
    original_interval = ps._AGING_SWEEP_INTERVAL_SECONDS
    ps._AGING_THRESHOLD_SECONDS = 0.05
    ps._AGING_SWEEP_INTERVAL_SECONDS = 0.02
    sched = PriorityWorkScheduler(max_workers=2, reserved_interactive=1)
    try:
        for _ in range(4):
            sched.submit(work, tier=BACKGROUND, cost=1.0)

        assert _await_stat(sched, "queued_promoted", 3) >= 3, f"aging did not run: {sched.stats()}"

        stats = sched.stats()
        assert stats["queued_interactive"] == 0, (
            f"aged background work is masquerading as interactive: {stats}"
        )
        assert stats["active"] == 1, f"reserved worker must still hold its slot open: {stats}"

        # And the reservation still works with the aged backlog in place.
        interactive = sched.submit(lambda: "interactive", tier=INTERACTIVE, cost=1.0)
        assert interactive.result(timeout=2) == "interactive"
    finally:
        release.set()
        sched.shutdown()
        ps._AGING_THRESHOLD_SECONDS = original_threshold
        ps._AGING_SWEEP_INTERVAL_SECONDS = original_interval


def test_promoted_work_still_outranks_plain_background():
    """Aging must keep doing its job: a starved task jumps its untouched
    peers. Only its access to the reservation was withdrawn."""
    from services import priority_scheduler as ps

    original_threshold = ps._AGING_THRESHOLD_SECONDS
    original_interval = ps._AGING_SWEEP_INTERVAL_SECONDS
    ps._AGING_THRESHOLD_SECONDS = 0.05
    ps._AGING_SWEEP_INTERVAL_SECONDS = 0.02
    # No reservation here: this is purely about queue ordering.
    sched = PriorityWorkScheduler(max_workers=1, reserved_interactive=0)
    try:
        order: list[str] = []
        lock = threading.Lock()
        release = threading.Event()

        def _blocker() -> str:
            release.wait(timeout=10)
            return "blocker"

        def _record(tag: str) -> str:
            with lock:
                order.append(tag)
            return tag

        sched.submit(_blocker, tier=BACKGROUND, cost=1.0)
        assert _await_active(sched, 1) == 1

        # `old` ages while we wait; `fresh` arrives after the sweep and is
        # deliberately cheaper, so cost-ordering alone would put it first.
        old = sched.submit(_record, "old", tier=BACKGROUND, cost=9.0)
        assert _await_stat(sched, "queued_promoted", 1) == 1

        fresh = sched.submit(_record, "fresh", tier=BACKGROUND, cost=0.1)

        release.set()
        old.result(timeout=5)
        fresh.result(timeout=5)

        assert order == ["old", "fresh"], f"aging lost its ordering benefit: {order}"
    finally:
        release.set()
        sched.shutdown()
        ps._AGING_THRESHOLD_SECONDS = original_threshold
        ps._AGING_SWEEP_INTERVAL_SECONDS = original_interval


def test_reservation_never_consumes_the_whole_pool():
    # A reservation larger than the pool would leave background work with no
    # worker at all -- the opposite failure.
    sched = PriorityWorkScheduler(max_workers=4, reserved_interactive=99)
    try:
        assert sched.stats()["reserved_interactive"] == 3
    finally:
        sched.shutdown()

    # Default sizing scales with the pool rather than being a fixed number.
    sized = PriorityWorkScheduler(max_workers=12)
    try:
        assert sized.stats()["reserved_interactive"] == 2
    finally:
        sized.shutdown()


# ── Observability ───────────────────────────────────────────────────────────


def test_dropped_tasks_are_counted_and_not_reported_as_completed():
    """A cancelled-while-queued task is discarded on pickup. That used to
    increment nothing, which is why the live counters looked impossible:
    work submitted, nothing completed, nothing failed, no error anywhere."""
    sched = PriorityWorkScheduler(max_workers=1, reserved_interactive=0)
    try:
        ran = threading.Event()
        release = threading.Event()

        def _blocker() -> str:
            release.wait(timeout=10)
            return "blocker"

        blocker = sched.submit(_blocker, tier=BACKGROUND, cost=1.0)
        assert _await_active(sched, 1) == 1

        doomed = sched.submit(ran.set, tier=BACKGROUND, cost=1.0)
        assert doomed.cancel() is True  # still queued, so cancel wins

        assert sched.stats()["dropped"] == 0, "nothing dropped until a worker picks it up"

        release.set()
        blocker.result(timeout=5)

        assert _await_stat(sched, "dropped", 1) == 1, f"discard went uncounted: {sched.stats()}"
        stats = sched.stats()
        assert stats["completed"] == 1, f"only the blocker actually completed: {stats}"
        assert not ran.is_set(), "the cancelled task must never have executed"
    finally:
        release.set()
        sched.shutdown()


def test_in_flight_reports_what_is_holding_each_worker_and_for_how_long():
    sched = PriorityWorkScheduler(max_workers=2, reserved_interactive=0)
    try:
        release = threading.Event()

        def _held() -> str:
            release.wait(timeout=10)
            return "held"

        future = sched.submit(_held, tier=BACKGROUND, cost=1.0, label="ats_scrape:workable")
        assert _await_active(sched, 1) == 1

        time.sleep(0.05)
        in_flight = sched.stats()["in_flight"]
        assert len(in_flight) == 1, f"expected one running task: {in_flight}"
        assert in_flight[0]["label"] == "ats_scrape:workable"
        assert in_flight[0]["age_seconds"] >= 0.04, in_flight

        release.set()
        future.result(timeout=5)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and sched.stats()["in_flight"]:
            time.sleep(0.01)
        assert sched.stats()["in_flight"] == [], "finished work must leave the in-flight list"
    finally:
        release.set()
        sched.shutdown()


def test_in_flight_is_ordered_oldest_first():
    """The oldest runner is the one worth naming when the pool is wedged."""
    sched = PriorityWorkScheduler(max_workers=3, reserved_interactive=0)
    try:
        release = threading.Event()

        def _held() -> str:
            release.wait(timeout=10)
            return "held"

        sched.submit(_held, tier=BACKGROUND, cost=1.0, label="old")
        assert _await_active(sched, 1) == 1
        time.sleep(0.08)
        sched.submit(_held, tier=BACKGROUND, cost=1.0, label="new")
        assert _await_active(sched, 2) == 2

        labels = [entry["label"] for entry in sched.stats()["in_flight"]]
        assert labels == ["old", "new"], labels
    finally:
        release.set()
        sched.shutdown()
