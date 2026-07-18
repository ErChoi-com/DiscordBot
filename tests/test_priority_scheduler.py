from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.priority_scheduler import BACKGROUND, INTERACTIVE, PriorityWorkScheduler


def _blocking(tag: str, order: list[str], duration: float = 0.05, lock: threading.Lock | None = None) -> str:
    time.sleep(duration)
    if lock is not None:
        with lock:
            order.append(tag)
    else:
        order.append(tag)
    return tag


@pytest.fixture
def scheduler():
    sched = PriorityWorkScheduler(max_workers=2)
    yield sched
    sched.shutdown(wait=True)


def test_uses_all_available_cpu_cores_by_default():
    import os

    sched = PriorityWorkScheduler()
    try:
        assert sched.stats()["workers"] >= min(4, os.cpu_count() or 4)
        assert sched.stats()["workers"] == max(4, os.cpu_count() or 4)
    finally:
        sched.shutdown()


def test_explicit_worker_count_respected():
    sched = PriorityWorkScheduler(max_workers=3)
    try:
        assert sched.stats()["workers"] == 3
    finally:
        sched.shutdown()


def test_interactive_task_preempts_queued_background_task(scheduler: PriorityWorkScheduler):
    # Single worker so ordering is deterministic: occupy it with a slow task,
    # then queue background before interactive -- interactive must still win.
    sched = PriorityWorkScheduler(max_workers=1)
    try:
        order: list[str] = []
        lock = threading.Lock()

        # Occupy the single worker so both submissions below queue up.
        blocker = sched.submit(_blocking, "blocker", order, 0.1, lock, tier=BACKGROUND, cost=1.0)
        time.sleep(0.02)  # let the blocker actually start running

        bg_future = sched.submit(_blocking, "background", order, 0.01, lock, tier=BACKGROUND, cost=1.0)
        interactive_future = sched.submit(_blocking, "interactive", order, 0.01, lock, tier=INTERACTIVE, cost=1.0)

        blocker.result(timeout=5)
        interactive_future.result(timeout=5)
        bg_future.result(timeout=5)

        assert order == ["blocker", "interactive", "background"]
    finally:
        sched.shutdown()


def test_shortest_job_first_within_same_tier():
    sched = PriorityWorkScheduler(max_workers=1)
    try:
        order: list[str] = []
        lock = threading.Lock()

        blocker = sched.submit(_blocking, "blocker", order, 0.1, lock, tier=BACKGROUND, cost=1.0)
        time.sleep(0.02)

        expensive = sched.submit(_blocking, "expensive", order, 0.01, lock, tier=BACKGROUND, cost=100.0)
        cheap = sched.submit(_blocking, "cheap", order, 0.01, lock, tier=BACKGROUND, cost=1.0)

        blocker.result(timeout=5)
        cheap.result(timeout=5)
        expensive.result(timeout=5)

        assert order == ["blocker", "cheap", "expensive"]
    finally:
        sched.shutdown()


def test_aging_promotes_starved_background_task():
    from services import priority_scheduler as ps

    original_threshold = ps._AGING_THRESHOLD_SECONDS
    original_interval = ps._AGING_SWEEP_INTERVAL_SECONDS
    ps._AGING_THRESHOLD_SECONDS = 0.15
    ps._AGING_SWEEP_INTERVAL_SECONDS = 0.05
    sched = PriorityWorkScheduler(max_workers=1)
    try:
        order: list[str] = []
        lock = threading.Lock()

        # Occupy the worker long enough for the queued background task to age past the threshold.
        blocker = sched.submit(_blocking, "blocker", order, 0.3, lock, tier=BACKGROUND, cost=1.0)
        time.sleep(0.02)
        background = sched.submit(_blocking, "background", order, 0.01, lock, tier=BACKGROUND, cost=1.0)

        # Keep submitting fresh interactive work -- without aging this would starve `background` forever.
        interactive_results = []
        for _ in range(6):
            time.sleep(0.05)
            fut = sched.submit(_blocking, "interactive", order, 0.01, lock, tier=INTERACTIVE, cost=1.0)
            interactive_results.append(fut)

        blocker.result(timeout=5)
        background.result(timeout=5)
        for fut in interactive_results:
            fut.result(timeout=5)

        assert "background" in order
        assert sched.stats()["promoted"] >= 1
    finally:
        sched.shutdown()
        ps._AGING_THRESHOLD_SECONDS = original_threshold
        ps._AGING_SWEEP_INTERVAL_SECONDS = original_interval


def test_run_is_awaitable_and_propagates_exceptions():
    import asyncio

    sched = PriorityWorkScheduler(max_workers=2)

    def _ok(x: int) -> int:
        return x * 2

    def _boom() -> None:
        raise ValueError("boom")

    async def _main():
        assert await sched.run(_ok, 21, tier=INTERACTIVE) == 42
        with pytest.raises(ValueError):
            await sched.run(_boom, tier=BACKGROUND)

    try:
        asyncio.run(_main())
    finally:
        sched.shutdown()


def test_stats_reports_queue_depth_by_tier():
    sched = PriorityWorkScheduler(max_workers=1)
    try:
        order: list[str] = []
        lock = threading.Lock()
        blocker = sched.submit(_blocking, "blocker", order, 0.2, lock, tier=BACKGROUND, cost=1.0)
        time.sleep(0.02)
        sched.submit(_blocking, "bg1", order, 0.01, lock, tier=BACKGROUND, cost=1.0)
        sched.submit(_blocking, "int1", order, 0.01, lock, tier=INTERACTIVE, cost=1.0)

        stats = sched.stats()
        assert stats["queued"] == 2
        assert stats["queued_interactive"] == 1
        assert stats["queued_background"] == 1

        blocker.result(timeout=5)
    finally:
        sched.shutdown()


def test_submit_after_shutdown_raises():
    sched = PriorityWorkScheduler(max_workers=1)
    sched.shutdown(wait=True)
    with pytest.raises(RuntimeError):
        sched.submit(lambda: None, tier=INTERACTIVE)


# ── Dynamic cost from measured duration ────────────────────────────────────


def _sleep_for(seconds: float) -> float:
    time.sleep(seconds)
    return seconds


def test_unmeasured_label_falls_back_to_default_cost():
    sched = PriorityWorkScheduler(max_workers=1)
    try:
        assert sched.sample_count("never_run") == 0
        assert sched.estimated_cost("never_run") == pytest.approx(1.0)
        assert sched.estimated_cost("never_run", default=42.0) == pytest.approx(42.0)
    finally:
        sched.shutdown()


def test_cost_derived_from_measured_duration_uses_median_not_mean():
    """One outlier run must not drag the learned cost up the way a mean would --
    this is the whole point of using the median."""
    sched = PriorityWorkScheduler(max_workers=2)
    try:
        durations = [0.01, 0.01, 0.01, 0.01, 0.2]  # one big outlier
        futures = [sched.submit(_sleep_for, d, tier=BACKGROUND, label="flaky_task") for d in durations]
        for fut in futures:
            fut.result(timeout=5)

        mean = sum(durations) / len(durations)
        median_cost = sched.estimated_cost("flaky_task")

        assert sched.sample_count("flaky_task") == len(durations)
        assert median_cost < mean
        assert median_cost == pytest.approx(0.01, abs=0.005)
    finally:
        sched.shutdown()


def test_omitted_cost_with_label_is_derived_and_orders_by_learned_cost():
    """End-to-end: submit a few 'slow_task' and 'fast_task' runs to build up
    history, then submit one more of each with no explicit cost -- the
    scheduler must have learned fast_task is cheaper and run it first."""
    sched = PriorityWorkScheduler(max_workers=1)
    try:
        order: list[str] = []
        lock = threading.Lock()

        for _ in range(3):
            sched.submit(_blocking, "slow_task_warmup", order, 0.05, lock, tier=BACKGROUND, label="slow_task").result(timeout=5)
            sched.submit(_blocking, "fast_task_warmup", order, 0.01, lock, tier=BACKGROUND, label="fast_task").result(timeout=5)

        order.clear()
        blocker = sched.submit(_blocking, "blocker", order, 0.1, lock, tier=BACKGROUND, cost=0.0)
        time.sleep(0.02)

        # Submitted slow-label first, fast-label second -- with no explicit cost,
        # dynamic costing must still run the cheaper (fast) one first.
        slow_future = sched.submit(_blocking, "slow_task", order, 0.001, lock, tier=BACKGROUND, label="slow_task")
        fast_future = sched.submit(_blocking, "fast_task", order, 0.001, lock, tier=BACKGROUND, label="fast_task")

        blocker.result(timeout=5)
        fast_future.result(timeout=5)
        slow_future.result(timeout=5)

        assert order == ["blocker", "fast_task", "slow_task"]
    finally:
        sched.shutdown()


def test_explicit_cost_overrides_learned_cost():
    sched = PriorityWorkScheduler(max_workers=2)
    try:
        sched.submit(_sleep_for, 0.01, tier=BACKGROUND, label="cheap_label").result(timeout=5)
        learned = sched.estimated_cost("cheap_label")
        assert learned < 1.0

        future = sched.submit(_sleep_for, 0.001, tier=BACKGROUND, cost=999.0, label="cheap_label")
        future.result(timeout=5)
        # The override cost must not have been recorded as this label's duration sample.
        assert sched.sample_count("cheap_label") == 2
    finally:
        sched.shutdown()


def test_cost_history_prunes_samples_older_than_two_day_rolling_window(monkeypatch: pytest.MonkeyPatch):
    from services import priority_scheduler as ps

    fake_now = [1_000_000.0]
    monkeypatch.setattr(ps.time, "time", lambda: fake_now[0])

    history = ps._DurationHistory()
    history.record(10.0)  # a long-ago sample: e.g. a stale ATS run
    assert history.sample_count() == 1
    assert history.median(default=0.0) == pytest.approx(10.0)

    fake_now[0] += ps._COST_HISTORY_WINDOW_SECONDS - 1  # just inside the 2-day window
    assert history.sample_count() == 1

    fake_now[0] += 2  # now just outside the 2-day window
    history.record(0.5)  # fresh sample alongside the now-stale one
    assert history.sample_count() == 1  # stale sample pruned, only the fresh one remains
    assert history.median(default=0.0) == pytest.approx(0.5)


def test_cost_history_does_not_cap_sample_count_within_window():
    """The whole point of using the median instead of trimming outliers is that
    the full rolling-window sample survives -- assert nothing artificially
    shrinks it (e.g. a fixed-size ring buffer would)."""
    from services import priority_scheduler as ps

    history = ps._DurationHistory()
    sample_count = 500
    for i in range(sample_count):
        history.record(float(i % 7))
    assert history.sample_count() == sample_count


# ── Cancellation safety (set_running_or_notify_cancel) ─────────────────────


def test_cancelling_before_dispatch_skips_execution_cleanly():
    sched = PriorityWorkScheduler(max_workers=1)
    try:
        ran = threading.Event()

        def _mark_ran():
            ran.set()
            return "done"

        # Occupy the single worker so the next submission stays queued.
        blocker_order: list[str] = []
        blocker = sched.submit(_blocking, "blocker", blocker_order, 0.1, tier=BACKGROUND, cost=1.0)
        time.sleep(0.02)

        future = sched.submit(_mark_ran, tier=BACKGROUND, cost=1.0)
        assert future.cancel() is True  # still PENDING (queued behind blocker) -- cancel succeeds

        blocker.result(timeout=5)
        time.sleep(0.05)  # give the worker a chance to pop the cancelled task

        assert not ran.is_set()  # fn() must never have run
        assert future.cancelled()
        sched_stats = sched.stats()
        assert sched_stats["workers"] == 1  # worker pool must not have shrunk
    finally:
        sched.shutdown()


def test_future_cannot_be_cancelled_once_execution_has_started():
    """Direct proof of the fix's mechanism: once set_running_or_notify_cancel()
    has run, Future.cancel() can no longer succeed (concurrent.futures.Future
    only allows PENDING->CANCELLED, never RUNNING->CANCELLED). That is exactly
    what makes the later set_result()/set_exception() call race-free against
    a caller cancelling concurrently -- without it, cancel() could still
    succeed while fn() was executing, and set_result() would then raise
    InvalidStateError on the now-cancelled future."""
    sched = PriorityWorkScheduler(max_workers=1)
    try:
        started = threading.Event()
        finish_gate = threading.Event()

        def _slow_fn():
            started.set()
            finish_gate.wait(timeout=5)
            return "ok"

        future = sched.submit(_slow_fn, tier=BACKGROUND)
        assert started.wait(timeout=5)  # fn is now actively running on the worker

        assert future.cancel() is False
        assert not future.cancelled()

        finish_gate.set()
        assert future.result(timeout=5) == "ok"
    finally:
        sched.shutdown()


def test_cancelling_asyncio_side_while_fn_is_running_does_not_kill_the_worker():
    """End-to-end regression guard using the actual asyncio.wrap_future path
    this codebase relies on (CommandRouter._run_interactive /
    WatcherManager._tracked_to_thread both go through PriorityWorkScheduler.run,
    which awaits asyncio.wrap_future). Cancelling the awaiting asyncio task
    while fn() is executing must not crash the worker thread."""
    import asyncio

    sched = PriorityWorkScheduler(max_workers=1)
    try:
        started = threading.Event()
        finish_gate = threading.Event()

        def _slow_fn():
            started.set()
            finish_gate.wait(timeout=5)
            return "completed anyway"

        async def _main():
            task = asyncio.create_task(sched.run(_slow_fn, tier=INTERACTIVE))
            await asyncio.to_thread(started.wait, 5)
            assert started.is_set()

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            finish_gate.set()  # let the worker thread actually finish fn()
            await asyncio.sleep(0.2)  # give the worker loop time to hit set_result/finally

        asyncio.run(_main())

        # The critical assertion: the worker thread must still be alive and
        # able to pick up new work, not dead from an uncaught InvalidStateError.
        proof_future = sched.submit(lambda: "proof of life", tier=BACKGROUND)
        assert proof_future.result(timeout=5) == "proof of life"
        assert sched.stats()["workers"] == 1
    finally:
        sched.shutdown()


def test_many_rapid_cancellations_never_shrink_the_worker_pool():
    """Regression guard at higher volume: repeatedly submit-and-immediately-cancel
    and confirm the pool still services new work afterward."""
    sched = PriorityWorkScheduler(max_workers=2)
    try:
        for _ in range(50):
            future = sched.submit(time.sleep, 0.001, tier=BACKGROUND)
            future.cancel()

        proof_future = sched.submit(lambda: "alive", tier=INTERACTIVE)
        assert proof_future.result(timeout=5) == "alive"
        assert sched.stats()["workers"] == 2
    finally:
        sched.shutdown()
