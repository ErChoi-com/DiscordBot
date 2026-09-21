from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import job_service
from services.health import WatcherHealthTracker
from state.store import RuntimeStore
from watchers.manager import WatcherManager


@pytest.fixture(autouse=True)
def _isolate_job_archive(monkeypatch):
    """Keep the watcher send path out of the real job archive.

    The send path calls services.jba.merge_data.log_jobs, whose storage is a
    module-level absolute path to data/jba/jobs -- the real, git-committed
    archive the bot pushes. Without this, these tests wrote their synthetic
    postings (example.invalid links) into production data AND became
    order-dependent: log_jobs dedups, so once a fixture job was logged for
    today, the next run found it already archived, sent nothing, and failed
    the "expected watcher to send at least one message" assertion.

    Archiving is not what these tests are about, so it is stubbed rather than
    redirected; the calls are still recorded in case a test wants them.
    """
    logged: list[list[dict]] = []
    monkeypatch.setattr(
        "services.jba.merge_data.log_jobs",
        lambda jobs, date_str=None: (logged.append(list(jobs)), len(jobs))[1],
    )
    return logged


def _health() -> WatcherHealthTracker:
    return WatcherHealthTracker()


async def _drive_one_job_watcher_iteration(
    manager: WatcherManager,
    store: RuntimeStore,
    channel_id: int,
    *,
    timeout: float = 10.0,
) -> None:
    """Run exactly one job-watcher iteration, then stop the watcher.

    Deliberately not a fixed number of ``await asyncio.sleep(0)`` ticks. The
    iteration hands both the scrape and the semantic filter to a
    PriorityWorkScheduler thread (``manager._tracked_to_thread``), so how many
    event-loop ticks it needs is decided by when a worker thread gets scheduled
    -- not by anything the test controls. A ``for _ in range(10)`` spin
    therefore passes on an idle machine and fails under a loaded full-suite
    run, where the whole suite's scheduler threads are competing. That was a
    real intermittent failure, not a theoretical one.

    Wait on the iteration's own completion signal instead: the health tracker
    records the scrape once every send has been attempted, and it does so on
    the error path as well as the success path -- which matters, because one
    caller here asserts that *nothing* was sent.

    The watcher cannot be awaited to completion: its loop ends with
    ``asyncio.sleep(max(60, refresh_seconds))`` before it re-reads ``enabled``,
    so it is still cancelled, just after the work is observably done.
    """
    task = asyncio.create_task(manager._run_job_watcher(channel_id))
    await asyncio.sleep(0)
    store.channel_job_settings[channel_id]["enabled"] = False

    def _iteration_recorded() -> bool:
        health = manager.health.get_job_health(channel_id)
        return health is not None and health.total_scrapes >= 1

    deadline = time.monotonic() + timeout
    while not _iteration_recorded() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    finished = _iteration_recorded()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Asserted after the cancel so a timeout still tears the task down rather
    # than leaking a live watcher into the rest of the suite.
    assert finished, (
        f"job watcher did not complete an iteration within {timeout}s; "
        "the assertions below would be reading state that was never written"
    )


class _DummyClient:
    def __init__(self, *, fetch_error: Exception | None = None, fetch_result: object | None = None) -> None:
        self._fetch_error = fetch_error
        self._fetch_result = fetch_result

    def get_channel(self, channel_id: int) -> object | None:
        return None

    async def fetch_channel(self, channel_id: int) -> object | None:
        if self._fetch_error is not None:
            raise self._fetch_error
        return self._fetch_result


class _SendableChannel:
    def __init__(self, *, fail_send: bool = False) -> None:
        self.fail_send = fail_send
        self.sent: list[str] = []

    async def send(self, content: str | None = None, embeds: object | None = None) -> None:
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent.append(str(content or ""))


class _SlowSendableChannel(_SendableChannel):
    async def send(self, content: str | None = None, embeds: object | None = None) -> None:
        # Force overlap between concurrent send attempts.
        await asyncio.sleep(0.01)
        await super().send(content=content, embeds=embeds)


class _ChannelClient(_DummyClient):
    def __init__(self, channel: _SendableChannel) -> None:
        super().__init__()
        self._channel = channel

    def get_channel(self, channel_id: int) -> object | None:
        return self._channel


class _CachedChannelClient(_DummyClient):
    def get_channel(self, channel_id: int) -> object | None:
        return object()


def _make_config(base_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        base_dir=base_dir,
        dedup_months_threshold=2,
        discord_history_check_limit=5,
        jobspy_python_exe=None,
    )


def test_restore_enabled_watchers_purges_missing_job_channel(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / ".bot_state.json"
    store = RuntimeStore(state_path)
    store.channel_job_settings = {
        111: {"enabled": True},
        222: {"enabled": True},
    }
    store.channel_job_seen = {
        111: {"https://example.invalid/a"},
        222: {"https://example.invalid/b"},
    }

    manager = WatcherManager(client=_DummyClient(), config=_make_config(tmp_path), store=store, health=_health())

    async def _fake_channel_exists(channel_id: int) -> bool:
        return channel_id != 111

    monkeypatch.setattr(manager, "_channel_exists", _fake_channel_exists)
    monkeypatch.setattr(manager, "start_job_watcher", lambda channel_id: True)

    restored = asyncio.run(manager.restore_enabled_watchers())

    assert restored["job"] == 1
    assert 111 not in store.channel_job_settings
    assert 111 not in store.channel_job_seen
    assert 222 in store.channel_job_settings


def test_restore_keeps_missing_disabled_job_channel(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / ".bot_state.json"
    store = RuntimeStore(state_path)
    store.channel_job_settings = {
        111: {"enabled": False},
        222: {"enabled": True},
    }
    store.channel_job_seen = {
        111: {"https://example.invalid/stale"},
        222: {"https://example.invalid/live"},
    }

    manager = WatcherManager(client=_DummyClient(), config=_make_config(tmp_path), store=store, health=_health())

    async def _fake_channel_exists(channel_id: int) -> bool:
        return channel_id != 111

    monkeypatch.setattr(manager, "_channel_exists", _fake_channel_exists)
    monkeypatch.setattr(manager, "start_job_watcher", lambda channel_id: True)

    restored = asyncio.run(manager.restore_enabled_watchers())

    assert restored["job"] == 1
    assert 111 in store.channel_job_settings
    assert 111 in store.channel_job_seen
    assert 222 in store.channel_job_settings


def test_channel_probe_fails_open_on_unexpected_error(tmp_path: Path) -> None:
    manager = WatcherManager(
        client=_DummyClient(fetch_error=RuntimeError("temporary transport error")),
        config=_make_config(tmp_path),
        store=RuntimeStore(tmp_path / ".bot_state.json"),
        health=_health(),
    )

    exists = asyncio.run(manager._channel_exists(999))
    assert exists is True


def test_channel_probe_uses_cached_channel_without_fetch(tmp_path: Path) -> None:
    manager = WatcherManager(
        client=_CachedChannelClient(fetch_error=RuntimeError("should not fetch")),
        config=_make_config(tmp_path),
        store=RuntimeStore(tmp_path / ".bot_state.json"),
        health=_health(),
    )

    exists = asyncio.run(manager._channel_exists(999))
    assert exists is True


def test_channel_probe_returns_false_when_fetch_returns_none(tmp_path: Path) -> None:
    manager = WatcherManager(
        client=_DummyClient(fetch_result=None),
        config=_make_config(tmp_path),
        store=RuntimeStore(tmp_path / ".bot_state.json"),
        health=_health(),
    )

    exists = asyncio.run(manager._channel_exists(999))
    assert exists is False


def test_purge_deleted_job_channel_removes_dedup_directory(tmp_path: Path) -> None:
    state_path = tmp_path / ".bot_state.json"
    store = RuntimeStore(state_path)
    store.channel_job_settings = {333: {"enabled": True}}
    store.channel_job_seen = {333: {"https://example.invalid/job"}}

    manager = WatcherManager(client=_DummyClient(), config=_make_config(tmp_path), store=store, health=_health())
    listing_file = manager._dedup_listing_file(333, "job")
    dedup_dir = job_service.ensure_dedup_directory_for_listing_file(listing_file)
    (dedup_dir / "listing_0.json").write_text("123 1700000000\n", encoding="utf-8")

    manager._purge_deleted_job_channel(333)

    assert 333 not in store.channel_job_settings
    assert 333 not in store.channel_job_seen
    assert not dedup_dir.exists()


def test_watcher_attaches_channel_specific_listing_files(tmp_path: Path) -> None:
    manager = WatcherManager(
        client=_DummyClient(),
        config=_make_config(tmp_path),
        store=RuntimeStore(tmp_path / ".bot_state.json"),
        health=_health(),
    )

    a = manager._attached_listing_file(111, "job")
    b = manager._attached_listing_file(222, "job")
    c = manager._attached_listing_file(111, "reddit")

    assert a != b
    assert a.name == ".message_listing_111_job.json"
    assert b.name == ".message_listing_222_job.json"
    assert c.name == ".message_listing_111_reddit.json"


def test_manager_record_dedup_updates_only_target_channel_folder(tmp_path: Path) -> None:
    manager = WatcherManager(
        client=_DummyClient(),
        config=_make_config(tmp_path),
        store=RuntimeStore(tmp_path / ".bot_state.json"),
        health=_health(),
    )

    listing_a = manager._attached_listing_file(111, "job")
    listing_b = manager._attached_listing_file(222, "job")
    dir_a = job_service.ensure_dedup_directory_for_listing_file(listing_a)
    dir_b = job_service.ensure_dedup_directory_for_listing_file(listing_b)

    (dir_a / "listing_0.json").write_text("100 111 1700000000\n", encoding="utf-8")
    (dir_b / "listing_0.json").write_text("200 222 1700000000\n", encoding="utf-8")

    before_a = (dir_a / "listing_0.json").read_text(encoding="utf-8")
    before_b = (dir_b / "listing_0.json").read_text(encoding="utf-8")

    assert asyncio.run(manager._record_dedup_after_send(111, "job", "[Site] Target A\nhttps://example.invalid/a")) is True

    after_a = (dir_a / "listing_0.json").read_text(encoding="utf-8")
    after_b = (dir_b / "listing_0.json").read_text(encoding="utf-8")

    assert after_a != before_a
    assert after_b == before_b


def test_manager_job_write_does_not_touch_reddit_folder_same_channel(tmp_path: Path) -> None:
    manager = WatcherManager(
        client=_DummyClient(),
        config=_make_config(tmp_path),
        store=RuntimeStore(tmp_path / ".bot_state.json"),
        health=_health(),
    )

    listing_job = manager._attached_listing_file(333, "job")
    listing_reddit = manager._attached_listing_file(333, "reddit")
    dir_job = job_service.ensure_dedup_directory_for_listing_file(listing_job)
    dir_reddit = job_service.ensure_dedup_directory_for_listing_file(listing_reddit)

    (dir_job / "listing_0.json").write_text("100 333 1700000000\n", encoding="utf-8")
    (dir_reddit / "listing_0.json").write_text("900 333 1700000000\n", encoding="utf-8")

    before_job = (dir_job / "listing_0.json").read_text(encoding="utf-8")
    before_reddit = (dir_reddit / "listing_0.json").read_text(encoding="utf-8")

    assert asyncio.run(manager._record_dedup_after_send(333, "job", "[Site] Job Only\nhttps://example.invalid/job")) is True

    after_job = (dir_job / "listing_0.json").read_text(encoding="utf-8")
    after_reddit = (dir_reddit / "listing_0.json").read_text(encoding="utf-8")

    assert after_job != before_job
    assert after_reddit == before_reddit


def test_concurrent_identical_send_is_emitted_once(tmp_path: Path) -> None:
    state_path = tmp_path / ".bot_state.json"
    store = RuntimeStore(state_path)
    channel_id = 777001
    channel = _SlowSendableChannel(fail_send=False)
    manager = WatcherManager(client=_ChannelClient(channel), config=_make_config(tmp_path), store=store, health=_health())

    async def _run() -> tuple[bool, bool]:
        msg = "[LinkedIn] Full Stack Developer Intern - AI Solutions\nhttps://www.linkedin.com/jobs/view/4425828853"
        first, second = await asyncio.gather(
            manager.send_watcher_message(channel_id, msg, check_duplicate=True, watcher_type="job"),
            manager.send_watcher_message(channel_id, msg, check_duplicate=True, watcher_type="job"),
        )
        return first, second

    first, second = asyncio.run(_run())

    assert (first, second).count(True) == 1
    assert len(channel.sent) == 1


def test_dedup_namespace_isolation(tmp_path: Path) -> None:
    listing_a = tmp_path / ".message_listing_1_job.json"
    listing_b = tmp_path / ".message_listing_2_job.json"
    message = "[Site] Same Title\nhttps://example.invalid/post"

    assert job_service.check_and_record_message(message, listing_a, months_threshold=2) is False
    assert job_service.check_and_record_message(message, listing_b, months_threshold=2) is False
    assert job_service.check_and_record_message(message, listing_a, months_threshold=2) is True


def test_same_message_rows_differ_between_channel_namespaces(tmp_path: Path) -> None:
    listing_a = tmp_path / ".message_listing_1482891850439458857_job.json"
    listing_b = tmp_path / ".message_listing_1508610365959704626_job.json"
    message = "[Glassdoor] Controls Engineer\nhttps://example.invalid/post"

    assert job_service.record_message_for_dedup(message, listing_a) is True
    assert job_service.record_message_for_dedup(message, listing_b) is True

    row_a = (job_service.dedup_directory_for_listing_file(listing_a) / "listing_0.json").read_text(encoding="utf-8").splitlines()[0]
    row_b = (job_service.dedup_directory_for_listing_file(listing_b) / "listing_0.json").read_text(encoding="utf-8").splitlines()[0]

    assert row_a != row_b


@pytest.mark.parametrize(
    ("listing_name_a", "listing_name_b", "expected_cross_duplicate"),
    [
        (".message_listing_100_job.json", ".message_listing_100_job.json", True),
        (".message_listing_100_job.json", ".message_listing_101_job.json", False),
        (".message_listing_100_job.json", ".message_listing_100_reddit.json", False),
    ],
)
def test_dedup_overlap_matrix_by_namespace(
    tmp_path: Path,
    listing_name_a: str,
    listing_name_b: str,
    expected_cross_duplicate: bool,
) -> None:
    listing_a = tmp_path / listing_name_a
    listing_b = tmp_path / listing_name_b
    message = "[SiteA] Overlap Posting\nhttps://example.invalid/overlap"

    assert job_service.check_and_record_message(message, listing_a, months_threshold=2) is False
    assert job_service.check_and_record_message(message, listing_b, months_threshold=2) is expected_cross_duplicate


def test_site_prefix_is_ignored_in_title_signature(tmp_path: Path) -> None:
    listing_file = tmp_path / ".message_listing_321_job.json"
    message_a = "[Indeed] Same Role Title\nhttps://example.invalid/1"
    message_b = "[LinkedIn] Same Role Title\nhttps://example.invalid/2"

    assert job_service.check_and_record_message(message_a, listing_file, months_threshold=2) is False
    assert job_service.check_and_record_message(message_b, listing_file, months_threshold=2) is True


def test_different_titles_in_same_namespace_are_not_duplicates(tmp_path: Path) -> None:
    listing_file = tmp_path / ".message_listing_654_job.json"
    msg_1 = "[Site] Backend Engineer\nhttps://example.invalid/a"
    msg_2 = "[Site] Frontend Engineer\nhttps://example.invalid/b"

    assert job_service.check_and_record_message(msg_1, listing_file, months_threshold=2) is False
    assert job_service.check_and_record_message(msg_2, listing_file, months_threshold=2) is False


def test_old_entries_outside_threshold_do_not_block(tmp_path: Path, monkeypatch) -> None:
    listing_file = tmp_path / ".message_listing_777_job.json"
    dedup_dir = job_service.ensure_dedup_directory_for_listing_file(listing_file)

    message = "[Site] Expired Duplicate Window\nhttps://example.invalid/old"
    sig = job_service.dedup_signature_for_message(message, listing_file)
    assert sig is not None
    now = 2_000_000_000
    old = now - (3 * 30 * 24 * 3600)
    (dedup_dir / "listing_0.json").write_text(f"{sig} {old}\n", encoding="utf-8")

    monkeypatch.setattr(job_service, "compress_timestamp", lambda: now)

    assert job_service.check_and_record_message(message, listing_file, months_threshold=2) is False


def test_malformed_line_does_not_hide_valid_duplicate(tmp_path: Path, monkeypatch) -> None:
    listing_file = tmp_path / ".message_listing_909_job.json"
    dedup_dir = job_service.ensure_dedup_directory_for_listing_file(listing_file)
    message = "[Site] Parse Robustness Role\nhttps://example.invalid/x"
    sig = job_service.dedup_signature_for_message(message, listing_file)
    assert sig is not None

    now = 2_000_000_000
    valid_recent = now - (10 * 24 * 3600)
    malformed = "this is malformed not_a_timestamp"
    valid = f"{sig} {valid_recent}"
    (dedup_dir / "listing_0.json").write_text(f"{malformed}\n{valid}\n", encoding="utf-8")

    monkeypatch.setattr(job_service, "compress_timestamp", lambda: now)

    assert job_service.check_and_record_message(message, listing_file, months_threshold=2) is True


def test_is_message_duplicate_accepts_legacy_plain_signature_rows(tmp_path: Path, monkeypatch) -> None:
    listing_file = tmp_path / ".message_listing_1508610365959704626_job.json"
    dedup_dir = job_service.ensure_dedup_directory_for_listing_file(listing_file)
    message = "[Site] Legacy Signature Role\nhttps://example.invalid/legacy"

    scoped = job_service.dedup_signature_for_message(message, listing_file)
    assert scoped is not None
    legacy = job_service.dedup_legacy_signature_from_scoped(scoped, listing_file)
    assert legacy

    now = 2_000_000_000
    recent = now - (3 * 24 * 3600)
    # Legacy row format intentionally omits namespace token.
    (dedup_dir / "listing_0.json").write_text(f"{legacy} {recent}\n", encoding="utf-8")
    monkeypatch.setattr(job_service, "compress_timestamp", lambda: now)

    assert job_service.is_message_duplicate(message, listing_file, months_threshold=2) is True


def test_namespace_sanitization_for_unusual_listing_names(tmp_path: Path) -> None:
    listing_file = tmp_path / ".message listing 12/34 ? job.json"
    dedup_dir = job_service.ensure_dedup_directory_for_listing_file(listing_file)

    assert dedup_dir.exists()
    assert " " not in dedup_dir.name
    assert "/" not in dedup_dir.name


def test_noncanonical_listing_names_do_not_collapse_to_same_namespace(tmp_path: Path) -> None:
    # These two names sanitize similarly; they must still map to distinct namespaces.
    listing_a = tmp_path / "a b.json"
    listing_b = tmp_path / "a+b.json"
    message = "[Site] Collision Probe Role\nhttps://example.invalid/collision"

    dir_a = job_service.ensure_dedup_directory_for_listing_file(listing_a)
    dir_b = job_service.ensure_dedup_directory_for_listing_file(listing_b)

    assert dir_a != dir_b
    assert dir_a.name != dir_b.name

    assert job_service.check_and_record_message(message, listing_a, months_threshold=2) is False
    # If namespaces accidentally collapse, this would incorrectly return True.
    assert job_service.check_and_record_message(message, listing_b, months_threshold=2) is False


def test_dedup_rotates_when_all_fifo_files_are_full(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(job_service, "DEDUP_MAX_FIFO_FILES", 2)
    monkeypatch.setattr(job_service, "DEDUP_MAX_ENTRIES_PER_FILE", 2)

    listing_file = tmp_path / ".message_listing_777_job.json"
    dedup_dir = job_service.ensure_dedup_directory_for_listing_file(listing_file)

    (dedup_dir / "listing_0.json").write_text("111 1700000000\n222 1700000001\n", encoding="utf-8")
    (dedup_dir / "listing_1.json").write_text("333 1700000002\n444 1700000003\n", encoding="utf-8")

    is_duplicate = job_service.check_and_record_message(
        "[Site] Brand New Posting\nhttps://example.invalid/new",
        listing_file,
        months_threshold=2,
    )

    assert is_duplicate is False
    assert (dedup_dir / "listing_0.json").exists()
    assert (dedup_dir / "listing_1.json").exists()

    lines_0 = [line for line in (dedup_dir / "listing_0.json").read_text(encoding="utf-8").splitlines() if line.strip()]
    lines_1 = [line for line in (dedup_dir / "listing_1.json").read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines_0) == 2
    assert len(lines_1) == 2
    # Newest entry is inserted at the top of listing_0.
    assert lines_0[0] != "111 1700000000"


def test_fifo_enforcement_trims_oversized_files_and_deletes_extra_indices(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(job_service, "DEDUP_MAX_FIFO_FILES", 2)
    monkeypatch.setattr(job_service, "DEDUP_MAX_ENTRIES_PER_FILE", 3)

    listing_file = tmp_path / ".message_listing_500_job.json"
    dedup_dir = job_service.ensure_dedup_directory_for_listing_file(listing_file)

    (dedup_dir / "listing_0.json").write_text("1 100\n2 101\n3 102\n4 103\n", encoding="utf-8")
    (dedup_dir / "listing_1.json").write_text("10 200\n11 201\n12 202\n13 203\n", encoding="utf-8")
    (dedup_dir / "listing_2.json").write_text("99 999\n", encoding="utf-8")

    changed = job_service.enforce_dedup_fifo_structure_for_listing_file(listing_file)
    assert changed is True

    lines_0 = [line for line in (dedup_dir / "listing_0.json").read_text(encoding="utf-8").splitlines() if line.strip()]
    lines_1 = [line for line in (dedup_dir / "listing_1.json").read_text(encoding="utf-8").splitlines() if line.strip()]

    # Sorted newest-first, then split into 2 chunks of 3. Oldest 3 dropped.
    assert lines_0 == ["99 999", "13 203", "12 202"]
    assert lines_1 == ["11 201", "10 200", "4 103"]
    assert not (dedup_dir / "listing_2.json").exists()


def test_record_inserts_at_top_and_pushes_overflow_to_next_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(job_service, "DEDUP_MAX_FIFO_FILES", 2)
    monkeypatch.setattr(job_service, "DEDUP_MAX_ENTRIES_PER_FILE", 2)
    monkeypatch.setattr(job_service, "compress_timestamp", lambda: 1_700_000_999)

    listing_file = tmp_path / ".message_listing_900_job.json"
    dedup_dir = job_service.ensure_dedup_directory_for_listing_file(listing_file)
    (dedup_dir / "listing_0.json").write_text("100 900 1700000001\n101 900 1700000002\n", encoding="utf-8")
    (dedup_dir / "listing_1.json").write_text("102 900 1700000003\n103 900 1700000004\n", encoding="utf-8")

    message = "[Site] New Head Entry\nhttps://example.invalid/new-head"
    assert job_service.record_message_for_dedup(message, listing_file) is True

    lines_0 = [line for line in (dedup_dir / "listing_0.json").read_text(encoding="utf-8").splitlines() if line.strip()]
    lines_1 = [line for line in (dedup_dir / "listing_1.json").read_text(encoding="utf-8").splitlines() if line.strip()]

    # Enforcement sorts newest-first, then cascade inserts at top and pops oldest.
    assert lines_0[0].endswith(" 900 1700000999")
    assert lines_0[1] == "103 900 1700000004"
    assert lines_1 == ["102 900 1700000003", "101 900 1700000002"]


def test_job_watcher_records_seen_and_dedup_only_after_successful_send(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / ".bot_state.json"
    store = RuntimeStore(state_path)
    channel_id = 123456
    store.channel_job_settings = {
        channel_id: {
            "enabled": True,
            "sites": ["all"],
            "keywords": "electrical intern",
            "location": "Canada",
            "hours_old": 24,
            "results_wanted": 10,
            "radius_miles": 25,
            "role_filters": ["internship"],
            "exclusion_terms": [],
            "semantic_threshold": 0.1,
            "refresh_seconds": 60,
            "allow_north_america": False,
        }
    }

    channel = _SendableChannel(fail_send=False)
    manager = WatcherManager(client=_ChannelClient(channel), config=_make_config(tmp_path), store=store, health=_health())

    async def _noop_channel_exists(channel_id: int) -> bool:
        return True

    def _fake_scrape(*args, **kwargs):
        return [
            {
                "site_label": "Glassdoor",
                "title": "Intern - Controls Engineer (Human Computer Lab, Toronto, ON)",
                "link": "https://example.invalid/job/1",
                "company": "Human Computer Lab",
            }
        ]

    monkeypatch.setattr(manager, "_channel_exists", _noop_channel_exists)
    monkeypatch.setattr(job_service, "scrape_job_postings", _fake_scrape)
    monkeypatch.setattr(job_service, "semantic_filter_items", lambda items, *args, **kwargs: list(items))

    asyncio.run(_drive_one_job_watcher_iteration(manager, store, channel_id))

    assert channel.sent, "expected watcher to send at least one message"
    assert "https://example.invalid/job/1" in store.channel_job_seen.get(channel_id, set())

    listing_file = manager._dedup_listing_file(channel_id, "job")
    assert job_service.is_message_duplicate(channel.sent[0], listing_file, months_threshold=2) is True


def test_job_watcher_does_not_record_seen_or_dedup_when_send_fails(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / ".bot_state.json"
    store = RuntimeStore(state_path)
    channel_id = 789012
    store.channel_job_settings = {
        channel_id: {
            "enabled": True,
            "sites": ["all"],
            "keywords": "electrical intern",
            "location": "Canada",
            "hours_old": 24,
            "results_wanted": 10,
            "radius_miles": 25,
            "role_filters": ["internship"],
            "exclusion_terms": [],
            "semantic_threshold": 0.1,
            "refresh_seconds": 60,
            "allow_north_america": False,
        }
    }

    channel = _SendableChannel(fail_send=True)
    manager = WatcherManager(client=_ChannelClient(channel), config=_make_config(tmp_path), store=store, health=_health())

    async def _noop_channel_exists(channel_id: int) -> bool:
        return True

    def _fake_scrape(*args, **kwargs):
        return [
            {
                "site_label": "Glassdoor",
                "title": "Fall 2026 Co-op - Electrical Engineer (Lockheed Martin, Ottawa, ON)",
                "link": "https://example.invalid/job/lockheed",
                "company": "Lockheed Martin",
            }
        ]

    monkeypatch.setattr(manager, "_channel_exists", _noop_channel_exists)
    monkeypatch.setattr(job_service, "scrape_job_postings", _fake_scrape)
    monkeypatch.setattr(job_service, "semantic_filter_items", lambda items, *args, **kwargs: list(items))

    asyncio.run(_drive_one_job_watcher_iteration(manager, store, channel_id))

    assert channel.sent == []
    assert "https://example.invalid/job/lockheed" not in store.channel_job_seen.get(channel_id, set())

    msg = "[Glassdoor] Fall 2026 Co-op - Electrical Engineer (Lockheed Martin, Ottawa, ON)\nhttps://example.invalid/job/lockheed"
    listing_file = manager._dedup_listing_file(channel_id, "job")
    assert job_service.is_message_duplicate(msg, listing_file, months_threshold=2) is False


def test_job_watcher_dedupes_duplicate_links_within_single_scrape_batch(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / ".bot_state.json"
    store = RuntimeStore(state_path)
    channel_id = 246810
    store.channel_job_settings = {
        channel_id: {
            "enabled": True,
            "sites": ["all"],
            "keywords": "full stack intern",
            "location": "Canada",
            "hours_old": 24,
            "results_wanted": 10,
            "radius_miles": 25,
            "role_filters": [],
            "exclusion_terms": [],
            "semantic_threshold": 0.1,
            "refresh_seconds": 60,
            "allow_north_america": False,
        }
    }

    channel = _SendableChannel(fail_send=False)
    manager = WatcherManager(client=_ChannelClient(channel), config=_make_config(tmp_path), store=store, health=_health())

    async def _noop_channel_exists(channel_id: int) -> bool:
        return True

    def _fake_scrape(*args, **kwargs):
        # Same posting link appears twice in one scrape result.
        return [
            {
                "site_label": "LinkedIn",
                "title": "Full Stack Developer Intern - AI Solutions",
                "link": "https://www.linkedin.com/jobs/view/4425828853",
                "company": "Digy4",
            },
            {
                "site_label": "LinkedIn",
                "title": "Full Stack Developer Intern - AI Solutions",
                "link": "https://www.linkedin.com/jobs/view/4425828853",
                "company": "Digy4",
            },
        ]

    monkeypatch.setattr(manager, "_channel_exists", _noop_channel_exists)
    monkeypatch.setattr(job_service, "scrape_job_postings", _fake_scrape)
    monkeypatch.setattr(job_service, "semantic_filter_items", lambda items, *args, **kwargs: list(items))

    asyncio.run(_drive_one_job_watcher_iteration(manager, store, channel_id))

    assert len(channel.sent) == 1
    assert channel.sent[0].count("4425828853") == 1
    assert "https://www.linkedin.com/jobs/view/4425828853" in store.channel_job_seen.get(channel_id, set())


def test_job_watcher_dedupes_canonical_link_variants_within_batch(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / ".bot_state.json"
    store = RuntimeStore(state_path)
    channel_id = 246811
    store.channel_job_settings = {
        channel_id: {
            "enabled": True,
            "sites": ["all"],
            "keywords": "full stack intern",
            "location": "Canada",
            "hours_old": 24,
            "results_wanted": 10,
            "radius_miles": 25,
            "role_filters": [],
            "exclusion_terms": [],
            "semantic_threshold": 0.1,
            "refresh_seconds": 60,
            "allow_north_america": False,
        }
    }

    channel = _SendableChannel(fail_send=False)
    manager = WatcherManager(client=_ChannelClient(channel), config=_make_config(tmp_path), store=store, health=_health())

    async def _noop_channel_exists(channel_id: int) -> bool:
        return True

    def _fake_scrape(*args, **kwargs):
        return [
            {
                "site_label": "LinkedIn",
                "title": "Full Stack Developer Intern - AI Solutions",
                "link": "https://www.linkedin.com/jobs/view/4425828853?utm_source=abc",
                "company": "Digy4",
            },
            {
                "site_label": "LinkedIn",
                "title": "Full Stack Developer Intern - AI Solutions",
                "link": "https://www.linkedin.com/jobs/view/4425828853",
                "company": "Digy4",
            },
        ]

    monkeypatch.setattr(manager, "_channel_exists", _noop_channel_exists)
    monkeypatch.setattr(job_service, "scrape_job_postings", _fake_scrape)
    monkeypatch.setattr(job_service, "semantic_filter_items", lambda items, *args, **kwargs: list(items))

    asyncio.run(_drive_one_job_watcher_iteration(manager, store, channel_id))

    assert len(channel.sent) == 1
    assert "https://www.linkedin.com/jobs/view/4425828853" in store.channel_job_seen.get(channel_id, set())


def test_copied_identical_dedup_files_do_not_cross_suppress_between_namespaces(tmp_path: Path) -> None:
    listing_a = tmp_path / ".message_listing_100_job.json"
    listing_b = tmp_path / ".message_listing_101_job.json"
    message = "[Glassdoor] Intern - Controls Engineer (Human Computer Lab, Toronto, ON)\nhttps://example.invalid/hcl"

    assert job_service.record_message_for_dedup(message, listing_a) is True

    dir_a = job_service.dedup_directory_for_listing_file(listing_a)
    dir_b = job_service.ensure_dedup_directory_for_listing_file(listing_b)
    for src in dir_a.glob("listing_*.json"):
        shutil.copy2(src, dir_b / src.name)

    # Watcher-scoped token keeps copied rows from cross-suppressing other channels.
    assert job_service.is_message_duplicate(message, listing_b, months_threshold=2) is False


def test_normalize_dedup_storage_removes_legacy_rows_for_namespace(tmp_path: Path) -> None:
    listing_file = tmp_path / ".message_listing_555_job.json"
    dedup_dir = job_service.ensure_dedup_directory_for_listing_file(listing_file)

    # Legacy unscoped row format.
    legacy_row = "500 520 1700000000"
    # Current v2 scoped row from previous format (should be converted).
    scoped_sig = "v2:message_listing_555_job:500 520"
    scoped_row = f"{scoped_sig} 1700000001"
    # Current v2 row but for a different namespace (also converted).
    wrong_ns_row = "v2:message_listing_777_job:500 520 1700000002"

    (dedup_dir / "listing_0.json").write_text(
        "\n".join([legacy_row, scoped_row, wrong_ns_row]) + "\n",
        encoding="utf-8",
    )

    changed = job_service.normalize_dedup_storage_for_listing_file(listing_file)
    assert changed is True

    lines = [line for line in (dedup_dir / "listing_0.json").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == [
        "500 520 555 1700000002",
        "500 520 555 1700000001",
        "500 520 555 1700000000",
    ]


# ---------------------------------------------------------------------------
# Ordering and gap invariants
# ---------------------------------------------------------------------------

def _extract_timestamp(row: str) -> int:
    """Pull the trailing unix timestamp from a dedup row."""
    parts = row.strip().split()
    return int(parts[-1])


def _read_listing_rows(dedup_dir: Path, file_idx: int) -> list[str]:
    fp = dedup_dir / f"listing_{file_idx}.json"
    if not fp.exists():
        return []
    return [line.strip() for line in fp.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_enforcement_fixes_ascending_legacy_data_to_descending(tmp_path: Path, monkeypatch) -> None:
    """Legacy data loaded in chronological (ascending) order must be sorted
    newest-first after enforcement so the cascade evicts the oldest entries."""
    monkeypatch.setattr(job_service, "DEDUP_MAX_FIFO_FILES", 3)
    monkeypatch.setattr(job_service, "DEDUP_MAX_ENTRIES_PER_FILE", 5)

    listing_file = tmp_path / ".message_listing_800_job.json"
    dedup_dir = job_service.ensure_dedup_directory_for_listing_file(listing_file)

    base_ts = 1_700_000_000
    # Seed all three files in ASCENDING order (the legacy bug).
    for file_idx in range(3):
        rows = []
        for row_idx in range(5):
            ts = base_ts + file_idx * 500 + row_idx * 60
            rows.append(f"100 800 {ts}")
        (dedup_dir / f"listing_{file_idx}.json").write_text(
            "\n".join(rows) + "\n", encoding="utf-8",
        )

    job_service.enforce_dedup_fifo_structure_for_listing_file(listing_file)

    # --- check: every file must be in descending (newest-first) order ---
    for file_idx in range(3):
        rows = _read_listing_rows(dedup_dir, file_idx)
        assert len(rows) == 5, f"listing_{file_idx} should still have 5 rows"
        timestamps = [_extract_timestamp(r) for r in rows]
        for i in range(1, len(timestamps)):
            assert timestamps[i - 1] >= timestamps[i], (
                f"listing_{file_idx} row {i - 1}→{i} is not descending: "
                f"{timestamps[i - 1]} < {timestamps[i]}"
            )

    # --- check: cross-file continuity (listing_N bottom >= listing_N+1 top) ---
    for file_idx in range(2):
        bottom = _extract_timestamp(_read_listing_rows(dedup_dir, file_idx)[-1])
        top_next = _extract_timestamp(_read_listing_rows(dedup_dir, file_idx + 1)[0])
        assert bottom >= top_next, (
            f"listing_{file_idx} bottom ({bottom}) < listing_{file_idx + 1} top ({top_next})"
        )


def test_cascade_inserts_preserve_descending_order_with_bounded_gaps(tmp_path: Path, monkeypatch) -> None:
    """After a burst of cascade inserts the within-file order must stay
    descending and the gap between any two adjacent entries (including
    across file boundaries) must not exceed MAX_GAP_SECONDS."""
    MAX_GAP_SECONDS = 600  # 10 minutes — generous for 60-second-spaced seeds

    monkeypatch.setattr(job_service, "DEDUP_MAX_FIFO_FILES", 3)
    monkeypatch.setattr(job_service, "DEDUP_MAX_ENTRIES_PER_FILE", 4)

    listing_file = tmp_path / ".message_listing_801_job.json"
    dedup_dir = job_service.ensure_dedup_directory_for_listing_file(listing_file)

    base_ts = 1_700_000_000
    # Seed listing_0 and listing_1 with 4 rows each, ascending (legacy).
    # Rows span base_ts .. base_ts + 420 (7 minutes total).
    for file_idx in range(2):
        rows = []
        for row_idx in range(4):
            ts = base_ts + file_idx * 240 + row_idx * 60
            rows.append(f"100 801 {ts}")
        (dedup_dir / f"listing_{file_idx}.json").write_text(
            "\n".join(rows) + "\n", encoding="utf-8",
        )

    # Insert 6 new entries, each 90 seconds apart, starting just after the seed.
    insert_base = base_ts + 540
    titles = [
        f"[Site] Cascade Entry {n}\nhttps://example.invalid/{n}"
        for n in range(6)
    ]
    for i, title in enumerate(titles):
        monkeypatch.setattr(
            job_service, "compress_timestamp", lambda _ts=insert_base + i * 90: _ts,
        )
        job_service.record_message_for_dedup(title, listing_file)

    # --- collect every timestamp across all files in logical order ---
    all_timestamps: list[int] = []
    for file_idx in range(3):
        rows = _read_listing_rows(dedup_dir, file_idx)
        all_timestamps.extend(_extract_timestamp(r) for r in rows)

    assert len(all_timestamps) > 0

    # --- within-file descending ---
    for file_idx in range(3):
        rows = _read_listing_rows(dedup_dir, file_idx)
        if not rows:
            continue
        timestamps = [_extract_timestamp(r) for r in rows]
        for i in range(1, len(timestamps)):
            assert timestamps[i - 1] >= timestamps[i], (
                f"listing_{file_idx} row {i - 1}→{i} not descending: "
                f"{timestamps[i - 1]} < {timestamps[i]}"
            )

    # --- cross-file continuity ---
    prev_bottom: int | None = None
    for file_idx in range(3):
        rows = _read_listing_rows(dedup_dir, file_idx)
        if not rows:
            continue
        top = _extract_timestamp(rows[0])
        if prev_bottom is not None:
            assert prev_bottom >= top, (
                f"listing_{file_idx - 1} bottom ({prev_bottom}) < "
                f"listing_{file_idx} top ({top})"
            )
        prev_bottom = _extract_timestamp(rows[-1])

    # --- gap check across the entire logical sequence ---
    for i in range(1, len(all_timestamps)):
        gap = all_timestamps[i - 1] - all_timestamps[i]
        assert gap >= 0, (
            f"sequence position {i - 1}→{i} is ascending "
            f"({all_timestamps[i - 1]} < {all_timestamps[i]})"
        )
        assert gap <= MAX_GAP_SECONDS, (
            f"sequence position {i - 1}→{i} gap too large: "
            f"{gap}s > {MAX_GAP_SECONDS}s  "
            f"(ts {all_timestamps[i - 1]} → {all_timestamps[i]})"
        )


def test_enforcement_on_scrambled_data_produces_monotonic_sequence(tmp_path: Path, monkeypatch) -> None:
    """Even with completely randomised row order across files, enforcement
    must produce a globally monotonic descending timestamp sequence."""
    import random

    monkeypatch.setattr(job_service, "DEDUP_MAX_FIFO_FILES", 3)
    monkeypatch.setattr(job_service, "DEDUP_MAX_ENTRIES_PER_FILE", 6)

    listing_file = tmp_path / ".message_listing_802_job.json"
    dedup_dir = job_service.ensure_dedup_directory_for_listing_file(listing_file)

    base_ts = 1_700_000_000
    all_rows = [f"100 802 {base_ts + i * 30}" for i in range(18)]

    rng = random.Random(42)
    rng.shuffle(all_rows)

    # Distribute shuffled rows across 3 files of 6.
    for file_idx in range(3):
        chunk = all_rows[file_idx * 6 : (file_idx + 1) * 6]
        (dedup_dir / f"listing_{file_idx}.json").write_text(
            "\n".join(chunk) + "\n", encoding="utf-8",
        )

    job_service.enforce_dedup_fifo_structure_for_listing_file(listing_file)

    all_timestamps: list[int] = []
    for file_idx in range(3):
        rows = _read_listing_rows(dedup_dir, file_idx)
        assert len(rows) == 6
        all_timestamps.extend(_extract_timestamp(r) for r in rows)

    for i in range(1, len(all_timestamps)):
        assert all_timestamps[i - 1] >= all_timestamps[i], (
            f"global position {i - 1}→{i} not descending: "
            f"{all_timestamps[i - 1]} < {all_timestamps[i]}"
        )


def test_dedup_file_work_does_not_run_on_the_event_loop_thread(tmp_path: Path, monkeypatch) -> None:
    """loop_watch caught the loop thread globbing and rewriting the dedup FIFO
    files on the ticks that ran late; both the check and the record now run on
    a worker while the per-channel send lock keeps their order."""
    import threading

    state_path = tmp_path / ".bot_state.json"
    store = RuntimeStore(state_path)
    channel_id = 9090
    store.channel_job_settings = {channel_id: {"enabled": True}}
    channel = _SendableChannel()
    manager = WatcherManager(client=_ChannelClient(channel), config=_make_config(tmp_path), store=store, health=_health())

    threads: dict[str, int] = {}

    def _check(content, listing_file, months_threshold=1):
        threads["check"] = threading.get_ident()
        return False

    def _record(content, listing_file):
        threads["record"] = threading.get_ident()
        return True

    monkeypatch.setattr(job_service, "is_message_duplicate", _check)
    monkeypatch.setattr(job_service, "record_message_for_dedup", _record)

    async def _go():
        monkeypatch.setattr(manager, "should_skip_duplicate_message", _never_skip)
        return await manager.send_watcher_message(channel_id, "[Site] Job\nhttps://example.invalid/x")

    async def _never_skip(*args, **kwargs):
        return False

    assert asyncio.run(_go()) is True
    loop_thread = threading.main_thread().ident
    assert threads["check"] != loop_thread, "the duplicate check ran on the loop thread"
    assert threads["record"] != loop_thread, "the dedup record ran on the loop thread"
    assert channel.sent, "and the message was still sent"
