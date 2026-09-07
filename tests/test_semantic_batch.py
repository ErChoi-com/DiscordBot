"""The semantic filter judges a channel's candidates in one model call, after
the links it has already sent are gone.

Measured on the live bot: semantic_filter ran 2,800s per channel for ~30 short
texts. Three things compounded. Every item was its own model call that
re-encoded the search text; torch took every core for each of those calls and
two channels' calls fought each other for them alongside 500 scrape threads;
and the same postings came back every refresh for the whole 24h window, were
embedded again, and were then dropped by the link check that ran *after*.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import job_service  # noqa: E402
from services.health import WatcherHealthTracker  # noqa: E402
from state.store import RuntimeStore  # noqa: E402
from watchers.manager import WatcherManager  # noqa: E402

KEYWORDS, LOCATION = "software intern", "Toronto"
SEARCH = job_service.search_text_for_semantic_match(KEYWORDS, LOCATION, [])


class _FakeModel:
    """Scores by rule, not by weights: the search text and anything that
    starts with "Software" point one way, everything else the other. That
    makes "best chunk wins" observable -- a title whose *second* chunk starts
    with Software must pass even though the whole title does not."""

    def __init__(self, hold: float = 0.0) -> None:
        self.calls: list[list[str]] = []
        self.hold = hold
        self.live = 0
        self.peak = 0
        self.lock = threading.Lock()

    def encode(self, texts, normalize_embeddings=True, batch_size=None):
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
            self.calls.append(list(texts))
        try:
            if self.hold:
                time.sleep(self.hold)
            return np.array([self._vec(t) for t in texts])
        finally:
            with self.lock:
                self.live -= 1

    @staticmethod
    def _vec(text: str):
        if text == SEARCH or text.startswith("Software"):
            return [1.0, 0.0]
        if text.startswith("Somewhat"):
            return [0.6, 0.8]          # cos 0.6 against the query
        return [0.0, 1.0]


@pytest.fixture
def model(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(job_service, "load_semantic_plugin_model", lambda: fake)
    return fake


def _item(title="", description="", link="https://x/1", company="Acme"):
    return {"title": title, "description": description, "link": link, "company": company, "location": "Toronto"}


# ── one call, right texts, right verdicts ────────────────────────────────────

def test_a_batch_is_one_model_call_with_the_query_first(model):
    items = [_item("Software Intern", "Software internship at Acme"),
             _item("Bakery cashier", "Cashier wanted"),
             _item("Data Analyst - Software Engineering Intern")]
    job_service.semantic_filter_items(items, KEYWORDS, LOCATION, [])
    assert len(model.calls) == 1, f"expected one encode call, got {len(model.calls)}"
    assert model.calls[0][0] == SEARCH
    assert len(model.calls[0]) > len(items), "every candidate's texts share the batch"


def test_descriptions_are_judged_and_scores_are_annotated(model):
    good = _item("Anything", "Software internship at Acme")
    bad = _item("Software Intern", "Cashier wanted")   # description wins over title
    kept = job_service.semantic_filter_items([good, bad], KEYWORDS, LOCATION, [])
    assert kept == [good]
    assert good["semantic_score"] == pytest.approx(1.0)
    assert bad["semantic_score"] == pytest.approx(0.0)


def test_a_title_is_judged_on_its_best_chunk_when_there_is_no_description(model):
    """'Data Analyst - Software Engineering Intern': the whole title scores
    low, its second chunk scores high, and the chunk is what counts."""
    item = _item("Data Analyst - Software Engineering Intern")
    assert job_service.semantic_filter_items([item], KEYWORDS, LOCATION, []) == [item]
    texts = model.calls[0]
    assert any(t.startswith("Software Engineering Intern") for t in texts)


def test_the_threshold_may_depend_on_the_item(model):
    """The watcher holds ATS rows to a different bar than site rows without a
    second pass; the bar is a function of the item."""
    site_row = _item("Somewhat related", link="https://x/site")
    ats_row = _item("Somewhat related", link="https://x/ats")
    ats_row["sites"] = ["greenhouse"]
    kept = job_service.semantic_filter_items(
        [site_row, ats_row], KEYWORDS, LOCATION, [],
        threshold=lambda it: 0.9 if it.get("sites") else 0.5,
    )
    assert kept == [site_row]
    assert site_row["semantic_score"] == pytest.approx(0.6)


def test_a_numeric_threshold_and_the_default_both_work(model):
    row = _item("Somewhat related")
    assert job_service.semantic_filter_items([row], KEYWORDS, LOCATION, [], threshold=0.7) == []
    assert job_service.semantic_filter_items([row], KEYWORDS, LOCATION, [], threshold=0.5) == [row]
    assert job_service.SEMANTIC_PLUGIN_THRESHOLD < 0.6
    assert job_service.semantic_filter_items([row], KEYWORDS, LOCATION, []) == [row]


def test_an_item_with_nothing_to_judge_is_kept_and_not_encoded(model):
    blank = {"link": "https://x/blank"}
    good = _item("Software Intern")
    kept = job_service.semantic_filter_items([blank, good], KEYWORDS, LOCATION, [])
    assert kept == [blank, good]
    assert "semantic_score" not in blank


def test_no_search_text_keeps_everything_without_touching_the_model(model):
    items = [_item("Bakery cashier")]
    assert job_service.semantic_filter_items(items, "", "", []) == items
    assert model.calls == []


def test_no_model_keeps_everything(monkeypatch):
    monkeypatch.setattr(job_service, "load_semantic_plugin_model", lambda: None)
    items = [_item("Bakery cashier")]
    assert job_service.semantic_filter_items(items, KEYWORDS, LOCATION, []) == items


def test_a_failing_model_keeps_everything_rather_than_dropping_the_batch(monkeypatch, capsys):
    class _Broken:
        def encode(self, *a, **k):
            raise RuntimeError("cuda fell over")

    monkeypatch.setattr(job_service, "load_semantic_plugin_model", lambda: _Broken())
    items = [_item("Bakery cashier")]
    assert job_service.semantic_filter_items(items, KEYWORDS, LOCATION, []) == items
    assert "cuda fell over" in capsys.readouterr().out


def test_the_input_list_is_not_mutated_by_filtering(model):
    items = [_item("Software Intern"), _item("Bakery cashier")]
    snapshot = list(items)
    job_service.semantic_filter_items(items, KEYWORDS, LOCATION, [])
    assert items == snapshot


# ── inference is serialised and thread-bounded ───────────────────────────────

def test_concurrent_channels_take_turns_at_the_model(monkeypatch):
    """The model parallelises inside a call; two calls at once only thrash
    the same cores. Four channels, one inside at any moment."""
    fake = _FakeModel(hold=0.05)
    monkeypatch.setattr(job_service, "load_semantic_plugin_model", lambda: fake)
    items = [_item("Software Intern")]

    threads = [threading.Thread(target=job_service.semantic_filter_items,
                                args=(list(items), KEYWORDS, LOCATION, [])) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(fake.calls) == 4
    assert fake.peak == 1, f"{fake.peak} inferences ran at once"


def test_inference_threads_are_a_bounded_share_of_the_host(monkeypatch):
    monkeypatch.setattr(job_service.capacity, "cpu_limit", lambda: 12.0)
    assert job_service.semantic_inference_threads() == 4
    monkeypatch.setattr(job_service.capacity, "cpu_limit", lambda: 2.0)
    assert job_service.semantic_inference_threads() == 1
    monkeypatch.setattr(job_service.capacity, "cpu_limit", lambda: 64.0)
    assert job_service.semantic_inference_threads() == job_service.SEMANTIC_INFERENCE_THREADS_MAX


def test_loading_the_model_bounds_torch(monkeypatch):
    """set_num_threads is called with the bounded count, after the model
    loads; and the tokenizers' pool is switched off before the import."""
    seen: dict = {}
    fake_torch = types.ModuleType("torch")
    fake_torch.set_num_threads = lambda n: seen.setdefault("threads", n)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = lambda name: seen.setdefault("model", name) or object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.setattr(job_service, "semantic_plugin_available", lambda: True)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

    assert job_service.load_semantic_plugin_model() is not None
    assert seen["threads"] == job_service.semantic_inference_threads()
    assert job_service.os.environ["TOKENIZERS_PARALLELISM"] == "false"


# ── the watcher: already-sent links never reach the model ───────────────────

class _Channel:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, content=None, embeds=None) -> None:
        self.sent.append(str(content or ""))


class _Client:
    def __init__(self, channel) -> None:
        self._channel = channel

    def get_channel(self, channel_id):
        return self._channel

    async def fetch_channel(self, channel_id):
        return self._channel


async def _one_iteration(manager: WatcherManager, store: RuntimeStore, channel_id: int) -> None:
    task = asyncio.create_task(manager._run_job_watcher(channel_id))
    await asyncio.sleep(0)
    store.channel_job_settings[channel_id]["enabled"] = False

    def _done() -> bool:
        h = manager.health.get_job_health(channel_id)
        return h is not None and h.total_scrapes >= 1

    deadline = time.monotonic() + 15
    while not _done() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    finished = _done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished, "the watcher never completed an iteration"


def test_the_watcher_drops_already_sent_links_before_the_model_sees_them(tmp_path, monkeypatch):
    channel_id = 4242
    store = RuntimeStore(tmp_path / ".bot_state.json")
    store.channel_job_settings = {channel_id: {
        "enabled": True, "sites": ["all"], "keywords": KEYWORDS, "location": LOCATION,
        "hours_old": 24, "results_wanted": 10, "radius_miles": 25, "role_filters": [],
        "exclusion_terms": [], "semantic_threshold": 0.1, "refresh_seconds": 60,
    }}
    store.channel_job_seen = {channel_id: {"https://example.invalid/job/old"}}
    channel = _Channel()
    manager = WatcherManager(
        client=_Client(channel),
        config=SimpleNamespace(base_dir=tmp_path, dedup_months_threshold=2,
                               discord_history_check_limit=5, jobspy_python_exe=None),
        store=store, health=WatcherHealthTracker(),
    )
    monkeypatch.setattr("services.jba.merge_data.log_jobs", lambda jobs, date_str=None: len(jobs))

    async def _exists(channel_id: int) -> bool:
        return True

    monkeypatch.setattr(manager, "_channel_exists", _exists)
    monkeypatch.setattr(job_service, "scrape_job_postings", lambda *a, **k: [
        {"site_label": "Glassdoor", "title": "Software Intern", "link": "https://example.invalid/job/old", "company": "Acme"},
        {"site_label": "Glassdoor", "title": "Software Intern", "link": "https://example.invalid/job/new", "company": "Acme"},
        {"site_label": "Glassdoor", "title": "Software Intern", "link": "", "company": "Acme"},
    ])
    judged: list[list[str]] = []

    def _recording_filter(items, *a, **k):
        judged.append([str(i.get("link")) for i in items])
        return list(items)

    monkeypatch.setattr(job_service, "semantic_filter_items", _recording_filter)

    asyncio.run(_one_iteration(manager, store, channel_id))

    assert judged == [["https://example.invalid/job/new"]], (
        "the model must see only what has not been sent and has a link"
    )
    assert len(channel.sent) == 1 and "job/new" in channel.sent[0]
    assert "https://example.invalid/job/new" in store.channel_job_seen[channel_id]
