"""Profile-vs-archive job ranking.

These drive a real profile directory written to tmp_path and real archive
records fed through a patched load_daily_log, so the assertions are about
actual parsing/scoring/dedup behaviour rather than mocks agreeing with
themselves. The one thing genuinely stubbed is the LLM transport -- the judge
call is exercised through the real prompt builder and the real response parser,
with only the network hop replaced.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import job_match
from services.jba import description_cache


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """Fail every un-stubbed posting fetch instantly instead of dialling out.

    Enrichment offers the whole shortlist to the fetcher, so any test that
    builds jobs with links and does not stub the scraper was making real
    requests to example.com -- one of them took 20 seconds on its own and the
    module's suite went from 7s to 98s. Tests must never depend on the network
    being there, or on it being absent.

    Tests that want a working fetch call _stub_scraper, whose monkeypatch
    replaces this one.
    """
    def offline(url, timeout_seconds=20, user_agent="", allow_browser=True):
        raise RuntimeError(f"network disabled in tests: {url}")

    monkeypatch.setattr("services.resumes.listing.scrape_job_posting", offline)


@pytest.fixture(autouse=True)
def isolated_description_cache(tmp_path, monkeypatch):
    """Point the description store at a throwaway database for every test.

    Not optional hygiene: the store lives inside the real data/jba/jobs/jobs.db,
    so without this the suite writes its fixture URLs into the same file the bot
    serves from -- and then later runs read those fixtures back and score
    against them. Caught exactly that way, having polluted the live database
    with 66 rows of example.com postings.
    """
    monkeypatch.setattr(description_cache, "_DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(description_cache, "_local", threading.local())
    yield
    connection = getattr(description_cache._local, "conn", None)
    if connection is not None:
        connection.close()


BASEINFO = """Candidate profile facts (source of truth for all resume content):

- Name: Test Candidate
- Program: Bachelor of Engineering in Computer Engineering (Toronto Metropolitan University, 2023-2027)

== EXPERIENCE ==

[ml] Machine Learning Intern, Goopter Commerce Solutions - Remote, Sep 2025 - Dec 2025
Notes:
Scale: streamed Kafka market data into asynchronous Python services.

[electrical] Electrical Team Member, Rocket Society - Toronto ON, Sep 2023 - Present
Notes:
Scope: STM32-based avionics flight computer.

[general] Technical Development Intern, City of Markham - Markham ON, Jul 2022 - Aug 2024
Notes:
Scope: documentation and milestone reporting.

== PROJECTS ==

[embedded] Mobile Robot System (Assembly, C)
Scope: line-following robot.

== SKILL ANCHORS ==
Languages: Python, C++, C, SQL, VHDL
Frameworks/Libraries: PyTorch Lightning, Flask, NumPy
Tools/Platforms: Git, Docker, Kafka, STM32, PCB
"""


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "testcandidate"
    directory.mkdir()
    (directory / "baseinfo.txt").write_text(BASEINFO, encoding="utf-8")
    return directory


@pytest.fixture
def signal(profile_dir: Path) -> job_match.ProfileSignal:
    return job_match.build_profile_signal(profile_dir, today=date(2026, 8, 25))


def _job(title: str, company: str = "Acme", location: str = "Toronto, ON", **kwargs) -> job_match.ArchivedJob:
    return job_match.ArchivedJob(
        title=title,
        company=company,
        location=location,
        link=kwargs.get("link", f"https://example.com/{abs(hash(title)) % 10**8}"),
        site_label=kwargs.get("site_label", "LinkedIn"),
        date_posted=kwargs.get("date_posted", "2026-08-24"),
        description=kwargs.get("description", ""),
    )


# ---------------------------------------------------------------------------
# Profile parsing
# ---------------------------------------------------------------------------

def test_profile_signal_reads_anchors_roles_locations_and_seniority(signal):
    assert "python" in signal.anchors
    assert "c++" in signal.anchors
    assert "stm32" in signal.anchors

    assert "machine learning intern" in signal.role_terms
    assert "electrical" in signal.role_terms
    assert "embedded" in signal.role_terms

    assert "toronto" in signal.locations
    assert "markham" in signal.locations
    assert "remote" in signal.locations

    # The degree runs to 2027, which is still ahead of the reference date.
    assert signal.seniority == "student"


def test_generic_tag_is_not_a_role_term(signal):
    """[general] names no domain, so a title containing "general" must not
    claim a full role match off it."""
    assert "general" not in signal.role_terms
    # The header on that same entry is still a usable role.
    assert "technical development intern" in signal.role_terms


def test_seniority_is_professional_when_no_degree_is_in_progress(tmp_path: Path):
    directory = tmp_path / "grad"
    directory.mkdir()
    (directory / "baseinfo.txt").write_text(
        "- Program: BEng Computer Engineering (TMU, 2015-2019)\n"
        "== EXPERIENCE ==\n[be] Backend Engineer, Acme - Toronto ON, 2019-2024\n",
        encoding="utf-8",
    )
    signal = job_match.build_profile_signal(directory, today=date(2026, 8, 25))
    assert signal.seniority == "professional"


def test_missing_baseinfo_yields_an_empty_signal(tmp_path: Path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    signal = job_match.build_profile_signal(empty, today=date(2026, 8, 25))
    assert signal.is_empty
    assert signal.document_text == ""


# ---------------------------------------------------------------------------
# Component scoring
# ---------------------------------------------------------------------------

def test_skills_score_counts_profile_tools_named_by_the_posting(signal):
    score, matched = job_match.score_skills(
        "embedded firmware co-op working on stm32 and pcb bring-up in c++".lower(), signal
    )
    assert {"stm32", "pcb", "c++"} <= set(matched)
    assert score > 0.5

    zero, none_matched = job_match.score_skills("registered nurse, night shift", signal)
    assert zero == 0.0
    assert none_matched == ()


def test_anchor_matching_respects_word_boundaries(signal):
    """"c" must not fire on "logistics", and "c++" must match despite the
    plus signs having no regex word boundary after them."""
    _, matched = job_match.score_skills("logistics coordinator for a pythonic startup", signal)
    assert "c" not in matched
    assert "python" not in matched

    _, matched_cpp = job_match.score_skills("c++ developer", signal)
    assert "c++" in matched_cpp


def test_role_score_rewards_the_candidates_own_domain(signal):
    ml_score, ml_matched = job_match.score_role("Machine Learning Engineer", signal)
    unrelated_score, _ = job_match.score_role("Dental Hygienist", signal)

    assert ml_score > unrelated_score
    assert "machine learning intern" in ml_matched


def test_level_word_alone_is_not_a_role_match(signal):
    """"intern" is scored by score_level; letting it also drive the role score
    would make every internship look like the candidate's own field."""
    score, matched = job_match.score_role("Sales Intern", signal)
    assert score == 0.0
    assert matched == ()


def test_level_score_prefers_junior_postings_for_a_student(signal):
    assert job_match.score_level("Software Engineering Intern", signal) == 1.0
    assert job_match.score_level("Senior Staff Engineer", signal) < 0.2
    # Nothing stated either way, and both stated at once, are both neutral.
    assert job_match.score_level("Software Engineer", signal) == 0.5
    assert job_match.score_level("Senior Intern Program Lead", signal) == 0.5


def test_level_score_inverts_for_a_professional_profile(tmp_path: Path):
    directory = tmp_path / "pro"
    directory.mkdir()
    (directory / "baseinfo.txt").write_text(
        "== EXPERIENCE ==\n[be] Backend Engineer, Acme - Toronto ON, 2015-2019\n", encoding="utf-8"
    )
    professional = job_match.build_profile_signal(directory, today=date(2026, 8, 25))

    assert professional.seniority == "professional"
    assert job_match.score_level("Senior Backend Engineer", professional) == 1.0
    assert job_match.score_level("Backend Intern", professional) < 0.5


def test_location_score_ranks_home_city_above_elsewhere(signal):
    assert job_match.score_location("Toronto, ON, CA", signal) == 1.0
    assert job_match.score_location("Markham, ON", signal) == 1.0
    assert job_match.score_location("Austin, TX", signal) == 0.3
    # Unknown location is neutral, not a penalty.
    assert job_match.score_location("", signal) == 0.5


def test_remote_counts_as_a_home_location_when_the_profile_has_remote_work(signal):
    """This profile holds a remote role, so remote postings are a full match,
    not the consolation score."""
    assert "remote" in signal.locations
    assert job_match.score_location("Remote", signal) == 1.0


def test_remote_is_a_partial_match_for_an_onsite_only_profile(tmp_path: Path):
    directory = tmp_path / "onsite"
    directory.mkdir()
    (directory / "baseinfo.txt").write_text(
        "== EXPERIENCE ==\n[be] Backend Engineer, Acme - Toronto ON, 2015-2019\n", encoding="utf-8"
    )
    onsite = job_match.build_profile_signal(directory, today=date(2026, 8, 25))

    assert "remote" not in onsite.locations
    assert job_match.score_location("Remote", onsite) == 0.9
    assert job_match.score_location("Toronto, ON", onsite) == 1.0


def test_prefilter_ranks_a_matching_internship_above_an_unrelated_senior_role(signal):
    good = job_match.score_job(_job("Embedded Software Engineering Intern (STM32)"), signal)
    bad = job_match.score_job(_job("Senior Physical Therapist", location="Austin, TX"), signal)

    assert good.total > bad.total
    assert 0.0 <= bad.total <= 1.0 and 0.0 <= good.total <= 1.0


# ---------------------------------------------------------------------------
# Window loading and dedup
# ---------------------------------------------------------------------------

def test_window_dates_counts_back_from_the_end_date():
    assert job_match.window_dates(1, date(2026, 8, 25)) == ["2026-08-25"]
    week = job_match.window_dates(7, date(2026, 8, 25))
    assert week[0] == "2026-08-25" and week[-1] == "2026-08-19"
    assert len(week) == 7


def _patch_archive(monkeypatch, days: dict[str, list[dict]]):
    monkeypatch.setattr(
        "services.jba.merge_data.load_daily_log", lambda date_key: days.get(date_key, [])
    )


def test_load_window_reads_both_archive_record_shapes(monkeypatch):
    """The ATS scraper writes job_url/_source_site; the watcher send path writes
    link/site_label. Both are in the archive and both must load."""
    _patch_archive(monkeypatch, {
        "2026-08-25": [
            {"title": "ATS Row", "company": "A", "location": "Toronto",
             "job_url": "https://ats.example/1", "_source_site": "greenhouse"},
            {"title": "Watcher Row", "company": "B", "link": "https://w.example/2",
             "site_label": "LinkedIn", "date_posted": "2026-08-25"},
        ],
    })
    jobs, dates = load = job_match.load_window_jobs(1, end_date=date(2026, 8, 25))

    assert dates == ["2026-08-25"]
    assert {job.title for job in jobs} == {"ATS Row", "Watcher Row"}
    assert {job.link for job in jobs} == {"https://ats.example/1", "https://w.example/2"}
    assert {job.site_label for job in jobs} == {"greenhouse", "LinkedIn"}


def test_same_posting_on_several_days_is_returned_once(monkeypatch):
    record = {"title": "Embedded Intern", "company": "Kepler",
              "link": "https://example.com/job/1", "site_label": "LinkedIn"}
    _patch_archive(monkeypatch, {
        "2026-08-25": [dict(record, date_posted="2026-08-25")],
        "2026-08-24": [dict(record, date_posted="2026-08-24")],
        "2026-08-23": [dict(record, date_posted="2026-08-23")],
    })
    jobs, _ = job_match.load_window_jobs(7, end_date=date(2026, 8, 25))

    assert len(jobs) == 1
    # Newest-first walk means the surviving copy is the most recent sighting.
    assert jobs[0].date_posted == "2026-08-25"


def test_same_role_under_different_urls_is_returned_once(monkeypatch):
    """Boards mint one listing per city for the same job; they render
    identically to the user apart from location."""
    _patch_archive(monkeypatch, {
        "2026-08-25": [
            {"title": "Co-op Electrical Intern", "company": "G&A ROBOT",
             "location": "Burnaby", "link": "https://example.com/a"},
            {"title": "Co-op Electrical Intern", "company": "G&A ROBOT",
             "location": "Canada", "link": "https://example.com/b"},
        ],
    })
    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 25))
    assert len(jobs) == 1


def test_records_without_a_title_are_dropped(monkeypatch):
    _patch_archive(monkeypatch, {
        "2026-08-25": [
            {"company": "Acme", "link": "https://example.com/a"},
            {"title": "  ", "link": "https://example.com/b"},
            {"title": "Real Job", "link": "https://example.com/c"},
        ],
    })
    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 25))
    assert [job.title for job in jobs] == ["Real Job"]


def test_a_failing_day_does_not_abort_the_window(monkeypatch):
    """One corrupt zip must cost that day, not the whole command."""
    def flaky(date_key: str):
        if date_key == "2026-08-24":
            raise zipfile_error()
        return [{"title": f"Job {date_key}", "link": f"https://example.com/{date_key}"}]

    def zipfile_error():
        return RuntimeError("bad zip")

    monkeypatch.setattr("services.jba.merge_data.load_daily_log", flaky)
    jobs, _ = job_match.load_window_jobs(3, end_date=date(2026, 8, 25))

    assert [job.title for job in jobs] == ["Job 2026-08-25", "Job 2026-08-23"]


def test_window_load_stops_at_the_record_cap(monkeypatch):
    _patch_archive(monkeypatch, {
        "2026-08-25": [
            {"title": f"Job {index}", "link": f"https://example.com/{index}"}
            for index in range(50)
        ],
    })
    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 25), max_records=10)
    assert len(jobs) == 10


def test_mojibake_in_archived_titles_is_repaired(monkeypatch):
    _patch_archive(monkeypatch, {
        "2026-08-25": [{
            "title": "Stagiaire en dÃ©veloppement IA",
            "company": "Autodesk", "link": "https://example.com/a",
        }],
    })
    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 25))
    assert "développement" in jobs[0].title


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

def test_judge_prompt_carries_the_profile_and_every_candidate(signal):
    jobs = [_job("Embedded Intern"), _job("ML Intern"), _job("Sales Lead")]
    prompt = job_match.build_judge_prompt(signal, jobs)

    assert "Machine Learning Intern" in prompt  # profile went in
    items = _prompt_items(prompt)
    assert [item["id"] for item in items] == [0, 1, 2]
    for index, job in enumerate(jobs):
        assert items[index]["title"] == job.title
        assert items[index]["company"] == job.company


def test_the_posting_table_is_cheaper_than_pretty_printed_json(signal):
    """The rows replaced JSON with indent=1, which repeated every field name
    once per posting. The saving is the point of the format, so it is pinned:
    a regression here is a silently more expensive call, not a broken one."""
    jobs = [_job(f"Engineer {index}") for index in range(30)]
    prompt = job_match.build_judge_prompt(signal, jobs)

    as_json = json.dumps(
        [
            {"id": index, "title": job.title, "company": job.company,
             "location": job.location}
            for index, job in enumerate(jobs)
        ],
        indent=1,
    )
    table = prompt[prompt.index(job_match.POSTING_TABLE_HEADER):]
    assert len(table) < len(as_json) * 0.75


def test_a_pipe_in_a_field_cannot_break_the_row(signal):
    """Job titles do contain pipes ("Intern | Summer 2027"), and one leaking
    through would shift every later column for that posting."""
    prompt = job_match.build_judge_prompt(
        signal, [_job("Intern | Summer 2027", company="Acme | Corp")]
    )
    item = _prompt_items(prompt)[0]

    assert item["title"] == "Intern / Summer 2027"
    assert item["company"] == "Acme / Corp"


def test_the_description_sent_to_the_judge_is_excerpted_not_head_sliced(signal):
    """A head slice keeps the company pitch and drops the qualifications,
    which are what the judge is being asked to reason about."""
    body = (
        "About us. We are an exciting company on a mission. " + ("Filler text. " * 90)
        + "Requirements: 3 years of experience with Python and Kafka."
    )
    assert len(body) > job_match.JUDGE_DESCRIPTION_CHARS
    job = _job("Data Intern", description=body)

    item = _prompt_items(job_match.build_judge_prompt(signal, [job]))[0]

    assert "Requirements: 3 years of experience" in item["description"]
    assert len(item["description"]) <= job_match.JUDGE_DESCRIPTION_CHARS
    assert body[: job_match.JUDGE_DESCRIPTION_CHARS] not in item["description"]


def test_parse_judge_response_scales_scores_and_keeps_reasons():
    parsed = job_match.parse_judge_response(
        '{"ranked": [{"id": 0, "score": 91, "reason": "embedded match"},'
        ' {"id": 1, "score": 12, "reason": "wrong field"}]}',
        count=2,
    )
    assert {index: (v.score, v.reason) for index, v in parsed.items()} == {
        0: (0.91, "embedded match"),
        1: (0.12, "wrong field"),
    }


def test_parse_judge_response_drops_bad_rows_but_keeps_good_ones():
    """One malformed row must not cost the user the whole ranking."""
    parsed = job_match.parse_judge_response(
        '{"ranked": [{"id": 99, "score": 90}, {"id": "x", "score": 5},'
        ' {"id": 1}, {"id": 0, "score": 80, "reason": "ok"},'
        ' {"id": 0, "score": 10, "reason": "duplicate id, ignored"}]}',
        count=2,
    )
    assert {index: (v.score, v.reason) for index, v in parsed.items()} == {0: (0.80, "ok")}


def test_parse_judge_response_rejects_unusable_payloads():
    assert job_match.parse_judge_response("not json at all", count=2) is None
    assert job_match.parse_judge_response('{"ranked": []}', count=2) is None
    assert job_match.parse_judge_response('{"other": [1]}', count=2) is None


def _prompt_items(prompt: str) -> list[dict]:
    """The candidate list the judge was actually shown, parsed back out of the
    prompt. Reading it (rather than assuming an order) is what makes the id
    mapping itself testable: ids are indices into the *prefiltered* shortlist,
    not into the caller's original list."""
    marker = job_match.POSTING_TABLE_HEADER + "\n"
    body = prompt[prompt.index(marker) + len(marker):]
    items = []
    for line in body.splitlines():
        if not line.strip():
            continue
        cells = line.split("|")
        item = {
            "id": int(cells[0]),
            "title": cells[1],
            "company": cells[2],
            "location": cells[3],
        }
        if len(cells) > 4:
            item["description"] = "|".join(cells[4:])
        items.append(item)
    return items


def _stub_judge(monkeypatch, scores: dict[str, int] | None, capture: dict | None = None):
    """Replace only the network hop; the real prompt builder and the real
    response parser still run. Verdicts are keyed by job title so a test states
    which job it is scoring rather than guessing its shortlist position."""
    def fake(prompt, settings, validate, client_factory=None, **kwargs):
        if capture is not None:
            capture["prompt"] = prompt
            capture["kwargs"] = kwargs
        if scores is None:
            return None, None
        payload = json.dumps({
            "ranked": [
                {"id": item["id"], "score": scores[item["title"]],
                 "reason": f"reason for {item['title']}"}
                for item in _prompt_items(prompt)
                if item["title"] in scores
            ]
        })
        return validate(payload), "gemini-flash"

    monkeypatch.setattr(
        "services.resumes.listing.generate_validated_with_providers", fake
    )


def test_judge_verdicts_override_the_prefilter_order(monkeypatch, signal):
    """The whole point of the LLM stage: it must be able to promote a job the
    keyword prefilter ranked last."""
    jobs = [
        _job("Embedded Software Engineering Intern STM32 C++"),  # prefilter favourite
        _job("Avionics Firmware Co-op"),                          # no keyword overlap
    ]
    prefilter = sorted(
        (job_match.score_job(job, signal).total for job in jobs), reverse=True
    )
    assert prefilter[0] > prefilter[1]

    _stub_judge(monkeypatch, {
        "Embedded Software Engineering Intern STM32 C++": 40,
        "Avionics Firmware Co-op": 95,
    })
    report = job_match.rank_jobs(signal, jobs, settings=object(), limit=5)

    assert report.judged
    assert report.ranker == "ok:gemini-flash"
    assert report.matches[0].job.title == "Avionics Firmware Co-op"
    assert report.matches[0].score.total == 0.95
    assert report.matches[0].score.llm_reason == "reason for Avionics Firmware Co-op"


def test_judge_call_asks_for_json_from_the_lowest_priority_provider(monkeypatch, signal):
    """Ranking is bulk work nobody is blocked on, so it must not spend the
    quota the interactive resume commands depend on."""
    capture: dict = {}
    _stub_judge(monkeypatch, {"Embedded Intern": 50}, capture)
    job_match.rank_jobs(signal, [_job("Embedded Intern")], settings=object(), limit=5)

    assert capture["kwargs"]["json_response"] is True
    assert capture["kwargs"]["lowest_priority_first"] is True


def test_unjudged_jobs_sort_below_judged_ones(monkeypatch, signal):
    """A job the judge skipped has not been shown to beat one it actually
    looked at, so it must not displace it."""
    jobs = [_job("Embedded Intern"), _job("Registered Nurse"), _job("ML Intern")]
    _stub_judge(monkeypatch, {"Registered Nurse": 30})  # only the nurse gets a verdict

    report = job_match.rank_jobs(signal, jobs, settings=object(), limit=5)

    assert report.matches[0].job.title == "Registered Nurse"
    assert report.matches[0].score.llm_score == 0.30
    assert {match.job.title for match in report.matches[1:]} == {"Embedded Intern", "ML Intern"}
    assert all(match.score.llm_score is None for match in report.matches[1:])


def test_ranking_falls_back_to_the_prefilter_when_every_provider_fails(monkeypatch, signal):
    _stub_judge(monkeypatch, None)
    jobs = [_job("Registered Nurse", location="Austin, TX"), _job("Embedded Intern STM32")]

    report = job_match.rank_jobs(signal, jobs, settings=object(), limit=5)

    assert not report.judged
    assert report.matches[0].job.title == "Embedded Intern STM32"
    assert all(match.score.llm_score is None for match in report.matches)


def test_judge_transport_exception_is_contained(monkeypatch, signal):
    def boom(*args, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr("services.resumes.listing.generate_validated_with_providers", boom)
    report = job_match.rank_jobs(signal, [_job("Embedded Intern")], settings=object(), limit=5)

    assert not report.judged
    assert "connection reset" in report.ranker
    assert len(report.matches) == 1


def test_no_settings_means_no_judge_call(monkeypatch, signal):
    def fail(*args, **kwargs):
        raise AssertionError("must not call a provider without settings")

    monkeypatch.setattr("services.resumes.listing.generate_validated_with_providers", fail)
    report = job_match.rank_jobs(signal, [_job("Embedded Intern")], settings=None, limit=5)

    assert not report.judged
    assert report.matches


def test_only_the_shortlist_reaches_the_judge(monkeypatch, signal):
    capture: dict = {}
    _stub_judge(monkeypatch, {"Engineer 0": 50}, capture)
    jobs = [_job(f"Engineer {index}") for index in range(job_match.LLM_CANDIDATES + 25)]

    report = job_match.rank_jobs(signal, jobs, settings=object(), limit=5)

    assert len(_prompt_items(capture["prompt"])) == job_match.LLM_CANDIDATES
    assert report.scanned == len(jobs)


# ---------------------------------------------------------------------------
# Limits and rendering
# ---------------------------------------------------------------------------

def test_limit_caps_results_and_is_clamped_to_the_maximum(monkeypatch, signal):
    _stub_judge(monkeypatch, None)
    jobs = [_job(f"Engineer {index}") for index in range(80)]

    assert len(job_match.rank_jobs(signal, jobs, limit=3).matches) == 3
    assert len(job_match.rank_jobs(signal, jobs, limit=999).matches) == job_match.MAX_LIMIT
    assert len(job_match.rank_jobs(signal, jobs, limit=0).matches) == 1


def test_report_shows_the_judges_reason_when_it_ranked(monkeypatch, signal):
    _stub_judge(monkeypatch, {"Embedded Intern": 88})
    report = job_match.rank_jobs(
        signal, [_job("Embedded Intern", company="Kepler")],
        settings=object(), limit=5, dates=["2026-08-25"],
    )
    text = job_match.format_match_report(report, "day")

    assert "ranked by gemini-flash" in text
    assert "88%" in text
    # Board and employer lead the line, named once; the location follows the
    # title rather than repeating the company.
    assert "[LinkedIn/Kepler] Embedded Intern (Toronto, ON)" in text
    # The judge's reasoning still drives the score and is still on the score
    # object, but the channel output is the listing and the percentage only.
    assert report.matches[0].score.llm_reason == "reason for Embedded Intern"
    assert "reason for Embedded Intern" not in text


def test_report_says_so_when_it_fell_back_to_keyword_ranking(monkeypatch, signal):
    """Serving the rougher ordering silently would misrepresent the scores."""
    _stub_judge(monkeypatch, None)
    report = job_match.rank_jobs(
        signal, [_job("Embedded Intern")], settings=object(), limit=5, dates=["2026-08-25"]
    )
    text = job_match.format_match_report(report, "day")

    assert "keyword ranking only" in text
    assert "ranked by" not in text


def test_report_renders_a_week_span_and_an_empty_window(signal):
    week = job_match.window_dates(7, date(2026, 8, 25))
    empty = job_match.MatchReport(matches=[], scanned=0, dates=week, profile_key="tester")
    text = job_match.format_match_report(empty, "week")

    assert "2026-08-19 to 2026-08-25" in text
    assert "No archived jobs in that window." in text


# ---------------------------------------------------------------------------
# Command dispatch
#
# These drive a real CommandRouter end to end: a message goes through the real
# handler tuple, the real payload parser, the real profile-key resolution and
# the real interactive scheduler, and the text the user would actually see is
# asserted on. Only best_jobs() itself is stubbed, so the archive and the LLM
# are not touched.
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

from commands.handlers import CMD_BEST_JOBS, CMD_BEST_JOBS_ALIAS, CommandRouter  # noqa: E402
from services.health import WatcherHealthTracker  # noqa: E402
from state.store import RuntimeStore  # noqa: E402
from watchers.manager import WatcherManager  # noqa: E402


class _FakeMessage:
    """The slice of discord.Message the best-jobs handler actually touches."""

    def __init__(self, content: str, channel, author_id: int, owner_id: int | None) -> None:
        self.content = content
        self.channel = channel
        self.author = type("Author", (), {"id": author_id, "name": "tester", "bot": False})()
        self.guild = None if owner_id is None else type("Guild", (), {"owner_id": owner_id})()
        self.id = 999

    def to_reference(self, fail_if_not_exists: bool = True):
        return object()


class _FakeChannel:
    def __init__(self, channel_id: int = 42) -> None:
        self.id = channel_id
        self.sent: list[str] = []
        self.deleted: list[str] = []

    async def send(self, content=None, **kwargs):
        self.sent.append(content or "")
        channel = self

        class _Sent:
            def __init__(self, text: str) -> None:
                self.text = text
                self.id = 1

            async def delete(self):
                channel.deleted.append(self.text)

            async def edit(self, content=None, **kw):
                self.text = content
                channel.sent.append(content or "")

        return _Sent(content or "")


class _Config:
    def __init__(self, tmp_path: Path) -> None:
        self.resume_profiles_dir = tmp_path / "resumes"
        self.resume_cache_dir = tmp_path / ".resume_cache"
        self.resume_profiles_dir.mkdir(parents=True, exist_ok=True)
        self.resume_cache_dir.mkdir(parents=True, exist_ok=True)
        self.main_user_profile_key = "owner-profile"
        self.base_dir = tmp_path
        self.discord_history_check_limit = 20


def _router(tmp_path: Path):
    config = _Config(tmp_path)
    store = RuntimeStore(tmp_path / ".bot_state.json")
    health = WatcherHealthTracker()
    manager = WatcherManager(client=object(), config=config, store=store, health=health)
    router = CommandRouter(
        client=object(), config=config, store=store, watcher_manager=manager, health=health
    )
    return router, _FakeChannel()


def _dispatch(router: CommandRouter, message) -> bool:
    async def drive():
        for handler in router.handlers:
            if await handler(message):
                return True
        return False

    return asyncio.run(drive())


def _stub_best_jobs(monkeypatch, capture: dict):
    def fake(profile_key, **kwargs):
        capture["profile_key"] = profile_key
        capture.update(kwargs)
        return job_match.MatchReport(
            matches=[job_match.JobMatch(
                job=_job("Embedded Intern", company="Kepler"),
                score=job_match.JobScore(total=0.9, llm_score=0.9, llm_reason="good fit"),
            )],
            scanned=1234,
            dates=["2026-08-25"],
            profile_key=str(profile_key),
            ranker="ok:gemini-flash",
        )

    monkeypatch.setattr(job_match, "best_jobs", fake)


def test_bestjobs_command_reports_matches_to_the_channel(tmp_path, monkeypatch):
    router, channel = _router(tmp_path)
    capture: dict = {}
    _stub_best_jobs(monkeypatch, capture)

    handled = _dispatch(router, _FakeMessage(
        f"{CMD_BEST_JOBS} week 5", channel, author_id=7, owner_id=7
    ))

    assert handled
    assert capture["window"] == "week"
    assert capture["limit"] == 5
    final = channel.sent[-1]
    assert "[LinkedIn/Kepler] Embedded Intern (Toronto, ON)" in final
    assert "ranked by gemini-flash" in final
    assert "good fit" not in final          # reasoning is not printed
    # The progress placeholder is cleaned up, not left behind.
    assert any("Best matches for the last" in text for text in channel.deleted)


def test_bestjobs_defaults_to_a_day_of_the_owner_profile(tmp_path, monkeypatch):
    router, channel = _router(tmp_path)
    capture: dict = {}
    _stub_best_jobs(monkeypatch, capture)

    _dispatch(router, _FakeMessage(CMD_BEST_JOBS, channel, author_id=7, owner_id=7))

    assert capture["window"] == "day"
    assert capture["limit"] == job_match.DEFAULT_LIMIT
    assert capture["profile_key"] == "owner-profile"


def test_bestjobs_short_alias_dispatches_too(tmp_path, monkeypatch):
    router, channel = _router(tmp_path)
    capture: dict = {}
    _stub_best_jobs(monkeypatch, capture)

    assert _dispatch(router, _FakeMessage(
        f"{CMD_BEST_JOBS_ALIAS} week", channel, author_id=7, owner_id=7
    ))
    assert capture["window"] == "week"


def test_bestjobs_reports_a_bad_argument_without_calling_the_ranker(tmp_path, monkeypatch):
    router, channel = _router(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("must not rank on invalid input")

    monkeypatch.setattr(job_match, "best_jobs", fail)
    assert _dispatch(router, _FakeMessage(
        f"{CMD_BEST_JOBS} 999", channel, author_id=7, owner_id=7
    ))
    assert "between 1 and" in channel.sent[-1]


def test_non_owner_cannot_target_another_profile(tmp_path, monkeypatch):
    """Same rule as the resume commands: a resume cache is not readable by
    whoever happens to type its folder name."""
    router, channel = _router(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("must not rank another user's profile")

    monkeypatch.setattr(job_match, "best_jobs", fail)
    assert _dispatch(router, _FakeMessage(
        f"{CMD_BEST_JOBS} someoneelse", channel, author_id=8, owner_id=7
    ))
    assert "Only the server owner" in channel.sent[-1]


def test_ranker_failure_is_reported_not_raised(tmp_path, monkeypatch):
    router, channel = _router(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("archive unreadable")

    monkeypatch.setattr(job_match, "best_jobs", boom)
    assert _dispatch(router, _FakeMessage(CMD_BEST_JOBS, channel, author_id=7, owner_id=7))
    assert any("archive unreadable" in text for text in channel.sent)


def test_long_report_is_paged_through_the_continuation_buffer(tmp_path, monkeypatch):
    """A 50-job report exceeds Discord's message cap, so it must land in the
    .more buffer rather than being truncated or rejected."""
    router, channel = _router(tmp_path)

    def fake(profile_key, **kwargs):
        return job_match.MatchReport(
            matches=[
                job_match.JobMatch(
                    job=_job(f"Engineer {index} " + "x" * 120, company="Acme"),
                    score=job_match.JobScore(total=0.5, llm_score=0.5, llm_reason="y" * 100),
                )
                for index in range(50)
            ],
            scanned=5000, dates=["2026-08-25"], profile_key="owner-profile",
            ranker="ok:gemini-flash",
        )

    monkeypatch.setattr(job_match, "best_jobs", fake)
    _dispatch(router, _FakeMessage(f"{CMD_BEST_JOBS} 50", channel, author_id=7, owner_id=7))

    assert len(channel.sent[-1]) <= 2000
    assert router.channel_continuations.get(channel.id)


def test_bestjobs_slash_form_dispatches_with_the_same_parsing(tmp_path, monkeypatch):
    """app.py re-dispatches /bestjobs as text, so the slash path must land on
    the same handler and the same argument parser as the dot form."""
    router, channel = _router(tmp_path)
    capture: dict = {}
    _stub_best_jobs(monkeypatch, capture)

    assert _dispatch(router, _FakeMessage(
        "/bestjobs week 5", channel, author_id=7, owner_id=7
    ))
    assert capture["window"] == "week"
    assert capture["limit"] == 5


def test_report_refuses_to_rank_for_a_profile_with_no_content(tmp_path):
    """An unseeded profile has nothing to match against, so a printed ranking
    would read as a real answer to a question that cannot be answered."""
    empty = tmp_path / "unseeded"
    empty.mkdir()
    signal = job_match.build_profile_signal(empty, today=date(2026, 8, 25))

    report = job_match.rank_jobs(signal, [_job("Embedded Intern")], limit=5)

    assert report.profile_empty
    text = job_match.format_match_report(report, "day")
    assert "no resume content" in text
    assert "Embedded Intern" not in text


def test_a_prose_only_profile_still_ranks(tmp_path):
    """No SKILL ANCHORS and no [tag] entries means a weak prefilter, but the
    judge reads prose fine -- that is not a reason to refuse."""
    prose = tmp_path / "prose"
    prose.mkdir()
    (prose / "baseinfo.txt").write_text(
        "Third-year computer engineering student who has written embedded "
        "firmware and trained models.\n", encoding="utf-8",
    )
    signal = job_match.build_profile_signal(prose, today=date(2026, 8, 25))

    assert signal.is_empty          # nothing for the prefilter to key on
    assert signal.document_text     # but plenty for the judge

    report = job_match.rank_jobs(signal, [_job("Embedded Intern")], limit=5)
    assert not report.profile_empty
    assert "Embedded Intern" in job_match.format_match_report(report, "day")


# ---------------------------------------------------------------------------
# Description enrichment
# ---------------------------------------------------------------------------

def _stub_scraper(monkeypatch, bodies: dict[str, str], *, slow: set[str] = frozenset(),
                  capture: dict | None = None):
    """Replace the network hop in the scrape ladder. Bodies are returned in the
    real wire shape -- the metadata preamble scrape_job_posting() prepends --
    so the preamble stripping is exercised rather than assumed."""
    import time as _time

    def fake(url, timeout_seconds=20, user_agent="", allow_browser=True):
        if capture is not None:
            capture.setdefault("urls", []).append(url)
            capture["allow_browser"] = allow_browser
        if url in slow:
            _time.sleep(2.0)
        if url not in bodies:
            raise RuntimeError("404")
        return type("Scraped", (), {
            "description": (
                f"Source site: test\nPosting URL: {url}\nTitle: T\n"
                f"Company: C\nLocation: L\nDescription:\n{bodies[url]}"
            ),
            "highlights": [],
        })()

    monkeypatch.setattr("services.resumes.listing.scrape_job_posting", fake)


def test_enrichment_fills_descriptions_and_strips_the_metadata_preamble(monkeypatch):
    jobs = [_job("Embedded Intern", link="https://example.com/a")]
    _stub_scraper(monkeypatch, {"https://example.com/a": "We need STM32 firmware experience."})

    filled = job_match.enrich_descriptions(jobs)

    assert filled == 1
    assert jobs[0].description == "We need STM32 firmware experience."
    # The preamble restates fields the judge prompt already carries.
    assert "Posting URL" not in jobs[0].description
    assert "Source site" not in jobs[0].description


def test_enrichment_never_queues_behind_the_shared_browser(monkeypatch):
    """There is one browser context and the Reddit watcher needs it; a bulk
    ranking pass must not be able to monopolise it."""
    jobs = [_job("Embedded Intern", link="https://example.com/a")]
    capture: dict = {}
    _stub_scraper(monkeypatch, {"https://example.com/a": "body"}, capture=capture)

    job_match.enrich_descriptions(jobs)
    assert capture["allow_browser"] is False


def test_a_failing_fetch_leaves_that_job_with_its_title(monkeypatch):
    """One dead board must cost one description, not the ranking."""
    jobs = [
        _job("Good", link="https://example.com/ok"),
        _job("Dead", link="https://example.com/gone"),
    ]
    _stub_scraper(monkeypatch, {"https://example.com/ok": "real text"})

    filled = job_match.enrich_descriptions(jobs)

    assert filled == 1
    assert jobs[0].description == "real text"
    assert jobs[1].description == ""


def test_enrichment_respects_its_time_budget(monkeypatch):
    """The budget is a promise to the user, not a target: whatever has come
    back when it expires is what the judge gets."""
    slow_url = "https://example.com/slow"
    jobs = [_job("Slow", link=slow_url)]
    _stub_scraper(monkeypatch, {slow_url: "text"}, slow={slow_url})

    started = time.monotonic()
    filled = job_match.enrich_descriptions(jobs, time_budget=0.3)
    elapsed = time.monotonic() - started

    assert filled == 0
    assert elapsed < 1.5


def test_jobs_that_already_have_a_description_are_not_refetched(monkeypatch):
    capture: dict = {}
    jobs = [
        _job("Has one", link="https://example.com/a", description="already here"),
        _job("Needs one", link="https://example.com/b"),
    ]
    _stub_scraper(monkeypatch, {"https://example.com/b": "fetched"}, capture=capture)

    job_match.enrich_descriptions(jobs)

    assert capture["urls"] == ["https://example.com/b"]
    assert jobs[0].description == "already here"


def test_records_without_a_link_are_skipped(monkeypatch):
    capture: dict = {}
    jobs = [job_match.ArchivedJob(
        title="No link", company="A", location="", link="",
        site_label="x", date_posted="2026-08-25",
    )]
    _stub_scraper(monkeypatch, {}, capture=capture)

    assert job_match.enrich_descriptions(jobs) == 0
    assert capture.get("urls") is None


def test_enriched_descriptions_reach_the_judge_prompt(signal, monkeypatch):
    """The whole point of the stage: the judge must actually see the text."""
    jobs = [_job("Embedded Intern", link="https://example.com/a")]
    _stub_scraper(monkeypatch, {"https://example.com/a": "STM32 firmware and PCB bring-up."})
    job_match.enrich_descriptions(jobs)

    prompt = job_match.build_judge_prompt(signal, jobs)
    assert "STM32 firmware and PCB bring-up." in prompt


def test_only_the_top_candidates_are_enriched_but_all_are_judged(signal, monkeypatch):
    """Enrichment costs a round trip each, so it is capped -- but a job that
    was not enriched must still reach the judge."""
    jobs = [_job(f"Engineer {i}", link=f"https://example.com/{i}")
            for i in range(job_match.ENRICH_CANDIDATES + 10)]
    capture: dict = {}
    _stub_scraper(
        monkeypatch,
        {f"https://example.com/{i}": f"body {i}" for i in range(len(jobs))},
        capture=capture,
    )
    _stub_judge(monkeypatch, {"Engineer 0": 50}, capture)

    report = job_match.rank_jobs(signal, jobs, settings=object(), limit=5)

    assert report.enriched == job_match.ENRICH_CANDIDATES
    assert len(_prompt_items(capture["prompt"])) == len(jobs)


def test_fast_mode_skips_the_fetch_entirely(signal, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("--fast must not fetch descriptions")

    monkeypatch.setattr("services.resumes.listing.scrape_job_posting", fail)
    _stub_judge(monkeypatch, {"Embedded Intern": 70})

    report = job_match.rank_jobs(
        signal, [_job("Embedded Intern", link="https://example.com/a")],
        settings=object(), limit=5, enrich=False,
    )
    assert report.enriched == 0
    assert report.judged


def test_report_states_how_many_descriptions_were_fetched(signal, monkeypatch):
    _stub_scraper(monkeypatch, {"https://example.com/a": "real posting text"})
    _stub_judge(monkeypatch, {"Embedded Intern": 70})

    report = job_match.rank_jobs(
        signal, [_job("Embedded Intern", link="https://example.com/a")],
        settings=object(), limit=5, dates=["2026-08-25"],
    )
    text = job_match.format_match_report(report, "day")

    assert report.enriched == 1
    assert "1 with full descriptions" in text


# ---------------------------------------------------------------------------
# Default profile resolution
#
# The same real router/handler path as above, but with a real resume cache on
# disk under tmp_path: the profile key the ranker is handed comes out of real
# directory lookups, not a stub.
# ---------------------------------------------------------------------------

import discord  # noqa: E402

import commands.handlers as handlers  # noqa: E402


def _seed_profile(cache_root: Path, key: str, baseinfo: str = "") -> Path:
    profile_dir = cache_root / key
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "baseinfo.txt").write_text(baseinfo, encoding="utf-8")
    (profile_dir / "instructions.txt").write_text("", encoding="utf-8")
    (profile_dir / "template.tex").write_text("", encoding="utf-8")
    return profile_dir


class _FakeMember:
    def __init__(self, user_id: int, name: str) -> None:
        self.id = user_id
        self.name = name


class _FakeGuild:
    """A guild whose members are uncached -- the Intents.default() case."""

    def __init__(self, owner_id: int, members: dict, cached: bool = False) -> None:
        self.owner_id = owner_id
        self._members = {uid: _FakeMember(uid, name) for uid, name in members.items()}
        self.owner = self._members.get(owner_id) if cached else None
        self.fetched: list = []

    def get_member(self, user_id: int):
        return self._members.get(user_id) if self.owner is not None else None

    async def fetch_member(self, user_id: int):
        self.fetched.append(user_id)
        member = self._members.get(user_id)
        if member is None:
            raise discord.NotFound(
                type("R", (), {"status": 404, "reason": "Not Found"})(), "unknown member"
            )
        return member


class _GuildMessage(_FakeMessage):
    def __init__(self, content: str, channel, author_id: int, guild) -> None:
        super().__init__(content, channel, author_id=author_id, owner_id=guild.owner_id)
        self.guild = guild


def _cache_router(tmp_path: Path, monkeypatch, main_user_profile_key=None):
    router, channel = _router(tmp_path)
    router.config.main_user_profile_key = main_user_profile_key or "example"
    cache_root = tmp_path / "resumes_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(handlers, "RESUMES_CACHE_ROOT", cache_root)
    return router, channel, cache_root


def test_bestjobs_defaults_to_the_channel_owners_own_profile(tmp_path, monkeypatch):
    router, channel, cache_root = _cache_router(tmp_path, monkeypatch)
    _seed_profile(cache_root, "example")
    _seed_profile(cache_root, "xboxsignout._", "Embedded systems, C, Python")
    _seed_profile(cache_root, "someone_else", "Nursing")
    capture: dict = {}
    _stub_best_jobs(monkeypatch, capture)

    guild = _FakeGuild(owner_id=7, members={7: "xboxsignout._", 9: "someone_else"})
    assert _dispatch(router, _GuildMessage(CMD_BEST_JOBS, channel, author_id=7, guild=guild))

    # Resolved through the owner's username, not their numeric id, and not the
    # empty "example" seed profile.
    assert capture["profile_key"] == "xboxsignout._"
    assert guild.fetched == [7], "must fall back to an HTTP fetch without the members intent"


def test_bestjobs_never_substitutes_another_members_profile_for_the_owners(tmp_path, monkeypatch):
    router, channel, cache_root = _cache_router(tmp_path, monkeypatch)
    _seed_profile(cache_root, "example")
    _seed_profile(cache_root, "owner_name")           # seeded but 0-byte baseinfo
    _seed_profile(cache_root, "someone_else", "Firmware engineer, RTOS")

    def fail(*args, **kwargs):
        raise AssertionError("must not rank another member's resume as the default")

    monkeypatch.setattr(job_match, "best_jobs", fail)
    guild = _FakeGuild(owner_id=7, members={7: "owner_name"})
    _dispatch(router, _GuildMessage(CMD_BEST_JOBS, channel, author_id=7, guild=guild))

    # The owner is known and their profile is simply empty. Ranking the only
    # other member who happens to have a resume would show their matches to the
    # whole channel, so say what is wrong instead.
    assert "No seeded resume profile" in channel.sent[-1]


def test_bestjobs_falls_back_to_the_lone_profile_when_the_owner_is_unknown(tmp_path, monkeypatch):
    router, channel, cache_root = _cache_router(tmp_path, monkeypatch)
    _seed_profile(cache_root, "example")
    _seed_profile(cache_root, "realprofile", "Firmware engineer, RTOS")
    capture: dict = {}
    _stub_best_jobs(monkeypatch, capture)

    # A guild whose owner the bot cannot resolve to a member at all.
    guild = _FakeGuild(owner_id=7, members={})
    _dispatch(router, _GuildMessage(CMD_BEST_JOBS, channel, author_id=7, guild=guild))

    assert capture["profile_key"] == "realprofile"


def test_bestjobs_default_prefers_an_explicitly_configured_profile(tmp_path, monkeypatch):
    router, channel, cache_root = _cache_router(
        tmp_path, monkeypatch, main_user_profile_key="pinned"
    )
    _seed_profile(cache_root, "pinned", "Pinned resume text")
    _seed_profile(cache_root, "owner_name", "Owner resume text")
    capture: dict = {}
    _stub_best_jobs(monkeypatch, capture)

    guild = _FakeGuild(owner_id=7, members={7: "owner_name"})
    _dispatch(router, _GuildMessage(CMD_BEST_JOBS, channel, author_id=7, guild=guild))

    assert capture["profile_key"] == "pinned"


def test_bestjobs_default_uses_a_cached_owner_without_fetching(tmp_path, monkeypatch):
    router, channel, cache_root = _cache_router(tmp_path, monkeypatch)
    _seed_profile(cache_root, "owner_name", "Owner resume text")
    _seed_profile(cache_root, "other", "Other resume text")
    capture: dict = {}
    _stub_best_jobs(monkeypatch, capture)

    guild = _FakeGuild(owner_id=7, members={7: "owner_name"}, cached=True)
    _dispatch(router, _GuildMessage(CMD_BEST_JOBS, channel, author_id=7, guild=guild))

    assert capture["profile_key"] == "owner_name"
    assert guild.fetched == []


def test_bestjobs_owner_can_target_another_member_by_mention(tmp_path, monkeypatch):
    router, channel, cache_root = _cache_router(tmp_path, monkeypatch)
    _seed_profile(cache_root, "owner_name", "Owner resume text")
    _seed_profile(cache_root, "shane_itwarie", "Mechanical engineering")
    capture: dict = {}
    _stub_best_jobs(monkeypatch, capture)

    guild = _FakeGuild(owner_id=7, members={7: "owner_name", 9: "shane_itwarie"})
    _dispatch(router, _GuildMessage(
        CMD_BEST_JOBS + " week <@9>", channel, author_id=7, guild=guild
    ))

    assert capture["profile_key"] == "shane_itwarie"
    assert capture["window"] == "week"


def test_bestjobs_owner_can_target_another_member_by_folder_name(tmp_path, monkeypatch):
    router, channel, cache_root = _cache_router(tmp_path, monkeypatch)
    _seed_profile(cache_root, "owner_name", "Owner resume text")
    _seed_profile(cache_root, "shane_itwarie", "Mechanical engineering")
    capture: dict = {}
    _stub_best_jobs(monkeypatch, capture)

    guild = _FakeGuild(owner_id=7, members={7: "owner_name"})
    _dispatch(router, _GuildMessage(
        CMD_BEST_JOBS + " shane_itwarie", channel, author_id=7, guild=guild
    ))

    assert capture["profile_key"] == "shane_itwarie"


def test_bestjobs_non_owner_cannot_target_someone_elses_profile(tmp_path, monkeypatch):
    router, channel, cache_root = _cache_router(tmp_path, monkeypatch)
    _seed_profile(cache_root, "owner_name", "Owner resume text")
    _seed_profile(cache_root, "shane_itwarie", "Mechanical engineering")

    def fail(*args, **kwargs):
        raise AssertionError("must not rank another member's profile for a non-owner")

    monkeypatch.setattr(job_match, "best_jobs", fail)
    guild = _FakeGuild(owner_id=7, members={7: "owner_name", 9: "shane_itwarie"})
    assert _dispatch(router, _GuildMessage(
        CMD_BEST_JOBS + " <@9>", channel, author_id=9, guild=guild
    ))
    assert "Only the server owner" in channel.sent[-1]


def test_bestjobs_says_what_to_do_when_no_profile_has_content(tmp_path, monkeypatch):
    router, channel, cache_root = _cache_router(tmp_path, monkeypatch)
    _seed_profile(cache_root, "example")
    _seed_profile(cache_root, "owner_name")

    def fail(*args, **kwargs):
        raise AssertionError("must not rank with no seeded profile")

    monkeypatch.setattr(job_match, "best_jobs", fail)
    guild = _FakeGuild(owner_id=7, members={7: "owner_name"})
    assert _dispatch(router, _GuildMessage(CMD_BEST_JOBS, channel, author_id=7, guild=guild))

    message = channel.sent[-1]
    assert ".resumebuild" in message
    assert "No seeded resume profile" in message


def test_bestjobs_in_a_dm_ranks_the_authors_own_profile(tmp_path, monkeypatch):
    router, channel, cache_root = _cache_router(tmp_path, monkeypatch)
    _seed_profile(cache_root, "tester", "DM author resume text")
    _seed_profile(cache_root, "other", "Someone else")
    capture: dict = {}
    _stub_best_jobs(monkeypatch, capture)

    # _FakeMessage with owner_id=None has no guild -- a DM.
    _dispatch(router, _FakeMessage(CMD_BEST_JOBS, channel, author_id=7, owner_id=None))

    assert capture["profile_key"] == "tester"


# ---------------------------------------------------------------------------
# Entry pool: the profile as a set of selectable parts
# ---------------------------------------------------------------------------

def _profile(tmp_path: Path, name: str, text: str) -> job_match.ProfileSignal:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "baseinfo.txt").write_text(text, encoding="utf-8")
    return job_match.build_profile_signal(directory, today=date(2026, 8, 25))


def test_entries_are_parsed_with_their_own_anchors(signal):
    """Each entry carries only the tools it actually evidences -- the whole
    point of the entry path is that the ML entry cannot claim the rocket
    society's STM32 work."""
    by_id = {entry.entry_id: entry for entry in signal.entries}
    assert set(by_id) == {
        "machine-learning-intern",
        "electrical-team-member",
        "technical-development-intern",
        "mobile-robot-system",
    }

    ml = by_id["machine-learning-intern"]
    assert "kafka" in ml.anchors and "python" in ml.anchors
    assert "stm32" not in ml.anchors

    electrical = by_id["electrical-team-member"]
    assert "stm32" in electrical.anchors
    assert "kafka" not in electrical.anchors


def test_generic_tag_is_dropped_from_an_entrys_role_terms(signal):
    entry = next(e for e in signal.entries if e.entry_id == "technical-development-intern")
    assert entry.categories == ()
    assert "general" not in entry.role_terms
    assert "technical development intern" in entry.role_terms


def test_every_entry_and_the_toolset_survive_into_the_judge_document(tmp_path: Path):
    """Regression for a real defect: the document was the raw file cut at a
    fixed character count, so on a profile longer than the budget the trailing
    entries and the whole SKILL ANCHORS section never reached the judge."""
    # Scaled off the constant so the fixture cannot drift out of being
    # oversized when the budget is retuned.
    filler = "Scope: " + ("detail " * (job_match.PROFILE_DOC_CHARS // 30))
    blocks = "\n\n".join(
        f"[tag{index}] Project Number {index} (C++)\n{filler}" for index in range(12)
    )
    long_profile = (
        "- Name: Test Candidate\n\n== PROJECTS ==\n\n"
        + blocks
        + "\n\n== SKILL ANCHORS ==\nLanguages: Python, C++, VHDL\n"
    )
    assert len(long_profile) > job_match.PROFILE_DOC_CHARS * 2

    signal = _profile(tmp_path, "long", long_profile)
    document = signal.document_text

    assert len(signal.entries) == 12
    # The last entry is the one a flat truncation always loses.
    assert "project-number-11" in document
    for entry in signal.entries:
        assert entry.entry_id in document
    assert "VHDL" in document          # the anchors section reached the model
    assert len(document) <= job_match.PROFILE_DOC_CHARS


def test_a_prose_only_profile_has_no_entries_and_keeps_the_flat_document(tmp_path: Path):
    prose = _profile(
        tmp_path, "prose", "Third-year student who has written embedded firmware.\n"
    )
    assert prose.entries == ()
    assert prose.document_text.startswith("Third-year student")


# ---------------------------------------------------------------------------
# Best-subset scoring
# ---------------------------------------------------------------------------

def test_an_unrelated_extra_entry_does_not_dilute_a_matching_job(tmp_path: Path):
    """The core of the entry model: a resume leaves off what does not fit, so
    owning an irrelevant project must not lower a job's score."""
    base = BASEINFO
    padded = BASEINFO.replace(
        "== SKILL ANCHORS ==",
        "[oop] Bookstore App (Java, JavaFX)\nScope: a desktop bookstore GUI.\n\n"
        "== SKILL ANCHORS ==",
    )
    job = _job("Embedded Firmware Intern (STM32, C)")

    before = job_match.score_job(job, _profile(tmp_path, "base", base))
    after = job_match.score_job(job, _profile(tmp_path, "padded", padded))

    assert after.total == pytest.approx(before.total)


def test_depth_separates_two_profiles_with_the_same_skill_union(tmp_path: Path):
    """A union of anchors cannot tell one embedded project from four. Depth
    can, and that is what decides whether a page can actually be filled."""
    header = "- Program: BEng (TMU, 2023-2027)\n\n== PROJECTS ==\n\n"
    anchors = "\n== SKILL ANCHORS ==\nLanguages: C, VHDL\nTools/Platforms: STM32\n"

    shallow = header + (
        "[embedded] Flight Controller (C)\nScope: STM32 firmware in C.\n\n"
        "[oop] Bookstore App\nScope: a desktop GUI.\n\n"
        "[web] Recipe Site\nScope: a static site.\n\n"
        "[art] Poster Series\nScope: printed posters.\n"
    ) + anchors
    deep = header + (
        "[embedded] Flight Controller (C)\nScope: STM32 firmware in C.\n\n"
        "[embedded] Motor Driver (C)\nScope: STM32 motor control in C.\n\n"
        "[embedded] Sensor Hub (C)\nScope: STM32 sensor firmware in C.\n\n"
        "[embedded] Bootloader (C)\nScope: STM32 bootloader in C.\n"
    ) + anchors

    job = _job("Embedded Firmware Intern (STM32, C)")
    shallow_score = job_match.score_job(job, _profile(tmp_path, "shallow", shallow))
    deep_score = job_match.score_job(job, _profile(tmp_path, "deep", deep))

    # Same best entry, so top1 ties -- depth is the whole difference.
    assert shallow_score.components["top1"] == pytest.approx(deep_score.components["top1"])
    assert deep_score.components["depth"] > shallow_score.components["depth"]
    assert deep_score.total > shallow_score.total


def test_entry_scores_name_the_entries_that_would_carry_the_posting(signal):
    score = job_match.score_job(_job("Embedded Firmware Intern (STM32)"), signal)
    fit_ids = [entry_id for entry_id, _ in score.entry_fits]

    assert "mobile-robot-system" in fit_ids or "electrical-team-member" in fit_ids
    assert all(0.0 < fit <= 1.0 for _, fit in score.entry_fits)
    assert len(score.entry_fits) <= job_match.DEPTH_ENTRIES


def test_entry_path_still_ranks_a_fitting_internship_above_an_unrelated_role(signal):
    good = job_match.score_job(_job("Machine Learning Intern (Python, Kafka)"), signal)
    bad = job_match.score_job(_job("Senior Physical Therapist", location="Austin, TX"), signal)

    assert set(good.components) == {"top1", "depth", "level", "location"}
    assert good.total > bad.total
    assert 0.0 <= bad.total <= 1.0 and 0.0 <= good.total <= 1.0


def test_the_prefilter_records_the_entries_that_carry_a_posting(signal):
    """Kept on the score for callers and for .resumebuild to act on; the
    channel report deliberately prints only the listing and the percentage."""
    report = job_match.rank_jobs(signal, [_job("Embedded Firmware Intern (STM32)")], limit=1)

    assert report.matches[0].score.entry_fits
    assert "use: " not in job_match.format_match_report(report, "day")


# ---------------------------------------------------------------------------
# Prefilter short-circuits
# ---------------------------------------------------------------------------

def test_the_substring_guard_agrees_with_the_boundary_regex():
    """`_names` puts a plain `in` test in front of the boundary regex to skip
    work whose answer is already known. It must therefore never disagree with
    the regex it guards -- including on the anchors whose punctuation is the
    reason that regex is hand-built in the first place."""
    anchors = ["c", "c++", "r", "node.js", "go", "sql", "power bi", "stm32"]
    texts = [
        "senior c++ developer",
        "c developer",
        "concurrency engineer",          # contains "c" but not as a word
        "node.js backend intern",
        "nodejs backend intern",         # no dot: must not match "node.js"
        "golang engineer",               # contains "go" but not as a word
        "postgresql administrator",      # contains "sql" but not as a word
        "power bi analyst",
        "stm32 firmware",
        "",
    ]
    for text in texts:
        for anchor in anchors:
            assert job_match._names(text, anchor) is (
                job_match._compiled_anchor(anchor).search(text) is not None
            ), f"{anchor!r} vs {text!r}"


def test_cached_term_words_match_a_fresh_split():
    """The role-term word sets are cached across every posting in a window, so
    a stale or wrong entry would silently mis-score the whole run."""
    for term in ("machine learning intern", "embedded", "full stack developer", "ml"):
        assert job_match._term_words(term) == frozenset(
            job_match._lower_words(term)
        ) - job_match._ROLE_STOPWORDS
        assert job_match._term_is_phrase(term) is (len(job_match._lower_words(term)) > 1)


def test_scoring_a_window_twice_is_stable(signal):
    """Every cache in the prefilter is keyed on profile constants, so a second
    pass over the same postings must reproduce the first exactly."""
    jobs = [
        _job("Embedded Firmware Intern (STM32, C)"),
        _job("Machine Learning Intern (Python)"),
        _job("Senior Physical Therapist", location="Austin, TX"),
        _job("Full Stack Developer"),
    ]
    first = [job_match.score_job(job, signal) for job in jobs]
    second = [job_match.score_job(job, signal) for job in jobs]

    for before, after in zip(first, second):
        assert before.total == after.total
        assert before.components == after.components
        assert before.entry_fits == after.entry_fits


# ---------------------------------------------------------------------------
# Dedup for records whose company is folded into the title
# ---------------------------------------------------------------------------

def _records(*titles):
    return [
        {"title": title, "link": f"https://example.com/{index}", "site_label": "Glassdoor",
         "date_posted": "2026-08-24"}
        for index, title in enumerate(titles)
    ]


def test_one_job_under_two_city_labels_is_returned_once(monkeypatch):
    """18% of archived records carry no company field -- the send path folds it
    into the title as "Role (Company, City)". Two city labels for one job used
    to match neither dedup rule and so spent two shortlist slots and two
    description fetches."""
    monkeypatch.setattr(
        job_match, "load_window_jobs", job_match.load_window_jobs
    )
    from services.jba import merge_data

    rows = _records(
        "Co-op Electrical Engineering Intern (G&A ROBOT, Burnaby)",
        "Co-op Electrical Engineering Intern (G&A ROBOT, Canada)",
    )
    monkeypatch.setattr(merge_data, "load_daily_log", lambda date_key: rows)

    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 24))
    assert len(jobs) == 1


def test_the_same_title_at_two_companies_is_not_collapsed(monkeypatch):
    """The dangerous half of that fix. "Software Engineer Intern" is one of the
    most common titles there is, so the company must stay in the identity --
    dropping the whole parenthetical would merge unrelated employers."""
    from services.jba import merge_data

    rows = _records(
        "Software Engineer Intern (Shopify, Toronto)",
        "Software Engineer Intern (Google, Toronto)",
    )
    monkeypatch.setattr(merge_data, "load_daily_log", lambda date_key: rows)

    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 24))
    assert len(jobs) == 2


def test_a_posting_with_no_company_gets_no_title_identity():
    """A bare title is too weak to collapse unrelated postings on, so a record
    that names no company is deduped by URL alone. "Finance Intern" would
    otherwise merge Acceldata with Laurentian Bank."""
    bare = _job("Embedded Intern", company="")
    urls, title, company = job_match._job_identity(bare)
    assert urls and (title, company) == ("", "")

    # A company folded into the title still counts as naming one.
    folded = _job("Embedded Intern (Acme, Toronto)", company="")
    _, title, company = job_match._job_identity(folded)
    assert (title, company) == ("embedded intern", "acme")

    # A parenthetical with no comma is a role note, not a company.
    _, title, company = job_match._job_identity(_job("Embedded Intern (Co-op)", company=""))
    assert (title, company) == ("", "")


def test_the_same_job_on_two_boards_is_returned_once(monkeypatch):
    """The case this dedup exists for. One posting syndicated to Indeed and
    Glassdoor gets a different URL on each, so only title+company can catch it
    -- and each board decorates the company its own way, so exact string
    equality never did."""
    from services.jba import merge_data

    rows = _records(
        "26-OTT05 - Mechanical Intern (Jp2g Consultants Inc., Ottawa, ON, CA)",
        "26-OTT05 - Mechanical Intern (Jp2g Consultants, Ottawa)",
    )
    rows[0]["site_label"], rows[1]["site_label"] = "Indeed", "Glassdoor"
    monkeypatch.setattr(merge_data, "load_daily_log", lambda date_key: rows)

    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 24))
    assert len(jobs) == 1


def test_board_specific_company_decoration_does_not_defeat_the_key(monkeypatch):
    """Real pairs from the archive: a corporate-structure word on one board and
    a country qualifier on the other."""
    from services.jba import merge_data

    rows = _records(
        "Motor Sports Product Development Intern (Soucy, Drummondville, QC)",
        "Motor Sports Product Development Intern (Soucy Group, Drummondville, QC, CA)",
        "Summer Intern - Toronto (PwC, Toronto, ON, CA)",
        "Summer Intern - Toronto (PwC Canada, Greater Toronto Area, Canada)",
    )
    monkeypatch.setattr(merge_data, "load_daily_log", lambda date_key: rows)

    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 24))
    assert len(jobs) == 2


def test_the_company_field_is_normalized_too(monkeypatch):
    """The ATS shape carries a real company field; it must reach the same key
    namespace as the send path's folded-in company."""
    _patch_archive(monkeypatch, {
        "2026-08-25": [
            {"title": "Embedded Intern", "company": "Acme Robotics Inc.",
             "location": "Toronto", "link": "https://example.com/a"},
            {"title": "Embedded  Intern", "company": "ACME ROBOTICS",
             "location": "Ottawa", "link": "https://example.com/b"},
        ],
    })
    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 25))
    assert len(jobs) == 1


def test_two_subsidiaries_sharing_a_first_word_are_not_merged(monkeypatch):
    """Real pair from the archive, and the reason companies are matched by
    prefix rather than by any shared word. "Hitachi Energy" and "Hitachi Rail"
    are different employers; neither leads the other, so both survive."""
    _patch_archive(monkeypatch, {
        "2026-08-25": [
            {"title": "Electro Mechanical Engineering Intern", "company": "Hitachi Energy",
             "location": "Toronto", "link": "https://example.com/a"},
            {"title": "Electro Mechanical Engineering Intern", "company": "Hitachi Rail",
             "location": "Toronto", "link": "https://example.com/b"},
        ],
    })
    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 25))
    assert len(jobs) == 2


def test_a_board_that_abbreviates_the_company_still_matches(monkeypatch):
    """The gap this closes. Suffix-stripping alone left seven duplicates in a
    week: one board writes the employer in full, another abbreviates it."""
    from services.jba import merge_data

    rows = [
        {"title": "Software Verification Engineer (Co-op/Intern) "
                  "(Lumentum Operations, Ottawa, ON, CA)",
         "site_label": "Indeed", "link": "https://ca.indeed.com/viewjob?jk=x"},
        {"title": "Software Verification Engineer (Co-op/Intern) "
                  "(Lumentum, Ottawa, Ontario, Canada)",
         "site_label": "LinkedIn", "link": "https://www.linkedin.com/jobs/view/2"},
    ]
    monkeypatch.setattr(merge_data, "load_daily_log", lambda date_key: rows)

    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 24))
    assert len(jobs) == 1


def test_a_bare_company_name_cannot_absorb_two_different_employers():
    """The ordering hazard. Every extension matches a stump, so a bare
    "Hitachi" arriving before "Hitachi Energy" and "Hitachi Rail" would absorb
    both and collapse two employers on scrape order alone. The entry narrows to
    the first full name it meets, so the second is compared against that and
    survives. Asserted in both orders, because only one of them was ever
    exercised by the archive."""
    for order in (
        ["hitachi", "hitachi energy", "hitachi rail"],
        ["hitachi energy", "hitachi rail", "hitachi"],
    ):
        seen: dict = {}
        kept = [
            company for company in order
            if not job_match._seen_before(seen, "intern", company, None)
        ]
        assert len(kept) == 2, f"{order} kept {kept}"
        assert {c for c, _ in seen["intern"]} == {"hitachi energy", "hitachi rail"}


def test_a_nameless_company_never_matches():
    """An empty name is a prefix of everything. `_job_identity` withholds a
    title identity in that case, but the rule must not depend on that: one
    nameless posting would otherwise absorb every job sharing its title."""
    assert not job_match._company_matches("acme", "")
    assert not job_match._company_matches("", "acme")
    # And the identity itself declines to offer a title to match on.
    bare = _job("Embedded Intern", company="")
    assert job_match._job_identity(bare)[1:] == ("", "")


def test_company_matching_is_a_prefix_not_a_shared_word():
    """The rule in isolation, both directions and both failure modes."""
    assert job_match._company_matches("lumentum", "lumentum operations")
    assert job_match._company_matches("lumentum operations", "lumentum")
    assert job_match._company_matches("bmo", "bmo financial")
    assert not job_match._company_matches("hitachi energy", "hitachi rail")
    assert not job_match._company_matches("shopify", "google")
    # A trailing word must not let an employer match a different one whose
    # name merely ends the same way.
    assert not job_match._company_matches("energy hitachi", "hitachi")


def test_a_disambiguating_parenthetical_is_not_stripped_from_a_title(monkeypatch):
    """The dangerous half of normalizing titles. On an ATS record the trailing
    parenthetical is what tells two real openings apart, so it stays in the
    key -- these are two distinct postings at one studio, not one job twice."""
    _patch_archive(monkeypatch, {
        "2026-08-25": [
            {"title": "Senior Combat Designer (Encounters)", "company": "2K",
             "location": "Montreal", "link": "https://example.com/a"},
            {"title": "Senior Combat Designer (AI archetype)", "company": "2K",
             "location": "Montreal", "link": "https://example.com/b"},
        ],
    })
    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 25))
    assert len(jobs) == 2


def test_a_listing_names_the_company_once_and_keeps_the_location():
    """The send path folds "Role (Company, City, Region, Country)" into the
    title and leaves both the company and location fields empty. The fold is
    split at the first comma: the company leads the line, the rest stays on as
    the location, and neither is printed twice."""
    job = job_match.ArchivedJob(
        title="Enterprise Strategy Consultant Intern (IBM, Ottawa, Ontario, Canada)",
        company="", location="", link="https://example.com/a",
        site_label="LinkedIn", date_posted="2026-08-24",
    )
    report = job_match.MatchReport(
        matches=[job_match.JobMatch(job=job, score=job_match.JobScore(total=0.9))],
        scanned=1, dates=["2026-08-24"], profile_key="p",
    )
    text = job_match.format_match_report(report, "day")

    assert (
        "[LinkedIn/IBM] Enterprise Strategy Consultant Intern "
        "(Ottawa, Ontario, Canada)" in text
    )
    # Named once: the company does not reappear inside the location.
    assert text.count("IBM") == 1


def test_a_role_parenthetical_survives_display():
    """An ATS record has a real company field, and its trailing parenthetical
    is part of the role -- dropping it would rename the job."""
    assert job_match._display_fields(
        _job("Outreach Coordinator (Cantonese/Mandarin)", company="ahschc")
    ) == ("Outreach Coordinator (Cantonese/Mandarin)", "ahschc", "Toronto, ON")


def test_a_parenthetical_without_a_comma_is_not_read_as_a_company():
    """The second guard. With no company field and no comma, "(Co-op/Intern)"
    is a role note, not "Role at company Co-op/Intern"."""
    assert job_match._display_fields(
        _job("Software Verification Engineer (Co-op/Intern)", company="", location="")
    ) == ("Software Verification Engineer (Co-op/Intern)", "", "")


def test_a_placeholder_site_label_is_not_printed():
    """`_normalize_record` defaults an absent site to "unknown"; showing
    "(unknown/Acme)" would be worse than showing the company alone."""
    text = job_match.format_match_report(
        job_match.MatchReport(
            matches=[job_match.JobMatch(
                job=_job("Embedded Intern", site_label="unknown"),
                score=job_match.JobScore(total=0.5),
            )],
            scanned=1, dates=["2026-08-24"], profile_key="p",
        ),
        "day",
    )
    assert text.count("[Acme] Embedded Intern") == 1 and "unknown" not in text


def test_a_nested_company_parenthetical_is_split_whole():
    """Real archive title. The old regex used [^()]* and matched nothing here,
    so the posting lost both its dedup identity and its company."""
    assert job_match._split_trailing_paren(
        "Actuarial intern (iA Financial Group (Industrial Alliance), Quebec, Canada)"
    ) == ("Actuarial intern", "iA Financial Group (Industrial Alliance), Quebec, Canada")
    assert job_match._split_trailing_paren("No parenthetical") == ("", "")
    assert job_match._split_trailing_paren("Unbalanced (oops") == ("", "")


def test_one_posting_on_an_ats_and_on_an_aggregator_collapses(monkeypatch):
    """The cross-family case: the ATS scraper and jobspy reach the same job by
    different routes and mint different URLs, so only title+company can catch
    it. The two record shapes differ in every surface detail -- the ATS record
    has a real company field and a bare title, the aggregator folds "(Company,
    City, Region, Country)" into the title and leaves the field empty -- and
    both still have to land on one key."""
    from services.jba import merge_data

    rows = [
        {"title": "Software Engineer Intern", "company": "shopify",
         "location": "Toronto", "_source_site": "greenhouse",
         "job_url": "https://job-boards.greenhouse.io/shopify/jobs/1"},
        {"title": "Software Engineer Intern (Shopify Inc., Toronto, ON, CA)",
         "site_label": "Indeed", "link": "https://ca.indeed.com/viewjob?jk=x"},
    ]
    monkeypatch.setattr(merge_data, "load_daily_log", lambda date_key: rows)

    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 24))
    assert len(jobs) == 1


def test_a_generic_title_shared_across_families_is_not_collapsed(monkeypatch):
    """The counterweight, and a real pair from the archive: "Finance Intern" at
    Acceldata in Bengaluru and at Laurentian Bank in Montreal are two jobs. The
    company is what keeps them apart, which is why title alone is never a key."""
    from services.jba import merge_data

    rows = [
        {"title": "Finance Intern", "company": "acceldata", "location": "Bengaluru",
         "_source_site": "lever", "job_url": "https://jobs.lever.co/acceldata/1"},
        {"title": "Finance Intern (Laurentian Bank, Montreal, Quebec, Canada)",
         "site_label": "LinkedIn", "link": "https://www.linkedin.com/jobs/view/2"},
    ]
    monkeypatch.setattr(merge_data, "load_daily_log", lambda date_key: rows)

    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 24))
    assert len(jobs) == 2


def test_an_abbreviated_ats_slug_is_a_known_cross_family_gap(monkeypatch):
    """Documented limitation, asserted so it cannot change unnoticed.

    An ATS record's company is the board slug. When the slug spells the company
    ("shopify", "acceldata") it normalizes onto the aggregator's display name
    and the duplicate collapses. When it is an abbreviation -- "ahschc" for
    Asian Health Services -- nothing connects the two strings, and the posting
    survives twice. Closing it needs a slug-to-name mapping the repo does not
    have: data/ats_companies/*.json are bare slug lists, and no archived record
    carries a job_url_direct pointing back at the ATS.

    Guessing across that gap is the worse failure. Collapsing on title alone
    would merge every "Clinic Assistant" in the archive."""
    from services.jba import merge_data

    rows = [
        {"title": "Clinic Assistant", "company": "ahschc", "location": "San Leandro",
         "_source_site": "lever", "job_url": "https://jobs.lever.co/ahschc/1"},
        {"title": "Clinic Assistant (Asian Health Services, San Leandro, CA)",
         "site_label": "Indeed", "link": "https://ca.indeed.com/viewjob?jk=y"},
    ]
    monkeypatch.setattr(merge_data, "load_daily_log", lambda date_key: rows)

    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 24))
    assert len(jobs) == 2  # the gap; change this only alongside a real fix


def test_one_role_open_in_two_countries_is_two_postings(monkeypatch):
    """Real pair from the archive: 2K lists "Senior Combat Designer" in
    Montreal and in Novato. Same title, same company, two jobs you apply to
    separately -- title+company alone wrongly merged them."""
    from services.jba import merge_data

    rows = [
        {"title": "Senior Level Artist", "company": "2k", "_source_site": "greenhouse",
         "location": "Montreal, Quebec, Canada", "job_url": "https://job-boards.greenhouse.io/2k/jobs/1"},
        {"title": "Senior Level Artist", "company": "2k", "_source_site": "greenhouse",
         "location": "Novato, California, United States", "job_url": "https://job-boards.greenhouse.io/2k/jobs/2"},
    ]
    monkeypatch.setattr(merge_data, "load_daily_log", lambda date_key: rows)

    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 24))
    assert len(jobs) == 2


def test_an_unresolved_country_still_merges_across_boards(monkeypatch):
    """The rule that makes country safe to add. Boards disagree on how much
    location they print for one posting -- "Ottawa, ON, CA" on Indeed against a
    bare "Ottawa" on Glassdoor -- and only 62% of archived records resolve to a
    country at all. Treating unknown as its own value would split these back
    apart, undoing the cross-board dedup."""
    from services.jba import merge_data

    rows = [
        {"title": "26-OTT05 - Mechanical Intern (Jp2g Consultants Inc., Ottawa, ON, CA)",
         "site_label": "Indeed", "link": "https://ca.indeed.com/viewjob?jk=x"},
        {"title": "26-OTT05 - Mechanical Intern (Jp2g Consultants, Ottawa)",
         "site_label": "Glassdoor", "link": "https://glassdoor.ca/job-listing/1"},
    ]
    monkeypatch.setattr(merge_data, "load_daily_log", lambda date_key: rows)

    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 24))
    assert len(jobs) == 1


def test_a_province_code_is_not_read_as_a_country():
    """The trap that rules out splitting the location on its last comma. "CA"
    is the commonest tail in the archive and means two different countries."""
    assert job_match._country_from_location("Toronto, ON, CA") == "CA"
    assert job_match._country_from_location("San Leandro, CA") == "US"
    # Two spellings of one country must agree, or cross-board pairs split.
    assert (
        job_match._country_from_location("Greater Toronto Area, Canada")
        == job_match._country_from_location("Toronto, ON, CA")
    )
    # Nothing resolvable is None, never a guess.
    assert job_match._country_from_location("3 Locations") is None
    assert job_match._country_from_location("") is None


def test_the_country_of_a_folded_title_comes_from_the_fold():
    """A send-path record has an empty location field; the country is only
    available inside the title's "(Company, City, Region, Country)" suffix."""
    job = job_match.ArchivedJob(
        title="Analyst (Deloitte, Calgary, Alberta, Canada)", company="", location="",
        link="https://example.com/a", site_label="LinkedIn", date_posted="2026-08-24",
    )
    assert job_match._job_country(job) == "CA"


def test_the_same_url_stays_one_job_whatever_the_location_says(monkeypatch):
    """Country splits title+company groups only. A shared URL is the same
    posting by definition, so it must not be split by a location edit."""
    from services.jba import merge_data

    rows = [
        {"title": "Embedded Intern", "company": "acme", "_source_site": "lever",
         "location": "Toronto, ON, CA", "job_url": "https://jobs.lever.co/acme/1"},
        {"title": "Embedded Intern", "company": "acme", "_source_site": "lever",
         "location": "Austin, TX, US", "job_url": "https://jobs.lever.co/acme/1"},
    ]
    monkeypatch.setattr(merge_data, "load_daily_log", lambda date_key: rows)

    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 24))
    assert len(jobs) == 1


def test_a_bare_city_is_placed_by_the_geo_table():
    """The string parser needs a country or admin token, so it gives up on a
    location that names only a city. The geo table places those by population,
    which is what lifts country coverage from 62% to 77% of the archive."""
    assert job_match._country_from_location("Basingstoke") == "GB"
    assert job_match._country_from_location("Dublin") == "IE"
    assert job_match._country_from_location("New York City") == "US"


def test_an_ambiguous_city_name_resolves_to_nothing():
    """A wrong country is worse than an unknown one: unknown still merges
    duplicates, wrong splits them apart. "Springfield" has no dominant
    claimant, so it stays unresolved rather than being guessed."""
    from services.jba.geo_db import country_for_city

    assert country_for_city("springfield") is None
    assert country_for_city("nowherecity") is None
    assert country_for_city("") is None


def test_a_bare_city_is_placed_by_the_geo_table():
    """The string parser needs a country or admin token, so it gives up on a
    location that names only a city. The geo table places those by population,
    which lifts country coverage from 62% to 77% of the archive."""
    assert job_match._country_from_location("Basingstoke") == "GB"
    assert job_match._country_from_location("Dublin") == "IE"
    assert job_match._country_from_location("New York City") == "US"


def test_an_ambiguous_city_name_resolves_to_nothing():
    """A wrong country is worse than an unknown one: unknown still merges
    duplicates, wrong splits them apart. "Springfield" has no dominant
    claimant, so it stays unresolved rather than being guessed."""
    from services.jba.geo_db import country_for_city

    assert country_for_city("springfield") is None
    assert country_for_city("nowherecity") is None
    assert country_for_city("") is None


def test_two_cities_in_one_country_are_one_listing_and_one_candidate(monkeypatch):
    """The behaviour the command is for. A board mints a separate listing per
    city, so one job arrives as several records that differ only in location.
    Within a country they are one posting: one row, one shortlist slot, and
    one description fetch -- not four of each."""
    from services.jba import merge_data

    rows = [
        {"title": "Co-op Electrical Intern", "company": "gnarobot", "_source_site": "lever",
         "location": city, "job_url": f"https://jobs.lever.co/gnarobot/{index}"}
        for index, city in enumerate(
            ["Burnaby, BC, CA", "Vancouver, BC, CA", "Toronto, ON, CA", "Canada"]
        )
    ]
    monkeypatch.setattr(merge_data, "load_daily_log", lambda date_key: rows)

    jobs, _ = job_match.load_window_jobs(1, end_date=date(2026, 8, 24))
    assert len(jobs) == 1

    # And the saving that matters: the judge is shown one candidate, and only
    # one posting is worth fetching a description for.
    fetched: list[str] = []
    monkeypatch.setattr(
        job_match, "enrich_descriptions",
        lambda candidates, max_fetches: fetched.extend(j.title for j in candidates) or 0,
    )
    judged: list[int] = []

    def fake_judge(signal, candidates, settings, client_factory):
        judged.append(len(candidates))
        return {}, "ok:stub"

    monkeypatch.setattr(job_match, "judge_jobs", fake_judge)
    report = job_match.rank_jobs(
        job_match.ProfileSignal(profile_key="p", document_text="embedded intern"),
        jobs, settings=object(), limit=10,
    )
    assert judged == [1]          # one batched judge call, over one candidate
    assert len(fetched) == 1      # one description fetch, not four
    assert report.scanned == 1


def test_a_company_that_is_only_a_suffix_word_keeps_its_name():
    """Stripping must never empty a company: an empty company is no identity,
    and would collapse every posting that shares a title."""
    assert job_match._normalize_company_name("Group") == "group"
    assert job_match._normalize_company_name("Co.") == "co"
    assert job_match._normalize_company_name("Soucy Group Inc.") == "soucy"
    assert job_match._normalize_company_name("PwC Canada") == "pwc"


# ---------------------------------------------------------------------------
# Profile document budget allocation
# ---------------------------------------------------------------------------

def test_a_short_entry_does_not_waste_the_allowance_a_long_one_needs():
    """A flat per-entry cap spent the same on every entry, so short ones left
    their remainder unused while the largest project kept 320 of its 1,062
    characters. The surplus has to flow to whoever is still truncated."""
    lengths = [100, 100, 100, 1000]
    allowance = job_match._allocate_entry_budget(lengths, 1300)

    assert allowance[:3] == [100, 100, 100]   # short entries complete
    assert allowance[3] == 1000               # and the long one fits too
    assert sum(allowance) <= 1300


def test_a_budget_too_small_for_everything_still_reaches_every_entry():
    """Degrading has to stay graceful: an entry the model never sees cannot be
    selected, so no entry may be dropped to zero while others run long."""
    lengths = [500, 500, 500, 500]
    allowance = job_match._allocate_entry_budget(lengths, 400)

    assert all(size > 0 for size in allowance)
    assert sum(allowance) <= 400


def test_allocation_never_exceeds_what_an_entry_actually_has():
    allowance = job_match._allocate_entry_budget([50, 60], 10_000)
    assert allowance == [50, 60]


def test_a_trimmed_entry_ends_at_a_boundary_not_mid_word():
    """A mid-word cut reads as corrupted text to the model rather than as an
    entry that simply ends."""
    text = "Built the scheduler. Monitored five ATS providers continuously."
    trimmed = job_match._trim_at_boundary(text, 30)

    assert trimmed == "Built the scheduler."
    assert not trimmed.endswith(" ")
    assert job_match._trim_at_boundary(text, len(text) + 10) == text
    assert job_match._trim_at_boundary(text, 0) == ""


def test_a_real_profile_reaches_the_judge_without_losing_an_entry(tmp_path: Path):
    """End to end: the entry pool is the thing the whole feature selects from,
    so every entry's own text should survive the budget intact at this size."""
    signal = _profile(tmp_path, "whole", BASEINFO)
    for entry in signal.entries:
        assert entry.text in signal.document_text


# ---------------------------------------------------------------------------
# Build plans: which entries the judge would put on the page
# ---------------------------------------------------------------------------

def test_the_judge_prompt_asks_for_a_build_plan_by_entry_id(signal):
    prompt = job_match.build_judge_prompt(signal, [_job("Embedded Intern")])

    assert "POOL of resume entries" in prompt
    assert '"use"' in prompt
    # The ids it is told to answer with must be ids it was actually shown.
    for entry in signal.entries:
        assert f"[{entry.entry_id}]" in prompt


def test_a_plan_naming_real_entries_is_kept():
    ids = frozenset({"eebot-mobile-robot-system", "electrical-team-member"})
    parsed = job_match.parse_judge_response(
        '{"ranked": [{"id": 0, "score": 80, "reason": "fits",'
        ' "use": ["eebot-mobile-robot-system", "electrical-team-member"],'
        ' "missing": "no RTOS experience"}]}',
        count=1,
        entry_ids=ids,
    )
    verdict = parsed[0]

    assert verdict.entries == ("eebot-mobile-robot-system", "electrical-team-member")
    assert verdict.gap == "no RTOS experience"


def test_an_invented_entry_id_is_dropped_from_the_plan():
    """The plan's whole value is that the user can act on it, so an entry id
    the profile does not contain is worse than no plan at all."""
    parsed = job_match.parse_judge_response(
        '{"ranked": [{"id": 0, "score": 80, "use": ["kubernetes-project", "eebot"]}]}',
        count=1,
        entry_ids=frozenset({"eebot"}),
    )
    assert parsed[0].entries == ("eebot",)


def test_a_plan_is_capped_and_deduplicated():
    ids = frozenset({f"e{n}" for n in range(8)})
    parsed = job_match.parse_judge_response(
        '{"ranked": [{"id": 0, "score": 50,'
        ' "use": ["e0", "e0", "e1", "e2", "e3", "e4", "e5"]}]}',
        count=1,
        entry_ids=ids,
    )
    entries = parsed[0].entries

    assert len(entries) == job_match.MAX_PLAN_ENTRIES
    assert len(set(entries)) == len(entries)
    assert entries[0] == "e0"


def test_a_response_without_a_plan_still_parses(signal):
    """The old whole-profile shape. A validator that rejected it would send the
    entire prompt to the next provider and pay for the call a second time."""
    parsed = job_match.parse_judge_response(
        '{"ranked": [{"id": 0, "score": 70, "reason": "adjacent field"}]}',
        count=1,
        entry_ids=frozenset({"eebot"}),
    )
    verdict = parsed[0]

    assert verdict.score == 0.70
    assert verdict.entries == ()
    assert verdict.gap == ""


def test_the_judges_plan_is_kept_on_the_score(monkeypatch, signal):
    """The build plan is the answer to "which parts of my resume would I use",
    and .resumebuild is its real consumer. It is recorded whether or not the
    channel report chooses to print it."""
    def fake(prompt, settings, validate, client_factory=None, **kwargs):
        item = _prompt_items(prompt)[0]
        return validate(json.dumps({"ranked": [{
            "id": item["id"], "score": 88, "reason": "strong firmware fit",
            "use": ["electrical-team-member"], "missing": "no RTOS experience",
        }]})), "gemini-flash"

    monkeypatch.setattr(
        "services.resumes.listing.generate_validated_with_providers", fake
    )
    report = job_match.rank_jobs(
        signal, [_job("Embedded Firmware Intern (STM32)")], settings=object(),
        limit=1, enrich=False,
    )
    text = job_match.format_match_report(report, "day")

    score = report.matches[0].score
    assert score.plan_entries == ("electrical-team-member",)
    assert score.plan_gap == "no RTOS experience"
    assert job_match.plan_text(score) == (
        "use: electrical-team-member | missing: no RTOS experience"
    )
    # ... but the channel sees the listing and the score, nothing else.
    assert "electrical-team-member" not in text
    assert "88%" in text


# ---------------------------------------------------------------------------
# Description cache
# ---------------------------------------------------------------------------

class _HttpError(Exception):
    """Stands in for requests.HTTPError, which carries the status on .response."""

    def __init__(self, status: int):
        super().__init__(str(status))
        self.response = type("R", (), {"status_code": status})()


def test_a_second_run_reuses_the_stored_description_without_refetching(monkeypatch):
    """The point of the store: every fetch used to be thrown away when the
    command returned, so running .bestjobs twice fetched the same postings
    twice."""
    capture = {}
    _stub_scraper(monkeypatch, {"https://example.com/a": "STM32 firmware work."},
                  capture=capture)

    first = [_job("Embedded Intern", link="https://example.com/a")]
    assert job_match.enrich_descriptions(first) == 1
    assert len(capture["urls"]) == 1

    second = [_job("Embedded Intern", link="https://example.com/a")]
    assert job_match.enrich_descriptions(second) == 1
    assert len(capture["urls"]) == 1          # no second network call
    assert second[0].description == "STM32 firmware work."


def test_a_remembered_failure_is_not_retried(monkeypatch):
    """LinkedIn and Indeed block reliably. Re-attempting them every run would
    spend the scarcest budget in the command rediscovering that."""
    capture = {}
    _stub_scraper(monkeypatch, {}, capture=capture)     # every fetch fails

    job_match.enrich_descriptions([_job("A", link="https://example.com/blocked")])
    assert len(capture["urls"]) == 1

    job_match.enrich_descriptions([_job("A", link="https://example.com/blocked")])
    assert len(capture["urls"]) == 1                    # still one


def test_cache_hits_do_not_consume_the_fetch_budget(monkeypatch):
    """Coverage compounds only if a hit is free: each run must inherit what
    earlier runs paid for and spend its whole budget on new postings."""
    bodies = {f"https://example.com/{n}": f"body {n}" for n in range(6)}
    capture = {}
    _stub_scraper(monkeypatch, bodies, capture=capture)

    warm = [_job(f"J{n}", link=f"https://example.com/{n}") for n in range(3)]
    job_match.enrich_descriptions(warm, max_fetches=3)
    assert len(capture["urls"]) == 3

    everything = [_job(f"J{n}", link=f"https://example.com/{n}") for n in range(6)]
    filled = job_match.enrich_descriptions(everything, max_fetches=3)

    # 3 free from cache + 3 newly fetched, on a budget of 3 fetches.
    assert filled == 6
    assert len(capture["urls"]) == 6


def test_failures_are_classified_so_they_expire_correctly():
    """A 404 is a fact about the posting; a timeout is a fact about the moment.
    Storing them identically would either retry dead links forever or make a
    transient block permanent."""
    assert job_match._failure_status(_HttpError(404)) == description_cache.STATUS_DEAD
    assert job_match._failure_status(_HttpError(410)) == description_cache.STATUS_DEAD
    assert job_match._failure_status(_HttpError(403)) == description_cache.STATUS_BLOCKED
    assert job_match._failure_status(_HttpError(429)) == description_cache.STATUS_BLOCKED
    assert job_match._failure_status(TimeoutError()) == description_cache.STATUS_ERROR


def test_a_stale_failure_is_offered_again_but_a_success_is_not(monkeypatch):
    """Failures expire so a board that starts answering again is picked back
    up; successes do not, because a posting's text does not change."""
    from datetime import datetime, timedelta, timezone

    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    description_cache.store("https://example.com/ok", "kept", description_cache.STATUS_OK)
    description_cache.store("https://example.com/bad", "", description_cache.STATUS_BLOCKED)
    conn = description_cache._connect()
    conn.execute("UPDATE job_descriptions SET fetched_at = ?", (old,))
    conn.commit()

    found = description_cache.lookup(
        ["https://example.com/ok", "https://example.com/bad"]
    )
    assert description_cache.url_key("https://example.com/ok") in found
    assert description_cache.url_key("https://example.com/bad") not in found


def test_the_same_posting_under_two_urls_is_stored_once():
    """Keyed the way the ranker dedupes, so one job is one row."""
    description_cache.store("https://example.com/x?utm_source=slack", "body")
    found = description_cache.lookup(["https://example.com/x"])
    assert found.get(description_cache.url_key("https://example.com/x")) == (
        description_cache.STATUS_OK, "body",
    )


def test_an_unusable_cache_does_not_break_enrichment(monkeypatch):
    """This is an optimisation, and an optimisation that can take the command
    down is a bug."""
    monkeypatch.setattr(description_cache, "_connect", lambda: None)
    _stub_scraper(monkeypatch, {"https://example.com/a": "still works"})

    jobs = [_job("Embedded Intern", link="https://example.com/a")]
    assert job_match.enrich_descriptions(jobs) == 1
    assert jobs[0].description == "still works"


def test_enrichment_can_be_asked_to_ignore_the_cache(monkeypatch):
    capture = {}
    _stub_scraper(monkeypatch, {"https://example.com/a": "fresh"}, capture=capture)

    for _ in range(2):
        job_match.enrich_descriptions(
            [_job("A", link="https://example.com/a")], use_cache=False
        )
    assert len(capture["urls"]) == 2


def test_the_stored_body_is_capped_to_what_is_actually_read_back(monkeypatch):
    """A cache hit is truncated to ENRICH_DESCRIPTION_CHARS on the way out, so
    storing more than that would keep text forever that nothing ever reads.
    Missed on the first pass: the record was capped and the cache row was not."""
    huge = "requirements " * 2000
    _stub_scraper(monkeypatch, {"https://example.com/big": huge})

    job_match.enrich_descriptions([_job("A", link="https://example.com/big")])

    _, body = description_cache.lookup(["https://example.com/big"])[
        description_cache.url_key("https://example.com/big")
    ]
    assert len(body) <= job_match.ENRICH_DESCRIPTION_CHARS


def test_an_ok_row_with_no_text_is_refetched_not_dropped(monkeypatch):
    """Defensive: such a row cannot be written today, but silently losing the
    posting from both the cache path and the fetch queue would be the worst
    possible failure -- it would simply vanish from the ranking."""
    conn = description_cache._connect()
    conn.execute(
        "INSERT INTO job_descriptions (url_key, body, status, fetched_at) "
        "VALUES (?, '', ?, ?)",
        (description_cache.url_key("https://example.com/hollow"),
         description_cache.STATUS_OK, "2026-08-27T00:00:00+00:00"),
    )
    conn.commit()
    _stub_scraper(monkeypatch, {"https://example.com/hollow": "real text"})

    jobs = [_job("A", link="https://example.com/hollow")]
    assert job_match.enrich_descriptions(jobs) == 1
    assert jobs[0].description == "real text"


def test_cached_descriptions_survive_the_jobs_table_lifecycle(monkeypatch):
    """The store shares a database file with the live scrape log, whose rows
    are deleted weekly, migrated on schema bumps, and VACUUMed monthly. The
    reason a side table was chosen over a field on each job record is that it
    outlives all of that -- pinned here so a later change to merge_data cannot
    quietly start discarding weeks of accumulated network work."""
    from services.jba import merge_data

    description_cache.store("https://example.com/keep", "requirements text")
    conn = description_cache._connect()

    # Everything merge_data does destructively to this file, in order.
    merge_data._init_tables(conn)                       # CREATE IF NOT EXISTS + DROP seen_urls
    conn.execute("DELETE FROM jobs")                    # weekly rotation
    conn.commit()
    conn.execute("VACUUM")                              # monthly consolidation

    found = description_cache.lookup(["https://example.com/keep"])
    assert found[description_cache.url_key("https://example.com/keep")] == (
        description_cache.STATUS_OK, "requirements text",
    )


def test_a_missing_database_file_rebuilds_the_table_empty(tmp_path, monkeypatch):
    """If jobs.db is ever deleted, the cache must come back empty rather than
    erroring -- the descriptions are gone, which is a cost, not a fault."""
    import threading as _threading

    monkeypatch.setattr(description_cache, "_DB_PATH", tmp_path / "gone" / "jobs.db")
    monkeypatch.setattr(description_cache, "_local", _threading.local())

    assert description_cache.lookup(["https://example.com/a"]) == {}
    assert description_cache.store("https://example.com/a", "body") is True
    assert description_cache.stats() == {description_cache.STATUS_OK: 1}
