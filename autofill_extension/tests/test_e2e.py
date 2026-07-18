"""Playwright e2e harness for the autofill extension.

Loads the real unpacked extension into a persistent Chromium context (MV3
extensions require this - see https://playwright.dev/python/docs/chrome-extensions),
seeds a test profile directly into chrome.storage.local via an extension
page, then triggers a fill by calling the background service worker's
exposed __afFillActiveTab() hook directly - this exercises the real
executeScript injection + content script pipeline without needing to
simulate a toolbar-icon click (activeTab's gesture requirement isn't
otherwise automatable). Fixtures are served over http://127.0.0.1 rather
than file:// to avoid Chrome's separate "allow file access" extension
toggle, which defaults off and isn't scriptable.

Run with: python -m pytest autofill_extension/tests/test_e2e.py -v
"""

from __future__ import annotations

import http.server
import json
import pathlib
import threading
import time
from functools import partial

import pytest
from playwright.sync_api import BrowserContext, Page, Worker, sync_playwright

EXTENSION_DIR = pathlib.Path(__file__).resolve().parent.parent
FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"

TEST_PROFILE = {
    "profile_id": "test",
    "contact": {
        "first_name": "Ada",
        "last_name": "Lovelace",
        "email": "ada@example.com",
        "phone": "5551234567",
        "linkedin": "linkedin.com/in/ada",
        "address": {"country": "Canada"},
    },
    "work_authorization": {"authorized": True},
}

# For the derivation-layer tests: everything asserted from this profile is
# EXTRAPOLATED (dates -> years, degree string -> education level, booleans
# -> polarized yes/no answers), nothing literal. All date ranges are
# deliberately CLOSED so the derived years figure never drifts as real
# time passes: 2022-05..2022-09 (4mo) + 2023-05..2025-11 (2.5y) = 2.83y,
# floored to "2" (floor, not round - overclaiming experience is worse
# than underclaiming).
PHASE3_PROFILE = {
    "profile_id": "phase3",
    "contact": {
        "first_name": "Ada",
        "last_name": "Lovelace",
        "email": "ada@example.com",
        "phone": "5551234567",
        "linkedin": "linkedin.com/in/ada",
        "address": {"city": "Toronto", "region": "Ontario", "country": "Canada"},
    },
    "work_authorization": {"authorized": True, "requires_sponsorship": False},
    "experience": [
        {"company": "Analytical Engines Inc", "title": "Software Engineer", "start": "2023-05", "end": "2025-11"},
        {"company": "Babbage Labs", "title": "Engineering Intern", "start": "2022-05", "end": "2022-09"},
    ],
    "education": [
        {
            "school": "Toronto Metropolitan University",
            "degree": "BEng, Computer Engineering",
            "start": "2022-09",
            "end": "2027-04",
        }
    ],
    "skills": ["Python", "TypeScript"],
}


@pytest.fixture(scope="module")
def fixture_server():
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(FIXTURES_DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def extension_context(tmp_path_factory):
    user_data_dir = tmp_path_factory.mktemp("chrome-profile")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(user_data_dir),
            headless=True,
            channel="chromium",
            args=[
                f"--disable-extensions-except={EXTENSION_DIR}",
                f"--load-extension={EXTENSION_DIR}",
            ],
        )
        sw = _wait_for_service_worker(context)
        extension_id = sw.url.split("/")[2]
        yield context, extension_id, sw
        context.close()


def _wait_for_service_worker(context: BrowserContext) -> Worker:
    for _ in range(100):
        if context.service_workers:
            return context.service_workers[0]
        time.sleep(0.1)
    return context.wait_for_event("serviceworker", timeout=5000)


def _seed_profile(context: BrowserContext, extension_id: str, profile: dict) -> None:
    page = context.new_page()
    page.goto(f"chrome-extension://{extension_id}/options/options.html")
    page.evaluate(
        """(profile) => chrome.storage.local.set({
            af_profiles: { [profile.profile_id]: profile },
            af_active_profile_id: profile.profile_id,
        })""",
        profile,
    )
    page.close()


def _clear_cache(context: BrowserContext, extension_id: str) -> None:
    page = context.new_page()
    page.goto(f"chrome-extension://{extension_id}/options/options.html")
    page.evaluate("() => chrome.storage.local.remove('af_field_cache')")
    page.close()


def _trigger_fill(sw: Worker) -> dict:
    return sw.evaluate("() => self.__afFillActiveTab()")


def _seed_llm_config(context: BrowserContext, extension_id: str, config: dict) -> None:
    page = context.new_page()
    page.goto(f"chrome-extension://{extension_id}/options/options.html")
    page.evaluate("(config) => chrome.storage.local.set({ af_llm_config: config })", config)
    page.close()


def _clear_llm_config(context: BrowserContext, extension_id: str) -> None:
    page = context.new_page()
    page.goto(f"chrome-extension://{extension_id}/options/options.html")
    page.evaluate("() => chrome.storage.local.remove('af_llm_config')")
    page.close()


@pytest.fixture(autouse=True)
def _fresh_cache(extension_context):
    context, extension_id, _sw = extension_context
    _clear_cache(context, extension_id)
    _clear_llm_config(context, extension_id)
    yield


def test_plain_form(extension_context, fixture_server):
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    page.goto(f"{fixture_server}/plain_form.html")
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    assert page.locator("#firstName").input_value() == "Ada"
    assert page.locator("#lastName").input_value() == "Lovelace"
    assert page.locator("#email").input_value() == "ada@example.com"
    assert page.locator("#phone").input_value() == "5551234567"
    assert page.locator("#linkedin").input_value() == "linkedin.com/in/ada"
    assert page.locator("#country").input_value() == "ca"
    assert page.locator('input[name="work_auth"][value="yes"]').is_checked()
    assert not page.locator("#subscribe").is_checked()

    # Honeypot: the visibility gate must keep this untouched.
    assert page.locator("#website2").input_value() == ""

    page.close()


def test_react_controlled_form(extension_context, fixture_server):
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    page.goto(f"{fixture_server}/react_form.html")
    page.wait_for_selector("#firstName")
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    # A plain `.value =` write would be silently reverted by React on its
    # next render - this only passes if filler.js's native-setter path
    # actually satisfied React's internal state.
    assert page.locator("#firstName").input_value() == "Ada"
    assert page.locator("#email").input_value() == "ada@example.com"
    assert page.locator("#country").input_value() == "ca"

    page.close()


def test_shadow_dom_form(extension_context, fixture_server):
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    page.goto(f"{fixture_server}/shadow_form.html")
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    # Playwright's locator engine pierces open shadow roots by default.
    assert page.locator("#firstName").input_value() == "Ada"
    assert page.locator("#email").input_value() == "ada@example.com"

    page.close()


def test_masked_input_form(extension_context, fixture_server):
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    page.goto(f"{fixture_server}/masked_form.html")
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    digits_only = page.locator("#phone").input_value().replace("(", "").replace(")", "").replace("-", "").replace(" ", "")
    assert digits_only == "5551234567"

    page.close()


def test_popup_renders_profile_and_empty_state(extension_context):
    # Deliberately does NOT click "Fill this page" here. Opening popup.html
    # via page.goto() makes the popup itself the "active tab" (real Chrome
    # popups aren't tabs at all, so this never happens in actual usage) -
    # that would make chrome.tabs.query({active:true}) target the popup
    # instead of the real form page, which is a test-harness artifact, not
    # a product bug. The fill pipeline itself is already covered above via
    # the service worker's __afFillActiveTab() hook; this test only checks
    # that the popup UI renders the right state and stays error-free.
    context, extension_id, _sw = extension_context

    # Earlier tests in this module-scoped context already seeded a
    # profile - clear it so the empty state is genuinely empty here.
    clear_page = context.new_page()
    clear_page.goto(f"chrome-extension://{extension_id}/options/options.html")
    clear_page.evaluate("() => chrome.storage.local.remove(['af_profiles', 'af_active_profile_id'])")
    clear_page.close()

    empty_page = context.new_page()
    console_errors = []
    empty_page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
    empty_page.goto(f"chrome-extension://{extension_id}/popup/popup.html")
    empty_page.wait_for_timeout(100)
    assert empty_page.locator("#emptyState").is_visible()
    assert not empty_page.locator("#mainState").is_visible()
    empty_page.close()

    _seed_profile(context, extension_id, TEST_PROFILE)
    page = context.new_page()
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
    page.goto(f"chrome-extension://{extension_id}/popup/popup.html")
    page.wait_for_timeout(100)
    assert page.locator("#mainState").is_visible()
    assert not page.locator("#emptyState").is_visible()
    assert page.locator("#profileSelect").input_value() == "test"
    page.close()

    assert console_errors == []


def test_cache_short_circuits_second_fill(extension_context, fixture_server):
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    page.goto(f"{fixture_server}/plain_form.html")
    page.bring_to_front()

    first = _trigger_fill(sw)
    assert first["ok"]

    page.fill("#firstName", "")
    second = _trigger_fill(sw)
    assert second["ok"]
    assert page.locator("#firstName").input_value() == "Ada"
    matching = [f for f in second["summary"]["fields"] if f["label"] == "First Name"]
    assert matching and matching[0]["source"] == "cache"

    page.close()


def _mock_gemini_success(context: BrowserContext, answers: dict):
    import json

    def handler(route):
        text = json.dumps(answers)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}),
        )

    context.route("**generativelanguage.googleapis.com**", handler)
    return handler


def test_llm_tier_fills_what_heuristics_cannot(extension_context, fixture_server):
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)
    _seed_llm_config(context, extension_id, {"gemini_api_key": "fake-key-for-test"})
    _mock_gemini_success(context, {"f0": "3", "f1": "LinkedIn"})

    page = context.new_page()
    page.goto(f"{fixture_server}/llm_fixture.html")
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    # Tier 1 (no LLM needed) still resolves the ordinary field.
    assert page.locator("#firstName").input_value() == "Ada"
    # Tier 3 fills what nothing else could: a free-text prompt and a
    # dropdown whose option text the LLM was told to match verbatim.
    assert page.locator("#pyYears").input_value() == "3"
    assert page.locator("#howHeard").input_value() == "linkedin"

    llm_fields = [f for f in result["summary"]["fields"] if f["source"] and f["source"].startswith("llm:")]
    assert len(llm_fields) == 2
    assert all(f["source"] == "llm:gemini" for f in llm_fields)

    context.unroute("**generativelanguage.googleapis.com**")
    page.close()


def test_llm_falls_back_to_next_provider_on_failure(extension_context, fixture_server):
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)
    _seed_llm_config(
        context,
        extension_id,
        {"gemini_api_key": "fake-gemini-key", "openrouter_api_key": "fake-openrouter-key"},
    )

    attempted = []

    def gemini_fails(route):
        attempted.append("gemini")
        route.fulfill(status=500, body="internal error")

    def openrouter_succeeds(route):
        attempted.append("openrouter")
        import json

        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"choices": [{"message": {"content": '{"f0": "3", "f1": "Referral"}'}}]}),
        )

    context.route("**generativelanguage.googleapis.com**", gemini_fails)
    context.route("**openrouter.ai**", openrouter_succeeds)

    page = context.new_page()
    page.goto(f"{fixture_server}/llm_fixture.html")
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    # Both gemini tiers (same key, two models) are tried before openrouter.
    assert attempted == ["gemini", "gemini", "openrouter"]
    assert page.locator("#pyYears").input_value() == "3"
    assert page.locator("#howHeard").input_value() == "referral"

    context.unroute("**generativelanguage.googleapis.com**")
    context.unroute("**openrouter.ai**")
    page.close()


def test_llm_tier_fails_open_with_no_keys_configured(extension_context, fixture_server):
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)
    # No af_llm_config seeded at all (autouse fixture already cleared it).

    page = context.new_page()
    page.goto(f"{fixture_server}/llm_fixture.html")
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    assert page.locator("#firstName").input_value() == "Ada"
    assert page.locator("#pyYears").input_value() == ""
    assert page.locator("#howHeard").input_value() == ""
    assert result["summary"]["skipped"] >= 2

    page.close()


def test_llm_answer_is_cached_and_not_needed_again(extension_context, fixture_server):
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)
    _seed_llm_config(context, extension_id, {"gemini_api_key": "fake-key-for-test"})
    _mock_gemini_success(context, {"f0": "3", "f1": "LinkedIn"})

    page = context.new_page()
    page.goto(f"{fixture_server}/llm_fixture.html")
    page.bring_to_front()

    first = _trigger_fill(sw)
    assert first["ok"]
    assert page.locator("#pyYears").input_value() == "3"

    # Remove the LLM entirely (no mock, no keys) - if the second fill still
    # gets the right answer, it came from the Tier-0 cache the first LLM
    # call wrote back to, not a fresh provider call.
    context.unroute("**generativelanguage.googleapis.com**")
    _clear_llm_config(context, extension_id)

    page.fill("#pyYears", "")
    page.select_option("#howHeard", "")
    second = _trigger_fill(sw)
    assert second["ok"]
    assert page.locator("#pyYears").input_value() == "3"
    assert page.locator("#howHeard").input_value() == "linkedin"

    cached_labels = {f["label"] for f in second["summary"]["fields"] if f["source"] == "cache"}
    assert {"Years of Python experience", "How did you hear about us?"} <= cached_labels

    page.close()


def test_label_wrapped_select_does_not_leak_option_text(extension_context, fixture_server):
    """Regression for a bug found live on jobs.lever.co: Lever wraps the
    <select> inside its <label>, so resolving the label via el.labels[0]
    without stripping the embedded control returns the question text PLUS
    every option's text. A country option list contains "United States" -
    /state/i matched inside it and the select was misclassified as
    address.region. With region set to "Georgia" (deliberately chosen: a
    US state that is also a country's name, so it matches an option
    exactly), the pre-fix scanner wrote the country "Georgia" into a
    location dropdown. The fix (scanner.js textOfLabelElement) strips
    embedded controls from wrapping labels before reading their text.
    """
    context, extension_id, sw = extension_context
    profile = {
        **TEST_PROFILE,
        "contact": {
            **TEST_PROFILE["contact"],
            "address": {"country": "Canada", "region": "Georgia"},
        },
    }
    _seed_profile(context, extension_id, profile)

    page = context.new_page()
    page.goto(f"{fixture_server}/label_wrapped_select.html")
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    # The heart of the regression: "What is your location?" matches no
    # heuristic once option text stops polluting the label, so the select
    # must stay on its placeholder - NOT get region "Georgia" written in.
    assert page.locator('select[name="candidate-location"]').input_value() == ""

    # Positive control, same wrapping-label markup: proves the strip fix
    # still resolves a clean label ("Country") and fills through it,
    # rather than passing because label-wrapped selects went dark.
    assert page.locator('select[name="country-select"]').input_value() == "CA"
    assert page.locator('input[name="name"]').input_value() == "Ada Lovelace"

    page.close()


def test_derivation_layer_and_polarity(extension_context, fixture_server):
    """Every answer on this form must be extrapolated from profile facts
    (see PHASE3_PROFILE): employer/title from the most recent experience
    entry, years from interval-merged date ranges, education level from
    the degree string, EEO from the decline default - plus the polarity
    trap pair lifted from a real Lever posting: "authorized ... without
    sponsorship" (Yes) right next to "require sponsorship" (No). A
    keyword table that routes on /sponsor/ first - which shipped
    job-apply bots actually do - answers the first one backwards.
    Salary stays blank by design: refusal, not failure.
    """
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, PHASE3_PROFILE)

    page = context.new_page()
    page.goto(f"{fixture_server}/derivation_form.html")
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    assert page.locator("#currentCompany").input_value() == "Analytical Engines Inc"
    assert page.locator("#currentTitle").input_value() == "Software Engineer"
    assert page.locator("#yearsExperience").input_value() == "2"
    assert page.locator("#educationLevel").input_value() == "bach"
    assert page.locator("#gradYear").input_value() == "2027"
    assert page.locator("#race").input_value() == "decline"

    # The polarity pair - both derive from the same two stored booleans
    # (authorized=True, requires_sponsorship=False), routed by pattern
    # specificity, never by runtime negation:
    assert page.locator('input[name="auth_without_sponsor"][value="Yes"]').is_checked()
    assert not page.locator('input[name="auth_without_sponsor"][value="No"]').is_checked()
    assert page.locator('input[name="needs_sponsorship"][value="No"]').is_checked()
    assert not page.locator('input[name="needs_sponsorship"][value="Yes"]').is_checked()
    assert page.locator('input[name="work_auth_ca"][value="Yes"]').is_checked()

    # Deliberate refusals: salary is never derivable, and preference
    # questions (referral source, work mode, relocation) have no stated
    # answer in this profile - all must stay blank, not guessed.
    assert page.locator("#salary").input_value() == ""
    assert page.locator("#referralSource").input_value() == ""
    assert page.locator("#workMode").input_value() == ""
    assert not page.locator('input[name="willing_relocate"][value="Yes"]').is_checked()
    assert not page.locator('input[name="willing_relocate"][value="No"]').is_checked()

    page.close()


# A realistic single-column resume as PLAIN TEXT - the exact input the
# importer receives after the user pastes it. Everything the fill
# pipeline later needs must come out of THIS text, not hand-authored
# JSON.
SAMPLE_RESUME_TEXT = """\
Ada Lovelace
Toronto, ON, Canada | 555-123-4567 | ada@example.com
linkedin.com/in/ada | github.com/ada

SUMMARY
Software engineer with a focus on analytical engines and compilers.

EXPERIENCE
Software Engineer, Analytical Engines Inc    May 2023 - Nov 2025
- Designed the difference pipeline for large-scale computation
- Cut processing time by 40% across the bernoulli module
Engineering Intern | Babbage Labs    May 2022 - Sep 2022
- Built internal tooling for mechanical computation research

EDUCATION
Toronto Metropolitan University    Sep 2022 - Apr 2027
Bachelor of Engineering, Computer Engineering
GPA: 3.8/4.0

TECHNICAL SKILLS
Languages: Python, TypeScript, C
Tools: Git, Docker, Playwright
"""

SAMPLE_DESCRIPTION_TEXT = """\
Canadian citizen, authorized to work in Canada and the US. No sponsorship
needed now or in the future. Looking for $85-95k. Available with 2 weeks
notice. Open to hybrid work in Toronto, but not willing to relocate.
I heard about most roles through LinkedIn.
"""


def test_importer_parses_resume_and_description(extension_context):
    """Runs the REAL importer inside the extension's options page (the
    same UI path a user takes): paste resume + description text, click
    parse, and assert the parsed JSON that lands in the editor. Every
    asserted value must have been EXTRACTED from the raw text - negation
    included: "No sponsorship needed" => requires_sponsorship False,
    while "authorized to work" => authorized True, and "not willing to
    relocate" => False in the same description that affirms hybrid work.
    """
    context, extension_id, _sw = extension_context

    page = context.new_page()
    page.goto(f"chrome-extension://{extension_id}/options/options.html")
    page.click("#importSection summary")  # expand the collapsed <details>
    page.fill("#importResumeText", SAMPLE_RESUME_TEXT)
    page.fill("#importDescriptionText", SAMPLE_DESCRIPTION_TEXT)
    page.click("#importParseButton")

    parsed = json.loads(page.locator("#profileJson").input_value())

    assert parsed["contact"]["first_name"] == "Ada"
    assert parsed["contact"]["last_name"] == "Lovelace"
    assert parsed["contact"]["email"] == "ada@example.com"
    assert parsed["contact"]["phone"] == "555-123-4567"
    assert parsed["contact"]["linkedin"] == "linkedin.com/in/ada"
    assert parsed["contact"]["github"] == "github.com/ada"
    assert parsed["contact"]["address"]["city"] == "Toronto"

    companies = {e["company"] for e in parsed["experience"]}
    titles = {e["title"] for e in parsed["experience"]}
    assert "Analytical Engines Inc" in companies
    assert "Software Engineer" in titles
    assert "Engineering Intern" in titles
    ongoing = [e for e in parsed["experience"] if e["company"] == "Analytical Engines Inc"]
    assert ongoing[0]["start"] == "May 2023"
    assert ongoing[0]["end"] == "Nov 2025"

    assert parsed["education"][0]["school"] == "Toronto Metropolitan University"
    assert "Bachelor of Engineering" in parsed["education"][0]["degree"]
    assert parsed["education"][0]["gpa"] == "3.8"
    assert "Python" in parsed["skills"]
    assert "Playwright" in parsed["skills"]
    assert "Languages" not in parsed["skills"]  # category prefixes stripped

    # Description facts, polarity-correct:
    assert parsed["work_authorization"]["authorized"] is True
    assert parsed["work_authorization"]["requires_sponsorship"] is False
    assert "85" in parsed["preferences"]["salary_expectation"]
    assert parsed["preferences"]["notice_period"] == "2 weeks"
    assert parsed["preferences"]["willing_to_relocate"] is False
    assert parsed["preferences"]["work_mode"] == "Hybrid"
    assert parsed["preferences"]["referral_source"] == "LinkedIn"

    page.close()


def test_imported_profile_fills_a_form_end_to_end(extension_context, fixture_server):
    """The full autonomous chain: raw resume + description text -> parse
    via the options UI -> SAVE through the real save button -> fill the
    derivation fixture. Every landed value below traveled
    text -> parser -> profile -> heuristics/derivation -> filler without
    any hand-authored profile JSON in between.
    """
    context, extension_id, sw = extension_context

    options = context.new_page()
    options.goto(f"chrome-extension://{extension_id}/options/options.html")
    options.fill("#profileIdInput", "imported-e2e")
    options.click("#importSection summary")  # expand the collapsed <details>
    options.fill("#importResumeText", SAMPLE_RESUME_TEXT)
    options.fill("#importDescriptionText", SAMPLE_DESCRIPTION_TEXT)
    options.click("#importParseButton")
    options.click("#saveButton")
    options.evaluate("() => chrome.storage.local.set({ af_active_profile_id: 'imported-e2e' })")
    options.close()

    page = context.new_page()
    page.goto(f"{fixture_server}/derivation_form.html")
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    # Derived from the parsed experience/education entries:
    assert page.locator("#currentCompany").input_value() == "Analytical Engines Inc"
    assert page.locator("#currentTitle").input_value() == "Software Engineer"
    # May 2023-Nov 2025 (2.5y) + May-Sep 2022 (0.33y) = 2.83 -> floor 2.
    assert page.locator("#yearsExperience").input_value() == "2"
    assert page.locator("#educationLevel").input_value() == "bach"
    assert page.locator("#gradYear").input_value() == "2027"

    # Stated in the description text, polarity-correct on the trap pair:
    assert page.locator('input[name="auth_without_sponsor"][value="Yes"]').is_checked()
    assert page.locator('input[name="needs_sponsorship"][value="No"]').is_checked()
    assert page.locator('input[name="work_auth_ca"][value="Yes"]').is_checked()
    assert page.locator('input[name="willing_relocate"][value="No"]').is_checked()
    assert page.locator("#workMode").input_value() == "hybrid"
    assert page.locator("#referralSource").input_value() == "li"
    assert "85" in page.locator("#salary").input_value()

    page.close()


def test_react_select_combobox_real_library(extension_context, fixture_server):
    """Drives REAL react-select v5 (loaded from esm.sh, not an imitation):
    open on mousedown, [role="option"] portal options, select on option
    click. Covers all three fuzzy answer classes - literal ("Canada"),
    numeric bucket (2.83 derived years -> "2-4 years"), and synonym class
    (eeo "decline" -> "Prefer not to say") - plus a precision control
    (T-shirt size) that must stay untouched. Assertions read the hidden
    React-state mirrors, so they prove the selection reached component
    state, not just pixels.
    """
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, PHASE3_PROFILE)

    page = context.new_page()
    page.goto(f"{fixture_server}/react_select_fixture.html")
    page.wait_for_selector("#country-input", timeout=30000)  # CDN module load
    page.wait_for_timeout(500)
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    assert page.eval_on_selector("#country-value", "el => el.value") == "ca"
    assert page.eval_on_selector("#years-value", "el => el.value") == "b24"
    assert page.eval_on_selector("#gender-value", "el => el.value") == "pnts"
    assert page.eval_on_selector("#tshirt-value", "el => el.value") == ""

    page.close()
