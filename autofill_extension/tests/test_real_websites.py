"""Real-website e2e tests: does the extension actually work against pages
this repo did not build, across different frameworks and label conventions?

Self-built fixtures (test_e2e.py) inevitably reflect the author's own
assumptions about how forms are labeled - this file exists specifically to
find the gaps that only show up against real, messy markup. Two real,
external gaps were found and fixed while writing this file: an
"Current Address" label that the address regex didn't match (the fixture
in test_e2e.py used a plain "Address" label, which happened to match), and
this test methodology issue: __afFillActiveTab() (used to trigger a fill
without simulating a real toolbar-icon click) bypasses the activeTab
permission grant entirely, since activeTab is only ever granted in
response to a genuine user-initiated browser action - something Playwright
cannot simulate for an extension's action button. A real user clicking the
toolbar icon on an arbitrary site never hits this; the automated harness
does. The fix is a TEST-ONLY manifest overlay (a copy of the real
extension directory with a few extra host_permissions entries added just
for the sites exercised below) - the shipped manifest.json is never
modified or touched by this file.

Scope, revised (owner's explicit decision, 2026-07): in addition to
automated-testing-practice sites (demoqa.com, the-internet.herokuapp.com,
automationexercise.com, seleniumbase.io), this file now ALSO exercises a
small, hand-curated set of live ATS job postings (2x Lever, 2x Ashby),
sourced from this repo's own already-scraped job data
(data/jba/jobs/2026-07/2026-07-w1.zip) - deliberately chosen postings,
not a fresh mass-scrape. Ground rules that keep this defensible and
stable:
  - Fill-and-inspect ONLY. Nothing in this file ever clicks submit, and
    the profile data is an obviously fictional test identity.
  - Each ATS test skips (never fails) when its posting has since closed -
    postings rot fast; one of the four originally chosen (an Ashby
    posting) died between selection and test-writing ("Job not found").
  - One posting per company, one visit per test run - lighter traffic
    than a single human opening the page.
Greenhouse/Workday/iCIMS remain an open gap: sampled real postings
redirected to generic search/cookie pages or need bespoke per-platform
navigation, so forcing them here would test the navigation hack, not the
extension. test_live_ats_posting_manual_check stays as the opt-in escape
hatch for any one-off posting.

Bugs found by this file so far (the reason it exists), each fixed at the
source and covered by a regression test: (1) an address regex too narrow
for real "Current Address" labels, (2) __afFillActiveTab bypassing
activeTab's gesture grant (the manifest-overlay fix below), (3) a
hit-test fragile to a transient ad overlay (multi-point sampling in
field_visibility.js), (4) Lever's label-wraps-select markup leaking the
entire country option list into the resolved label, so /state/i matched
"United States" inside the options and misclassified the select
(textOfLabelElement in scanner.js; fixture regression in test_e2e.py),
and (5) the service worker's FILL_RESULT aggregation settling on an
empty subframe's instant total:0 reply while the main frame was still
mid-fill - "0 filled" reported for a form that WAS filling (fixed by
counting FILL_STARTED acks per frame instead of guessing with a timer).

Run with: python -m pytest autofill_extension/tests/test_real_websites.py -v
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import time

import pytest
from playwright.sync_api import BrowserContext, Worker, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

REAL_EXTENSION_DIR = pathlib.Path(__file__).resolve().parent.parent

EXTRA_TEST_HOST_PERMISSIONS = [
    "https://demoqa.com/*",
    "https://the-internet.herokuapp.com/*",
    "https://jobs.lever.co/*",
    "https://jobs.ashbyhq.com/*",
    "https://automationexercise.com/*",
    "https://www.automationexercise.com/*",
    "https://seleniumbase.io/*",
]

TEST_PROFILE = {
    "profile_id": "test",
    "contact": {
        "first_name": "Ada",
        "last_name": "Lovelace",
        "email": "ada@example.com",
        "phone": "5551234567",
        "linkedin": "linkedin.com/in/ada",
        "github": "github.com/ada",
        "website": "ada.dev",
        # A coherent address on purpose: derived.location joins it into
        # "Toronto, Ontario, Canada" and the live tests assert that exact
        # string lands in Lever typeaheads / the Ashby location picker.
        # (The old region="Georgia" label-leak tripwire moved to
        # test_e2e.py's label_wrapped_select fixture regression.)
        "address": {
            "street": "123 Main St",
            "city": "Toronto",
            "region": "Ontario",
            "country": "Canada",
        },
    },
    # Both polarities stored explicitly - questions are ROUTED to the
    # matching boolean by pattern specificity, never negated at runtime.
    "work_authorization": {"authorized": True, "requires_sponsorship": False},
    # Closed date range so derived years never drift with real time.
    "experience": [
        {"company": "Analytical Engines Inc", "title": "Software Engineer", "start": "2023-05", "end": "2025-11"},
    ],
    "education": [
        {
            "school": "Toronto Metropolitan University",
            "degree": "BEng, Computer Engineering",
            "start": "2022-09",
            "end": "2027-04",
        }
    ],
    "skills": ["Python"],
}


@pytest.fixture(scope="module")
def test_only_extension_dir(tmp_path_factory):
    # Copy, don't symlink: we're about to edit manifest.json and must never
    # touch the real one that ships.
    dest = tmp_path_factory.mktemp("extension-copy") / "autofill_extension"
    shutil.copytree(REAL_EXTENSION_DIR, dest, ignore=shutil.ignore_patterns("tests", "__pycache__"))

    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["host_permissions"] = manifest["host_permissions"] + EXTRA_TEST_HOST_PERMISSIONS
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return dest


@pytest.fixture(scope="module")
def extension_context(test_only_extension_dir, tmp_path_factory):
    user_data_dir = tmp_path_factory.mktemp("chrome-profile")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(user_data_dir),
            headless=True,
            channel="chromium",
            args=[
                f"--disable-extensions-except={test_only_extension_dir}",
                f"--load-extension={test_only_extension_dir}",
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


@pytest.fixture(autouse=True)
def _fresh_cache(extension_context):
    context, extension_id, _sw = extension_context
    _clear_cache(context, extension_id)
    yield


def test_demoqa_practice_form_native_fields(extension_context):
    """demoqa.com/automation-practice-form: a real, external, purpose-built
    QA practice form. Its text inputs use PLACEHOLDER text with no
    <label for>, no aria-label, and autocomplete="off" - the weakest, most
    fallback-dependent label signal our scanner supports - unlike this
    repo's own fixtures, which all used explicit <label for> associations.
    """
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    page.goto("https://demoqa.com/automation-practice-form", wait_until="domcontentloaded")
    page.wait_for_timeout(1000)
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    assert page.locator("#firstName").input_value() == "Ada"
    assert page.locator("#lastName").input_value() == "Lovelace"
    assert page.locator("#userEmail").input_value() == "ada@example.com"
    assert page.locator("#userNumber").input_value() == "5551234567"
    # "Current Address" - real label text this repo's own fixtures never
    # used (they said just "Address"); this is the exact real-world gap
    # this file exists to find. See field_heuristics.js's address pattern.
    assert page.locator("#currentAddress").input_value() == "123 Main St"


def test_demoqa_practice_form_leaves_unmapped_widgets_alone(extension_context):
    """Fields with nothing corresponding in the profile schema (gender - no
    such concept in our profile) or that are Phase-3 territory (a
    react-datepicker date-of-birth field, a react-select subjects
    autocomplete, cascading react-select state/city pickers, a file
    upload) must NOT get a wrong or garbage value. Silence is the correct
    behavior here, not a bug to chase - this test documents the current,
    honest boundary rather than asserting these will someday be filled.
    """
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    page.goto("https://demoqa.com/automation-practice-form", wait_until="domcontentloaded")
    page.wait_for_timeout(1000)
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    assert not page.locator("#gender-radio-1").is_checked()
    assert not page.locator("#gender-radio-2").is_checked()
    assert not page.locator("#gender-radio-3").is_checked()
    assert page.locator("#subjectsInput").input_value() == ""
    assert page.eval_on_selector("#uploadPicture", "el => el.value") == ""


def test_the_internet_login_form_does_not_touch_credentials(extension_context):
    """the-internet.herokuapp.com/login: username/password have nothing to
    do with a job-application profile. Confirms precision, not just
    recall - the tool should never guess-fill a field it has no real
    mapping for, even one with a completely ordinary <label for>.
    """
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    page.goto("https://the-internet.herokuapp.com/login", wait_until="domcontentloaded")
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    assert page.locator("#username").input_value() == ""
    assert page.locator("#password").input_value() == ""


def test_the_internet_dropdown_is_detected_but_not_forced(extension_context):
    """the-internet.herokuapp.com/dropdown: a real native <select> with
    generic option text ("Option 1"/"Option 2") that matches nothing in a
    real profile. Confirms the scanner finds real external <select>
    elements (not just ones from our own fixtures) and correctly leaves
    an unmappable one on its default selection rather than picking
    something arbitrary.
    """
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    page.goto("https://the-internet.herokuapp.com/dropdown", wait_until="domcontentloaded")
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result
    assert page.locator("#dropdown").input_value() == ""


# --- Live ATS postings (curated from this repo's own scraped job data) ---

LEVER_LIFESTANCE_APPLY = "https://jobs.lever.co/lifestance/cbb57d24-40fb-4555-9780-28e140b03d12/apply"
LEVER_NOVIR_APPLY = "https://jobs.lever.co/novir/b1503045-2226-4775-8b59-9f848fa4b2f4/apply"
ASHBY_NOTION_POSTING = "https://jobs.ashbyhq.com/notion/156aac9e-a4f5-41ae-96ba-2209c16a0153"
ASHBY_LINEAR_POSTING = "https://jobs.ashbyhq.com/linear/c0abc97e-b4d7-4ad7-bd13-bc40f4023227"


def _goto_lever_apply_or_skip(page, url: str) -> None:
    """A dead Lever posting stops rendering the apply form (redirect to the
    company's job list or a 404 page). input[name="name"] is raw Lever
    markup, checked with plain Playwright - its absence means the posting
    is gone, not that the extension regressed, so skip rather than fail.
    """
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(2000)
    if page.locator('input[name="name"]').count() == 0:
        pytest.skip(f"Lever posting no longer serves an apply form: {url}")


def _goto_ashby_application_or_skip(page, posting_url: str) -> None:
    """Ashby is a React SPA; /application by direct navigation renders the
    form after hydration (verified live + against Ashby's own bundle
    source). A closed posting renders "Job not found" instead - skip,
    the same way the finni-health posting originally picked for this file
    died before the test could be written.
    """
    page.goto(f"{posting_url}/application", wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_selector("#_systemfield_name", timeout=25000)
    except PlaywrightTimeoutError:
        body = page.evaluate("() => (document.body && document.body.innerText) || ''")
        if "job not found" in body.lower() or "no longer accepting" in body.lower():
            pytest.skip(f"Ashby posting closed: {posting_url}")
        raise
    page.wait_for_timeout(1000)  # let the rest of the form hydrate


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def test_lever_lifestance_precision_on_eeo_and_pronouns(extension_context):
    """A Lever apply form whose bulk is fields the profile must NOT touch:
    a 12-checkbox pronouns block, and gender/race/veteran EEO selects.
    Asserts the four real fills land AND that every sensitive/unmappable
    field stays untouched - on a live form, precision failures are worse
    than recall failures.
    """
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    _goto_lever_apply_or_skip(page, LEVER_LIFESTANCE_APPLY)
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    assert page.locator('[data-qa="name-input"]').input_value() == "Ada Lovelace"
    assert page.locator('[data-qa="email-input"]').input_value() == "ada@example.com"
    assert page.locator('[data-qa="phone-input"]').input_value() == "5551234567"
    assert page.locator('input[name="urls[LinkedIn]"]').input_value() == "linkedin.com/in/ada"
    # Derived answers - none of these strings exist literally in the
    # profile's flat fields:
    assert page.locator('[data-qa="org-input"]').input_value() == "Analytical Engines Inc"
    for eeo in ("gender", "race", "veteran"):
        assert (
            page.locator(f'select[name="eeo[{eeo}]"]').input_value() == "Decline to self-identify"
        ), eeo

    # Documented limitation, verified live: Lever's location typeahead
    # CLEARS free text on blur unless a suggestion from its own (non-ARIA)
    # dropdown was picked - even real keyboard typing doesn't survive. The
    # filler's post-blur read-back reports this honestly (ok:false), so
    # the field ends empty rather than lying in the summary.
    assert page.locator('[data-qa="location-input"]').input_value() == ""
    assert result["summary"]["filled"] == 8
    # name="pronouns" is shared by the checkboxes AND the free-text
    # "Custom" pronoun input, so constrain by type for is_checked().
    pronoun_boxes = page.locator('input[type="checkbox"][name="pronouns"]')
    assert pronoun_boxes.count() >= 10
    for i in range(pronoun_boxes.count()):
        assert not pronoun_boxes.nth(i).is_checked()
    assert page.locator('input[type="text"][name="pronouns"]').first.input_value() == ""

    # type=hidden fields (hCaptcha's response slot among them) must never
    # be written - scanner excludes them before the visibility gate runs.
    assert page.eval_on_selector('input[name="h-captcha-response"]', "el => el.value") == ""

    page.close()


def test_lever_novir_derivation_polarity_and_refusals(extension_context):
    """The heaviest live test. This posting's custom cards ask FIVE yes/no
    questions; exactly one ("Are you authorized to work in the United
    States without sponsorship?") is derivable from the stored
    work-authorization booleans - and only via the COMPOUND pattern,
    because a /sponsor/-routed keyword table (which shipped job-apply
    bots actually have) answers that phrasing backwards. The other four
    (transportation, physical requirements, license, vaccination) plus
    the rate/availability textareas are judgment calls the derivation
    layer must REFUSE, not guess. The "What is your location?" country
    dropdown resolves "Toronto, Ontario, Canada" -> Canada via fuzzy
    containment (this same select is bug #4's label-leak site; that
    regression now lives in test_e2e.py's fixture test).
    """
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    _goto_lever_apply_or_skip(page, LEVER_NOVIR_APPLY)
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    assert page.locator('[data-qa="name-input"]').input_value() == "Ada Lovelace"
    assert page.locator('[data-qa="email-input"]').input_value() == "ada@example.com"
    assert page.locator('[data-qa="phone-input"]').input_value() == "5551234567"
    assert page.locator('input[name="urls[LinkedIn]"]').input_value() == "linkedin.com/in/ada"
    assert page.locator('input[name="urls[GitHub]"]').input_value() == "github.com/ada"
    assert page.locator('input[name="urls[Portfolio]"]').input_value() == "ada.dev"
    assert page.locator('[data-qa="org-input"]').input_value() == "Analytical Engines Inc"
    # Lever's location typeahead clears unselected free text on blur (see
    # the lifestance test) - the COUNTRY select below is where the derived
    # location actually lands on this form.
    assert page.locator('[data-qa="location-input"]').input_value() == ""
    assert page.locator('[data-qa="candidate-location-select"]').input_value() == "CA"

    # THE polarity assertion, on a real employer's real question. The
    # work-auth card is the posting's first custom card (field0).
    assert page.locator('input[type="radio"][name$="[field0]"][value="Yes"]').is_checked()
    assert not page.locator('input[type="radio"][name$="[field0]"][value="No"]').is_checked()

    # The four non-derivable yes/no cards must stay untouched - refusal,
    # not failure.
    for field_index in (1, 2, 3, 4):
        group = page.locator(f'input[type="radio"][name$="[field{field_index}]"]')
        for i in range(group.count()):
            assert not group.nth(i).is_checked(), f"field{field_index} radio {i} was guessed"

    # No profile concept maps to these; silence is correct.
    assert page.locator('input[name="urls[Twitter]"]').input_value() == ""
    assert page.locator('input[name="urls[Other]"]').input_value() == ""
    for i in range(page.locator('textarea[name^="cards["]').count()):
        assert page.locator('textarea[name^="cards["]').nth(i).input_value() == ""

    # 9 = name, email, phone, company, LinkedIn, GitHub, portfolio, the
    # country select, and the work-auth radio. The location typeahead
    # attempt is honestly reported ok:false (cleared on blur), so it is
    # NOT in the filled count.
    assert result["summary"]["filled"] == 9

    page.close()


def test_ashby_notion_application_react_fills_stick(extension_context):
    """Ashby: fully React-rendered SPA form (label for=id association,
    field "path" as both id and name). The values must still be present
    after a settle wait - a plain `.value =` write is exactly what React
    reverts on re-render, so surviving the wait proves the native-setter
    path works against a real production React form, not just the CDN
    fixture in test_e2e.py.
    """
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    _goto_ashby_application_or_skip(page, ASHBY_NOTION_POSTING)
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result
    page.wait_for_timeout(500)  # give React a re-render window to revert

    assert page.locator("#_systemfield_name").input_value() == "Ada Lovelace"
    assert page.locator("#_systemfield_email").input_value() == "ada@example.com"
    # Ashby's tel input may reformat; assert the digits, not the mask.
    assert _digits(page.get_by_label("Phone", exact=True).input_value()) == "5551234567"
    assert page.get_by_label("LinkedIn Profile", exact=True).input_value() == "linkedin.com/in/ada"

    # The location autocomplete is a genuine custom combobox (typed query
    # -> async Places-style suggestions -> option click). The widget
    # driver must have typed "Toronto", waited for the network-backed
    # option list, and picked the exact city. Ashby renders the committed
    # pick as the input's value (NOT as element text - inner_text() shows
    # nothing even on success).
    assert (
        page.locator('input[aria-haspopup="listbox"]').first.input_value()
        == "Toronto, Ontario, Canada"
    )
    assert result["summary"]["filled"] == 5

    page.close()


def test_ashby_linear_country_text_field_and_essay_silence(extension_context):
    """A second, structurally different Ashby form: no phone field, a
    free-text "What country are you based in?" (the one live field that
    exercises contact.address.country end-to-end), plus cover-letter and
    essay textareas that must stay empty with no LLM keys configured
    (fail-open check against a real page, not a fixture).
    """
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    _goto_ashby_application_or_skip(page, ASHBY_LINEAR_POSTING)
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result
    page.wait_for_timeout(500)

    assert page.locator("#_systemfield_name").input_value() == "Ada Lovelace"
    assert page.locator("#_systemfield_email").input_value() == "ada@example.com"
    assert page.get_by_label("LinkedIn profile", exact=True).input_value() == "linkedin.com/in/ada"
    assert (
        page.get_by_label("What country are you based in?", exact=True).input_value() == "Canada"
    )
    assert result["summary"]["filled"] == 4

    assert page.get_by_label("Twitter handle", exact=True).input_value() == ""
    textareas = page.locator("textarea")
    assert textareas.count() >= 3  # cover letter + two essay questions
    for i in range(textareas.count()):
        assert textareas.nth(i).input_value() == ""

    page.close()


def test_automationexercise_rationalization_fills_only_first_email(extension_context):
    """automationexercise.com/login has THREE fields that all classify as
    contact.email (login email, signup email, footer subscribe box) on
    one page - no ids or <label>s on the form fields, so matching runs
    off name attributes and placeholders alone. The rationalization pass
    must fill exactly one (the first in scan order: the login form's) and
    leave the other two empty, and must never touch the password.
    """
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    page.goto("https://automationexercise.com/login", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(1500)
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    assert page.locator('[data-qa="login-email"]').input_value() == "ada@example.com"
    assert page.locator('[data-qa="signup-name"]').input_value() == "Ada Lovelace"
    assert result["summary"]["filled"] == 2

    assert page.locator('[data-qa="signup-email"]').input_value() == ""
    assert page.locator("#susbscribe_email").input_value() == ""  # sic - site's own typo
    assert page.locator('[data-qa="login-password"]').input_value() == ""

    page.close()


def test_seleniumbase_demo_page_is_a_complete_noop(extension_context):
    """seleniumbase.io/demo_page packs every native widget type onto one
    page (pre-filled inputs, a READONLY field, a range slider, a native
    select, five checkboxes incl. one pre-checked, a radio group) - and
    none of it maps to a job-application profile. The strongest available
    precision assertion: snapshot every control before the fill, fill,
    and require the page byte-identical afterwards. summary.total guards
    against the vacuous pass where the page simply failed to load.
    """
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    page.goto("https://seleniumbase.io/demo_page", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(1000)
    page.bring_to_front()

    snapshot_js = """
        () => Array.from(document.querySelectorAll('input, select, textarea')).map((el) => ({
            id: el.id,
            name: el.name || '',
            type: el.type || el.tagName.toLowerCase(),
            value: el.value,
            checked: el.checked === true,
        }))
    """
    before = page.evaluate(snapshot_js)

    result = _trigger_fill(sw)
    assert result["ok"], result

    after = page.evaluate(snapshot_js)
    assert result["summary"]["total"] >= 10  # the scanner really saw the page
    assert result["summary"]["filled"] == 0
    assert before == after

    # Spot-check the single trickiest case explicitly: the READONLY field
    # (which the visibility gate must exclude before anything can write
    # to it) kept its shipped text.
    assert page.locator("#readOnlyText").input_value() == "The Color is Green"

    page.close()


def test_imported_resume_profile_fills_live_lever(extension_context):
    """The full autonomous promise, against a REAL employer form: raw
    resume text + description text -> parsed through the actual options
    UI -> saved -> fills the novir Lever posting. No hand-authored
    profile JSON anywhere in this test. Notable extras this exercises
    beyond the local fixture chain: the resume's contact line says
    "Toronto, ON, Canada" (abbreviated region, as real resumes do), and
    the country dropdown still resolves to Canada; the description's
    "No sponsorship needed" still answers the compound authorization
    question correctly; and the hourly-rate textarea stays EMPTY even
    though a salary expectation was stated - annual-to-hourly is a unit
    conversion routing must refuse.
    """
    from test_e2e import SAMPLE_DESCRIPTION_TEXT, SAMPLE_RESUME_TEXT

    context, extension_id, sw = extension_context

    options = context.new_page()
    options.goto(f"chrome-extension://{extension_id}/options/options.html")
    options.fill("#profileIdInput", "imported-live")
    options.click("#importSection summary")
    options.fill("#importResumeText", SAMPLE_RESUME_TEXT)
    options.fill("#importDescriptionText", SAMPLE_DESCRIPTION_TEXT)
    options.click("#importParseButton")
    options.click("#saveButton")
    options.evaluate("() => chrome.storage.local.set({ af_active_profile_id: 'imported-live' })")
    options.close()

    page = context.new_page()
    _goto_lever_apply_or_skip(page, LEVER_NOVIR_APPLY)
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result

    assert page.locator('[data-qa="name-input"]').input_value() == "Ada Lovelace"
    assert page.locator('[data-qa="email-input"]').input_value() == "ada@example.com"
    assert page.locator('[data-qa="phone-input"]').input_value() == "555-123-4567"
    assert page.locator('[data-qa="org-input"]').input_value() == "Analytical Engines Inc"
    assert page.locator('input[name="urls[LinkedIn]"]').input_value() == "linkedin.com/in/ada"
    assert page.locator('input[name="urls[GitHub]"]').input_value() == "github.com/ada"
    assert page.locator('[data-qa="candidate-location-select"]').input_value() == "CA"
    assert page.locator('input[type="radio"][name$="[field0]"][value="Yes"]').is_checked()

    # Stated $85-95k is ANNUAL; the "desired hourly rate" textarea must
    # stay empty rather than receive an unconverted range.
    for i in range(page.locator('textarea[name^="cards["]').count()):
        assert page.locator('textarea[name^="cards["]').nth(i).input_value() == ""

    # name, email, phone, company, LinkedIn, GitHub, country select,
    # work-auth radio. (No portfolio in the sample resume; the location
    # typeahead clears on blur as documented above.)
    assert result["summary"]["filled"] == 8

    page.close()


@pytest.mark.skipif(
    not os.environ.get("AF_LIVE_ATS_TEST_URL"),
    reason=(
        "Opt-in only: set AF_LIVE_ATS_TEST_URL to a specific real job-posting "
        "URL to run this manually. Not run by default - see this file's "
        "module docstring for why real ATS postings aren't in the automated "
        "suite."
    ),
)
def test_live_ats_posting_manual_check(extension_context):
    context, extension_id, sw = extension_context
    _seed_profile(context, extension_id, TEST_PROFILE)

    page = context.new_page()
    page.goto(os.environ["AF_LIVE_ATS_TEST_URL"], wait_until="domcontentloaded")
    page.wait_for_timeout(1500)
    page.bring_to_front()

    result = _trigger_fill(sw)
    assert result["ok"], result
    print(json.dumps(result["summary"], indent=2))
    # Deliberately no submit anywhere in this file.
