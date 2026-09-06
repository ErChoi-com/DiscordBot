# pyright: reportMissingImports=false

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import job_service


def test_all_supported_job_sites_includes_custom_and_indeed(monkeypatch) -> None:
    monkeypatch.setattr(job_service, "supported_jobspy_sites", lambda configured_exe=None: ["linkedin", "google"])

    sites = job_service.all_supported_job_sites()

    assert "glassdoor" in sites
    assert "zip_recruiter" in sites
    assert "linkedin" in sites
    assert "google" in sites
    assert "indeed" in sites


def test_scrape_job_descriptions_from_all_sites_normalizes_output(monkeypatch) -> None:
    monkeypatch.setattr(job_service, "all_supported_job_sites", lambda configured_python_exe=None: ["indeed", "linkedin"])

    calls: list[str] = []

    def _fake_scrape_job_postings(
        site_names,
        keywords,
        location,
        configured_python_exe=None,
        hours_old=168,
        results_wanted=20,
        radius_miles=25,
        country_indeed="USA",
        source_url=None,
        allow_north_america=False,
    ):
        site = site_names[0]
        calls.append(site)
        if site == "indeed":
            return [
                {
                    "title": "Python Developer",
                    "link": "https://www.indeed.com/viewjob?jk=abc123&utm_source=test",
                    "description": "Build APIs and automate workflows.",
                    "site": "indeed",
                    "sites": ["indeed"],
                }
            ]
        return [
            {
                "title": "Backend Engineer",
                "link": "https://www.linkedin.com/jobs/view/123",
                "description": "Maintain backend services.",
                "site": "linkedin",
                "sites": ["linkedin"],
                "apply_link": "https://company.example/apply/123",
            }
        ]

    monkeypatch.setattr(job_service, "scrape_job_postings", _fake_scrape_job_postings)

    rows = job_service.scrape_job_descriptions_from_all_sites(
        keywords="python",
        location="Toronto",
        results_per_site=2,
        max_descriptions=10,
    )

    assert calls == ["indeed", "linkedin"]
    assert len(rows) == 2

    indeed_row = next(row for row in rows if row["site"] == "indeed")
    assert indeed_row["link"] == "https://www.indeed.com/viewjob?jk=abc123"
    assert "Description:" in indeed_row["description"]

    linkedin_row = next(row for row in rows if row["site"] == "linkedin")
    assert linkedin_row["apply_link"] == "https://company.example/apply/123"
    assert "Maintain backend services." in linkedin_row["description"]


class _FakeResp:
    def __init__(self, status=200, body=""):
        self.status_code = status
        self.text = body


class _FakeSession:
    def __init__(self, body: str):
        self._body = body

    def get(self, url, timeout=15):
        return _FakeResp(200, self._body)


def test_glassdoor_location_from_detail_page_includes_province_and_country() -> None:
    body = '{"addressLocality":"Lively","addressRegion":"ON","addressCountry":"CA"}'

    location = job_service.glassdoor_location_from_detail_page(
        _FakeSession(body), "https://www.glassdoor.com/job-listing/123", "Lively"
    )

    assert location == "Lively, Ontario, Canada"


def test_glassdoor_location_from_detail_page_unescapes_nested_json() -> None:
    body = '{\\"addressLocality\\":\\"Lively\\",\\"addressRegion\\":\\"ON\\",\\"addressCountry\\":\\"CA\\"}'

    location = job_service.glassdoor_location_from_detail_page(
        _FakeSession(body), "https://www.glassdoor.com/job-listing/123", "Lively"
    )

    assert location == "Lively, Ontario, Canada"


def test_glassdoor_location_from_detail_page_falls_back_when_no_match() -> None:
    location = job_service.glassdoor_location_from_detail_page(
        _FakeSession("<html>no structured data here</html>"),
        "https://www.glassdoor.com/job-listing/123",
        "Lively",
    )

    assert location == "Lively"


def test_shape_job_item_never_repeats_the_company_inside_the_location_parens():
    """`[LinkedIn/Trench Group] AI & ML Intern (Trench Group, Scarborough, ...)`
    named the company twice in one line and ran it together with the city.
    The label carries the company; the parentheses carry the location.
    """
    item = job_service.shape_job_item(
        {
            "title": "AI & ML Intern - Electrical Engineer",
            "company": "Trench Group",
            "location": "Scarborough, Ontario, Canada",
            "job_url": "https://www.linkedin.com/jobs/view/4434008080",
            "_source_site": "linkedin",
        },
        "linkedin",
    )
    assert item["site_label"] == "LinkedIn/Trench Group"
    assert item["title"] == "AI & ML Intern - Electrical Engineer (Scarborough, Ontario, Canada)"
    assert item["title"].count("Trench Group") == 0


def test_shape_job_item_drops_a_company_that_is_just_the_job_board_name():
    """A "company" of "LinkedIn" on a LinkedIn posting is not a company, so it
    is dropped rather than moved into the parentheses next to the city.
    """
    item = job_service.shape_job_item(
        {
            "title": "AI & ML Intern",
            "company": "LinkedIn",
            "location": "Toronto, ON",
            "job_url": "https://www.linkedin.com/jobs/view/1",
            "_source_site": "linkedin",
        },
        "linkedin",
    )
    assert item["site_label"] == "LinkedIn"
    assert item["title"] == "AI & ML Intern (Toronto, ON)"


def test_shape_job_item_keeps_a_company_the_label_could_not_carry():
    """When the company never made it into the label there is nowhere else for
    it to go, so it stays in the parentheses rather than being lost.
    """
    item = job_service.shape_job_item(
        {
            "title": "AI & ML Intern",
            "company": "Trench Group",
            "location": "Toronto, ON",
            "job_url": "https://example.com/1",
        },
        "custom",
    )
    assert "Trench Group" in item["title"]


def test_a_job_board_name_is_never_stored_as_the_company():
    """A company of "LinkedIn" on a LinkedIn posting would reach the archive and
    let job_match collapse unrelated postings that merely share it.
    """
    item = job_service.shape_job_item(
        {"title": "Data Intern", "company": "LinkedIn", "location": "Toronto, ON",
         "job_url": "https://www.linkedin.com/jobs/view/1", "_source_site": "linkedin"},
        "linkedin",
    )
    assert "company" not in item
    assert item["title"] == "Data Intern (Toronto, ON)"


def test_an_employer_that_shares_a_job_boards_name_keeps_its_own():
    """Google, Lever, Greenhouse, Ashby and Workday are job boards AND companies
    that post their own jobs. Matching the company against every known board
    dropped a Google internship's employer entirely; only the board this row
    actually came from can be a placeholder.
    """
    for company, site in [("Google", "linkedin"), ("Lever", "linkedin"), ("Greenhouse", "linkedin")]:
        item = job_service.shape_job_item(
            {"title": "Software Developer Intern", "company": company,
             "location": "Toronto, Ontario, Canada",
             "job_url": "https://www.linkedin.com/jobs/view/4459787219",
             "_source_site": site},
            site,
        )
        assert item["site_label"] == f"LinkedIn/{company}", company
        assert item["company"] == company, company


def test_a_placeholder_matching_its_own_board_is_still_dropped():
    """The case the check exists for: "Greenhouse" on a Greenhouse posting."""
    item = job_service.shape_job_item(
        {"title": "Eng Intern", "company": "Greenhouse", "location": "SF, CA",
         "job_url": "https://boards.greenhouse.io/x/1", "_source_site": "greenhouse"},
        "greenhouse",
    )
    assert item["site_label"] == "Greenhouse"
    assert "company" not in item


def test_a_real_company_name_is_not_mangled_into_the_label():
    """`.replace("-", " ").title()` turned "CaptiveAire - Region 114 Western PA"
    into "Captiveaire   Region 114 Western Pa" -- three spaces where the dash
    was, and the casing of both "CaptiveAire" and "PA" destroyed. A name that
    already reads as one is used verbatim.
    """
    item = job_service.shape_job_item(
        {"title": "Technical Sales Intern- Mississauga",
         "company": "CaptiveAire - Region 114 Western PA",
         "location": "Mississauga, Ontario, Canada",
         "job_url": "https://www.linkedin.com/jobs/view/4461028224",
         "_source_site": "linkedin"},
        "linkedin",
    )
    assert item["site_label"] == "LinkedIn/CaptiveAire - Region 114 Western PA"
    assert item["title"] == "Technical Sales Intern- Mississauga (Mississauga, Ontario, Canada)"


def test_an_ats_slug_is_still_opened_up_into_a_readable_name():
    """The reason the rewrite exists: ATS boards hand over "trench-group"."""
    item = job_service.shape_job_item(
        {"title": "Data Intern", "company": "trench-group", "location": "Toronto, ON",
         "job_url": "https://boards.greenhouse.io/x/1", "_source_site": "greenhouse"},
        "greenhouse",
    )
    assert item["site_label"].endswith("/Trench Group")


def test_an_acronym_company_keeps_its_capitals():
    """.title() would render "IBM" as "Ibm" and "iA Group" as "Ia Group"."""
    for name in ("IBM", "iA Group (Industrial)", "AT&T"):
        item = job_service.shape_job_item(
            {"title": "Intern", "company": name, "location": "Toronto, ON",
             "job_url": "https://x/1", "_source_site": "linkedin"},
            "linkedin",
        )
        assert item["site_label"] == f"LinkedIn/{name}"
