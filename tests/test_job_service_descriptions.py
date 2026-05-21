# pyright: reportMissingImports=false

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import job_service


def test_all_supported_job_sites_includes_custom_and_indeed(monkeypatch) -> None:
    monkeypatch.setattr(job_service, "supported_jobspy_sites", lambda configured_exe=None: ["linkedin", "google"])

    sites = job_service.all_supported_job_sites()

    assert job_service.JOBBANK_CANADA_SITE in sites
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
        jobbank_search_query=None,
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
