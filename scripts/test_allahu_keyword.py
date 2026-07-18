import sys
sys.path.insert(0, 'src')
from services.job_service import scrape_job_postings

results = scrape_job_postings(
    site_names=["indeed", "linkedin", "glassdoor"],
    keywords='"Electrical engineering" intern',
    location="Canada",
    configured_python_exe=None,
    hours_old=72,
    results_wanted=30,
    radius_miles=100,
    country_indeed="CANADA",
    allow_north_america=False,
)
daanaa = [r for r in results if 'daanaa' in str(r.get('company', '')).lower()]
print("Total:", len(results))
print("Daanaa found:", len(daanaa))
for d in daanaa:
    print(" -", d.get('title'), "|", d.get('company'), "|", d.get('site_label'))
if not daanaa:
    print("Daanaa not live on job boards right now (job may have aged out)")
