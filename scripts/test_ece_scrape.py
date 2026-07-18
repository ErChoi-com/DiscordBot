import sys
sys.path.insert(0, 'src')
from services.job_service import scrape_job_postings, matches_role_filters, matches_search_parameters_semantic

print("Scraping with ECE job goon settings: keywords='computer software \"electrical engineering\" \"intern\"'")

results = scrape_job_postings(
    site_names=["indeed", "linkedin", "glassdoor"],
    keywords='computer software "electrical engineering" "intern"',
    location="Canada",
    configured_python_exe=None,
    hours_old=72,
    results_wanted=50,
    radius_miles=100,
    country_indeed="CANADA",
    source_url="ECE-test",
    allow_north_america=False,
)

daanaa = [r for r in results if 'daanaa' in str(r.get('company', '')).lower()]
print("Total raw results: " + str(len(results)))
print("Daanaa Resolution in ECE search: " + str(len(daanaa)))

if daanaa:
    for d in daanaa:
        print("  FOUND: " + str(d.get('title')) + " | " + str(d.get('company')) + " | " + str(d.get('location')))
else:
    print("  NOT FOUND - Daanaa Resolution does not come back from job boards with ECE keywords")

role_passed = [r for r in results if matches_role_filters(str(r.get("title", "")), ["internship"])]
sem_passed = [r for r in role_passed if matches_search_parameters_semantic(r, 'computer software "electrical engineering" "intern"', 'Canada', ['internship'], threshold=0.42)]
print("After role_filter: " + str(len(role_passed)))
print("After semantic thr 0.42: " + str(len(sem_passed)))

print()
print("=== Top 15 titles from ECE search ===")
for r in results[:15]:
    print("  - " + str(r.get('title')) + " @ " + str(r.get('company')))
