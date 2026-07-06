# TODO

## Job Scraping

- [x] Use https://github.com/Feashliaa/job-board-aggregator to scrape job sites dynamically
  - Scrapes ATS platforms directly (Greenhouse, Lever, Ashby, Workday) via `ats_service.py`
  - Company slug lists in `data/ats_companies.json` (86 companies total)
  - Concurrent per-platform scraping with ThreadPoolExecutor
  - Dead slug caching to skip 404/410 companies on future runs
  - Hooked into `scrape_job_postings` pipeline with dedupe + shape_job_item
  - BambooHR dropped (Cloudflare-blocked, requires headless browser)
