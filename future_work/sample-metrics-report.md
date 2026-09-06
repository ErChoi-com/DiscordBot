# Operating metrics

Generated 2026-07-29T16:31:11+00:00 from the committed job archives.

- Rows scanned: **689,886**
- Unique postings: **581,575**
- `date_posted` coverage: **90.3%**
- Cold-start rows excluded from latency: **150,726**

## Detection latency by source

Hours between a posting appearing on its source board and this system recording
it. Cold-start backfill excluded. `precise p50` covers only sources returning a
time component — date-only sources are floored to midnight and read high.

| source | postings | `date_posted` fill | n | p50 (h) | p90 (h) | precise p50 (h) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greenhouse | 208,163 | 100.0% | 135,573 | 2.9 | 35.9 | 2.9 |
| workday | 135,476 | 100.0% | 97,196 | 45.8 | 130.8 | — |
| icims | 130,373 | 100.0% | 101,803 | 20.3 | 80.2 | 20.3 |
| ashby | 56,387 | 0.0% | 0 | — | — | — |
| lever | 49,955 | 100.0% | 38,805 | 2.7 | 9.7 | 2.7 |
| linkedin | 789 | 100.0% | 697 | 18.1 | 24.8 | — |
| indeed | 261 | 100.0% | 233 | 18.9 | 24.2 | — |
| glassdoor | 166 | 100.0% | 151 | 16.9 | 25.8 | — |
| bamboohr | 3 | 0.0% | 0 | — | — | — |
| unknown | 2 | 100.0% | 0 | — | — | — |
| **all** | 581,575 | 90.3% | 374,458 | 16.2 | 74.9 | |
