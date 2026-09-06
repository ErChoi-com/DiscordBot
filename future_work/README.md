# Future work — remaining items

Two of the four items here have now landed. What is left is deliberately **not**
wired into the running bot: nothing in this folder is imported, scheduled, or
executed by anything, and pytest does not collect from it (`pytest.ini` scopes
collection to `tests/`).

## Applied

| Item | Where it went |
| --- | --- |
| `compute_metrics.py` | `scripts/compute_metrics.py` |
| its tests | `tests/test_archive_metrics.py` (7 tests) |
| `01-ats-flush-dead-slugs-fix.patch` | applied to `src/services/ats_service.py`; covered by `tests/test_dead_slug_ttl.py` (6 tests) |

The dead-slug fix was preventative rather than corrective: at the time it landed
all 19,697 cached dead slugs were still inside the 90-day TTL, so nothing was
immediately eligible for re-probing. Without it the TTL would never have fired
for a platform that recorded no *new* deaths in a run, and the effective company
list would have shrunk permanently.

## Still here

| File | What it is | Where it goes |
| --- | --- | --- |
| `mypy.ini` | Type-checking ratchet | repo root |
| `02-ci-mypy-step.patch` | Adds the mypy step to `.github/workflows/ci.yml` | `git apply` |
| `metrics.yml` | Daily GitHub Actions job that regenerates and commits the metrics report | `.github/workflows/` |
| `sample-metrics-report.md` | Example output, generated 2026-07-29 | reference only |

```bash
# Type ratchet (no runtime impact; gates CI once the patch below lands)
mv future_work/mypy.ini .
python -m pip install mypy && python -m mypy      # expect: Success

# CI type-check step
git apply future_work/02-ci-mypy-step.patch
```

Note before enabling `metrics.yml`: it commits from CI, and the archive commit
path in `merge_data._git_commit_monthly` has its own identity problem under a
service account. Settle that first.
