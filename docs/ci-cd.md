# CI/CD Setup

This repository now includes GitHub Actions workflows in `.github/workflows/`.

## Workflows

- `ci.yml`
  - Runs on pushes to `main` or `master`, and on all pull requests.
  - Uses Python `3.14` — the interpreter the bot actually runs under
    (`run.bat`'s `py -3` / `PYTHON_EXE`) and the one the suite is verified
    against locally.
  - Installs dependencies and runs `pytest -q`.

- `cd-release.yml`
  - Runs when you push a tag starting with `v` (example: `v1.0.0`).
  - Also supports manual runs via **Actions -> CD Release -> Run workflow**.
  - Executes tests, creates a zip archive from the current commit, and publishes a GitHub Release with that artifact.

## How to trigger release CD

1. Commit and push your changes.
2. Create and push a version tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

3. Check the **Actions** tab for the `CD Release` run.

## Secrets and permissions

- No custom secret is required for these workflows.
- The workflows use the built-in `GITHUB_TOKEN`.
- Ensure Actions have permission to create releases:
  - Repo Settings -> Actions -> General -> Workflow permissions -> `Read and write permissions`.

## Notes

- The full suite passes locally on Python 3.14 (`589 passed` as of 2026-07-18);
  CI is pinned to the same version so a red run means a real regression, not
  an environment mismatch.
- Semantic job matching needs `requirements-semantic.txt` (torch is heavy);
  the suite does not require it — the feature degrades to keyword matching
  when absent, so CI installs only `requirements.txt`.
- The workflows intentionally run tests before publishing release artifacts to prevent shipping broken commits.
