"""One-time migration: import existing JSON/zip job data into SQLite."""
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services.jba.merge_data import _get_conn, _dedup_key, _JOBS_DIR

def main():
    conn = _get_conn()

    # --- Migrate seen_urls.json ---
    seen_path = _JOBS_DIR / "_seen_urls.json"
    if seen_path.exists():
        data = json.loads(seen_path.read_text(encoding="utf-8"))
        urls = data if isinstance(data, list) else []
        conn.executemany(
            "INSERT OR IGNORE INTO seen_urls (url, first_seen) VALUES (?, ?)",
            [(u, "migrated") for u in urls],
        )
        conn.commit()
        print(f"Migrated {len(urls):,} seen URLs")
    else:
        print("No _seen_urls.json found, skipping")

    # --- Migrate job logs from zip files and loose JSONs ---
    total_jobs = 0
    for month_dir in sorted(_JOBS_DIR.iterdir()):
        if not month_dir.is_dir():
            continue

        files_to_import: dict[str, str] = {}

        # Collect from zip
        for zp in month_dir.glob("*.zip"):
            if len(zp.stem) != 7:
                continue
            try:
                with zipfile.ZipFile(zp, "r") as zf:
                    for name in zf.namelist():
                        stem = name.removesuffix(".json")
                        if len(stem) == 10:
                            files_to_import[stem] = zf.read(name).decode("utf-8")
            except Exception as e:
                print(f"  Warning: failed to read {zp}: {e}")

        # Collect from loose JSONs (override zip if both exist)
        for jp in month_dir.glob("*.json"):
            if len(jp.stem) == 10:
                files_to_import[jp.stem] = jp.read_text(encoding="utf-8")

        for date_key in sorted(files_to_import):
            try:
                jobs = json.loads(files_to_import[date_key])
                if not isinstance(jobs, list):
                    continue
            except Exception:
                continue

            rows = []
            for job in jobs:
                key = _dedup_key(job)
                if not key:
                    continue
                scraped = job.get("scraped_at", "migrated")
                rows.append((date_key, key, scraped, json.dumps(job, default=str)))

            if rows:
                conn.executemany(
                    "INSERT OR IGNORE INTO jobs (date_key, dedup_key, scraped_at, data) VALUES (?, ?, ?, ?)",
                    rows,
                )
                total_jobs += len(rows)
                print(f"  {date_key}: {len(rows):,} jobs")

        conn.commit()

    print(f"\nDone. Total jobs migrated: {total_jobs:,}")
    print(f"DB size: {(_JOBS_DIR / 'jobs.db').stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
