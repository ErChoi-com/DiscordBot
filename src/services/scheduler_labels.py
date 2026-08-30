from __future__ import annotations

"""Centralized cost-tracking labels for PriorityWorkScheduler.

A label is the join key for a task's learned median-duration cost (see
PriorityWorkScheduler.estimated_cost). A typo or an inconsistent literal at
a new call site silently starts a fresh, never-warmed-up cost bucket instead
of erroring -- importing these constants instead of hand-typing strings
turns that into a NameError/AttributeError at import time.

Zero dependencies on purpose: this module is imported from watchers/manager.py,
commands/handlers.py, services/reddit_service.py, and ui/views.py, and must
never become a source of circular imports between them.
"""

# -- Job watcher domain --
JOB_SCRAPE = "job_scrape"
SEMANTIC_FILTER = "semantic_filter"
ATS_SCRAPE = "ats_scrape"
GEONAMES_SYNC = "geonames_sync"
REDDIT_SCRAPE = "reddit_scrape"

# -- Resume commands --
RESUME_ENSURE_PROFILE_CACHE = "resume_ensure_profile_cache"
RESUME_LOAD_STRUCTURED_PROFILE = "resume_load_structured_profile"
RESUME_ENSURE_GEMINI_CACHE = "resume_ensure_gemini_cache"
RESUME_REWRITE = "resume_rewrite"
RESUME_COMPILE_LATEX = "resume_compile_latex"
RESUME_REPAIR_LATEX = "resume_repair_latex"
RESUME_CONDENSE_LATEX = "resume_condense_latex"
RESUME_READ_TEMPLATE = "resume_read_template"
RESUME_TEMPLATE_PREVIEW_COMPILE = "resume_template_preview_compile"

# Cover letters get their own rewrite/compile buckets rather than sharing
# RESUME_REWRITE/RESUME_COMPILE_LATEX: they're typically much shorter
# documents than a full resume, so pooling them would skew both medians.
RESUME_COVER_REWRITE = "resume_cover_rewrite"
RESUME_COVER_COMPILE_LATEX = "resume_cover_compile_latex"

# -- Job-board test/debug commands --
JOB_PIPELINE_TEST_SCRAPE = "job_pipeline_test_scrape"
JOB_PIPELINE_TEST_SEMANTIC_FILTER = "job_pipeline_test_semantic_filter"

# -- Archive ranking --
BEST_JOBS_RANK = "best_jobs_rank"

# -- .scrape command --
SCRAPE_COMMAND_JOBSITE = "scrape_command_jobsite"
SCRAPE_COMMAND_GENERIC = "scrape_command_generic"
SCRAPE_COMMAND_AI_CLEANUP = "scrape_command_ai_cleanup"


def job_scrape_label(channel_id: int) -> str:
    return f"{JOB_SCRAPE}:{channel_id}"


def semantic_filter_label(channel_id: int) -> str:
    return f"{SEMANTIC_FILTER}:{channel_id}"


def ats_scrape_label(platform: str) -> str:
    return f"{ATS_SCRAPE}:{platform}"


def reddit_scrape_label(subreddit: str) -> str:
    return f"{REDDIT_SCRAPE}:{subreddit}"
