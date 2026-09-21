"""Prompts, prefixes, and anchor texts for the Pure Nomic Semantic Architecture."""
from __future__ import annotations

# ============================================================================
# Model IDs & Defaults
# ============================================================================
MODEL_ID = "nomic-embed-text-v1.5"
HF_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
GGUF_MODEL_FILE = "nomic-embed-text-v1.5.f16.gguf"

# Matryoshka dimension scaling (Nomic natively supports 768 down to 64)
MATRYOSHKA_DIMS: tuple[int, ...] = (768, 512, 384, 256, 128, 64)
MASTER_EMBEDDING_DIM: int = 768
DEFAULT_EMBEDDING_DIM = 384

# ============================================================================
# Nomic Native Task Types
# ============================================================================
TASK_SEARCH_DOC = "search_document"
TASK_SEARCH_QUERY = "search_query"
TASK_CLASSIFICATION = "classification"
TASK_CLUSTERING = "clustering"

# Fallback string prefixes (used when running via raw transformers / sentence-transformers)
PREFIX_SEARCH_DOC = "search_document: "
PREFIX_SEARCH_QUERY = "search_query: "
PREFIX_CLASSIFY = "classification: "
PREFIX_CLUSTER = "clustering: "

# ============================================================================
# Seniority Classification Anchors
# ============================================================================
SENIORITY_LABELS: tuple[str, ...] = ("intern", "newgrad", "junior", "mid", "senior", "staff")

SENIORITY_ANCHORS: dict[str, str] = {
    "intern": "Internship, co-op, student trainee, or apprentice position",
    "newgrad": "Entry-level software engineer, university graduate, or new grad role",
    "junior": "Junior developer or early career engineer with 1-2 years experience",
    "mid": "Mid-level software engineer, developer, or standard professional individual contributor",
    "senior": "Senior software engineer, technical lead, or experienced developer with 5+ years experience",
    "staff": "Staff engineer, principal architect, engineering director, VP, or executive leadership",
}

# ============================================================================
# Boilerplate & Noise Filtering Anchors
# ============================================================================
BOILERPLATE_ANCHOR = (
    "Equal opportunity employer, diversity statement, 401k retirement plans, "
    "health dental insurance, legal disclaimers, salary ranges, benefits blurb"
)
DUTY_ANCHOR = (
    "Core technical software development duties, coding responsibilities, "
    "system architecture, database maintenance, framework design"
)

# ============================================================================
# Technical Skill Extraction Concept Anchors
# ============================================================================
SKILL_ANCHOR = (
    "Core software engineering technical skill, programming language, database, "
    "cloud infrastructure, developer framework, or software library"
)
