# Semantic Machine Learning Architecture: Pure Nomic (v1.5)

The Discord job board aggregator and resume matching engine use a unified, local Machine Learning pipeline powered exclusively by **Nomic Embed Text v1.5** (`nomic[local]` / `Embed4All`) to perform high-precision filtering, categorization, deduplication, and matching across scraped postings and candidate resumes.

---

## 1. Architectural Foundation: Pure Nomic Engine

BGE and multi-model fallbacks have been removed in favor of a single, unified Nomic architecture executing locally.

```
                              [Incoming Text / Document / Query]
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │      nomic-embed-text-v1.5.f16.gguf       │
                        │  • 274 MB unquantized full float16        │
                        │  • SHA-256 checksum-verified weights      │
                        │  • 8,192 token context window             │
                        │  • Built-in local runtime (nomic[local])  │
                        └─────────────────────┬─────────────────────┘
                                              │
                                     [Matryoshka Slicing]
                                              │
                                              ▼
                                 Dense 384-d Embedding
                                    (||z||_2 = 1.0)
                                              │
        ┌───────────────────┬─────────────────┴─────────────────┬───────────────────┐
        ▼                   ▼                                   ▼                   ▼
 [Seniority Classifier] [Boilerplate Cleaner]        [Semantic Dedup]      [Resume-to-Job Fit]
  6-Class Classification  Duty vs Noise Anchors       Cosine Sim >= 0.92     Asymmetric Retrieval
  (intern -> staff)       (< 0.1% Content Loss)       (8k Full-Text Context) (Query vs Doc)
```

### Core Specifications
* **Engine / Package**: `nomic[local]` (via `Embed4All` / local GGUF engine).
* **Weights File**: `nomic-embed-text-v1.5.f16.gguf` (274,290,560 bytes) located in `~/.cache/gpt4all/`.
* **Precision**: **Full float16 (unquantized)**. Zero quantization loss.
* **Integrity Guarantee**: Automatic SHA-256 checksum verification ensures model weights can never be corrupted.
* **Context Length**: **8,192 tokens** (handles full multi-page resumes and long ATS job descriptions without truncation).
* **Dimensionality Scaling**: Native Matryoshka representation learning supports reducing the vector size from 768 down to **384** (or 256/128) via `dimensionality=384`. Normalized vectors lie on the unit hypersphere ($\|z\|_2 = 1.0$), ensuring standard cosine distance equals the vector dot product.

---

## 2. Production Capabilities

All semantic use cases execute through `services.semantic.engine.SemanticEngine`:

1. **Seniority & Tier Classification:**
   * Maps job titles to one of 6 canonical levels (`intern`, `newgrad`, `junior`, `mid`, `senior`, `staff`) via cosine similarity against pre-computed tier anchor vectors using `task_type="classification"`.
2. **Channel Routing & Alert Filtering:**
   * Feeds the predicted seniority tier into channel role gates (e.g., `#internships`, `#entry`, `#junior`, `#senior`) with regex safety guards (`_PROGRAM_ROLE`, term dates) to protect feeds.
3. **Cross-Board Semantic Deduplication:**
   * Embeds complete job listings using `task_type="search_document"` over the full 8k context (zero 400-char clipping). Cosine similarity $\ge 0.92$ merges cross-board duplicates without false collisions between distinct roles.
4. **Resume-to-Job Match Scoring:**
   * Computes asymmetric similarity between the candidate's resume (`task_type="search_query"`) and the job description (`task_type="search_document"`).
5. **Skill & Keyword Extraction:**
   * Scores candidate phrases and technical terms against semantic anchor concepts (`"software engineering technical skill, programming language, database..."`).
6. **Scraper Noise & Boilerplate Rejection:**
   * Compares description paragraphs against boilerplate anchors (EEO, 401k, legal disclaimers) vs. core technical duty anchors. Boilerplate chunks are filtered while maintaining $< 0.10\%$ content loss on technical responsibilities.

---

## 3. Configuration & Verification

Configured in `settings.toml` under `[semantic]`:
* `semantic_enabled`: Global toggle for semantic filtering.
* `semantic_threshold`: Cosine similarity threshold required to pass search relevance filters (default: `0.30`).
* `semantic_dedup_threshold`: Cosine similarity required to flag duplicate listings (default: `0.92`).
* `semantic_dim`: Embedding dimension (default: `384` via Matryoshka slicing).

### Running Tests
```powershell
# Run unit tests for the pure Nomic engine
.venv\Scripts\python.exe -m pytest tests/test_semantic_engine.py -v

# Run the complete 7-component model pipeline test
.venv\Scripts\python.exe tests/test_all_model_pipeline.py

# Run all semantic test suites
.venv\Scripts\python.exe -m pytest tests/test_semantic_engine.py tests/test_posting_classifier.py tests/test_job_level.py tests/test_semantic_dedup.py
```
