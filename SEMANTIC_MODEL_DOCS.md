# Semantic Filtering Model (MiniLM)

The Discord job board aggregator uses a local Machine Learning model to filter out irrelevant job postings, boilerplate text, and low-quality matches. This document outlines how the model works and its specific use cases within the application.

## The Model

**Model Used:** `sentence-transformers/all-MiniLM-L6-v2`
**Size:** ~90 MB (Weights)

The `all-MiniLM-L6-v2` is a lightweight, fast, and highly efficient sentence-transformer model that maps sentences and paragraphs to a 384-dimensional dense vector space. It is designed for tasks like clustering or semantic search.

### Why MiniLM?
- **Speed & Efficiency:** It processes short texts (like job titles and descriptions) extremely quickly. On typical hardware, running a batch of 30+ candidates takes less than a second.
- **Resource Constraints:** Since the bot runs multiple background scraping threads, a lightweight model ensures that the CPU and memory aren't entirely consumed by ML inference. The application explicitly bounds inference threads (`SEMANTIC_INFERENCE_THREADS_MAX = 4`) and batches sizes proportionally to host memory so that inference never starves the job scrapers.

## How It Works

1. **Text Extraction:** When a job is scraped, the application extracts key fields (Title, Company, Location, Site, and Description/Snippet).
2. **Normalization:** The `normalize_description_text` function strips out boilerplate, HTML tags, and unnecessary formatting from the job description.
3. **Embedding:** The combined job text is passed through the MiniLM model via the `sentence-transformers` library to generate a semantic embedding (a vector representation of the text's meaning).
4. **Similarity Scoring:** This embedding is compared against the target keywords/search criteria using cosine similarity.
5. **Filtering:** If the similarity score falls below a configured threshold (`SEMANTIC_PLUGIN_THRESHOLD`, default `0.30`), the job is discarded as a mismatch or boilerplate spam.

## Primary Use Cases

* **Removing Boilerplate:** Job boards are notorious for injecting generic boilerplate (e.g., "Equal Opportunity Employer", "About Us", generic recruiter spam) into every post. The model evaluates the actual semantic weight of the listing against the search query, naturally penalizing listings that are mostly boilerplate with zero relevance.
* **Semantic Filtering:** Keyword matching is fragile (e.g., a search for "Software Engineer" might match a "Software Sales" role simply because of the word "Software"). The semantic model understands the *meaning* of the job description, ensuring that a "Python Developer" role scores highly for a "Software Engineer" search, even if the exact keyword isn't perfectly matched.
* **Aggregator Deduplication / Quality Control:** By evaluating the semantic similarity of descriptions, the model acts as a quality gatekeeper before jobs are broadcasted to the Discord channels, ensuring users only see highly relevant, high-signal postings.

## Configuration

You can tweak the model's behavior in `settings.toml`:
- `semantic_enabled`: Toggle the semantic filter on/off.
- `semantic_threshold`: The similarity score required to pass the filter (default: `0.30`).
- `semantic_match_target`: The field to compare against (default: `description`).
- `semantic_description_char_limit`: Truncation limit for descriptions to prevent memory spikes on massive text walls.
