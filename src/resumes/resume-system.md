# Resume System Overview

This document explains what the `$resume` system is designed to do, how it works, and where each part lives in the codebase.

## Goal

The resume system lets a user reply to a posted job listing in Discord with `$resume`, then automatically:

1. Parses the replied message for job title and URL.
2. Scrapes the job posting page for role details.
3. Builds Gemini context from resume source files and profile-specific prompt files.
4. Asks Gemini to tailor resume content for the job.
5. Optionally compiles returned LaTeX (inside `<latex>...</latex>`) into a PDF.
6. Sends tailored content and PDF back to Discord.

## High-Level Behavior

`$resume` is intended as a role-targeting assistant for one candidate profile system, with per-profile prompt/template files in `resumes_cache`.

- Access control: only the server owner can run `$resume`.
- Profile model:
  - Owner can use a named profile folder (for example `xboxsignout._`).
  - Other users can get auto-generated profile folders by user ID.
  - Missing profile files are scaffolded automatically.
- Resume source facts used for Gemini explicit cache come from `rebuilt_app/resumes/*.md|*.txt`.
- Prompt guidance/template context comes from profile files in `src/services/resumes/resumes_cache/<profile>/`.

## Command Entry Point

Main command router:

- `src/commands/handlers.py`

`handle_resume` performs:

1. Authorization check:
   - Must run in a guild channel.
   - Caller must match `message.guild.owner_id`.
2. Resolve replied message and parse `JobContext`.
3. Ensure profile folder exists in `resumes_cache`.
4. Ensure Gemini explicit cache exists/reused for resume source files.
5. Generate tailored resume text through Gemini.
6. If LaTeX is present, compile to PDF and upload.

## Data Flow and Components

### 1) Job Context Parsing

File:

- `src/services/resumes/listing.py`

Key behavior:

- Extracts job title and first URL from the replied message.
- Supports optional `Apply:` URL parsing.
- Produces `JobContext`.

### 2) Job Posting Scrape

File:

- `src/services/resumes/listing.py`

Key behavior:

- Fetches job page HTML with `requests`.
- Parses visible text with BeautifulSoup.
- Extracts:
  - Page title
  - Company hints (meta tags)
  - Location hints
  - Highlights from requirement-like lines

### 3) Gemini Explicit Cache for Resume Sources

File:

- `src/services/resumes/cache.py`

Key behavior:

- Reads local source files from `rebuilt_app/resumes/` (`.md`, `.txt`).
- Creates/reuses remote Gemini explicit cache.
- Stores local cache metadata in:
  - `rebuilt_app/.resume_cache/resume_explicit_cache.json`

Purpose:

- Keep stable candidate background context in Gemini cache.
- Avoid rebuilding cache on every request unless content/model/TTL changes.

### 4) Profile Prompt/Template Context

Files:

- `src/services/resumes/resume.py`
- `src/services/resumes/listing.py`

Key behavior:

- Profile folders are under:
  - `src/services/resumes/resumes_cache/<profile_key>/`
- Scaffolds these files when missing:
  - `baseinfo.txt`
  - `instructions.txt`
  - `template.tex`
- Any other files or directories in profile folders are purged.
- Owner profile detection:
  - Uses `MAIN_USER_PROFILE` when set.
  - Otherwise infers when only one folder exists under `resumes_cache`.
- New profiles are seeded from owner/global seed content when available.

### 5) Gemini Generation

File:

- `src/services/resumes/listing.py`

Key behavior:

- Builds one prompt containing:
  - Parsed job metadata
  - Scraped description/highlights
  - Profile base info
  - Profile instructions
  - LaTeX template text
- Calls Gemini model (`google-genai`) with optional `cached_content` from explicit cache.
- Parses output:
  - Returns formatted text/JSON if possible.
  - Extracts rewritten TeX from JSON `rewritten_tex` when present.
  - Falls back to extracting TeX from `<latex>...</latex>`, fenced `tex/latex` blocks, or inline `\documentclass...\end{document}` content.

### 6) LaTeX Compile + PDF Response

File:

- `src/services/resumes/resume.py`

Key behavior:

- Sends extracted rewritten `.tex` to Discord as a reply attachment.
- Checks local LaTeX environment readiness (`pdflatex`, `kpsewhich`, template dependencies).
- Compiles returned LaTeX into PDF in a temp workspace.
- Writes compile log under `.resume_cache/compile_logs/<profile>.log`.
- Returns PDF bytes + filename to Discord handler for upload.

## Authorization Rules

Current `$resume` auth rule:

- Server owner only (`guild.owner_id`).
- Not available in direct messages.

## Folder Layout

Candidate source files (explicit cache inputs):

- `rebuilt_app/resumes/*.md|*.txt`

Gemini explicit cache metadata:

- `rebuilt_app/.resume_cache/resume_explicit_cache.json`

Profile prompt/template files:

- `rebuilt_app/src/services/resumes/resumes_cache/<profile_key>/baseinfo.txt`
- `rebuilt_app/src/services/resumes/resumes_cache/<profile_key>/instructions.txt`
- `rebuilt_app/src/services/resumes/resumes_cache/<profile_key>/template.tex`
- `rebuilt_app/src/services/resumes/resumes_cache/<profile_key>/template.log`

## Environment Variables

Commonly relevant variables:

- `discordtoken`
- `GOOGLE_AI_STUDIO_API_KEY` or `GEMINI_API_KEY` or `geminiAPI`
- `GEMINI_MODEL`
- `GEMINI_RESUME_CACHE_TTL_SECONDS`
- `MAIN_USER_PROFILE` (optional named owner profile folder)

## Operational Prerequisites

Before `$resume` will work in Discord, all of these must be true:

1. Discord Developer Portal:
  - Message Content Intent enabled for this bot application.
2. Bot runtime:
  - Correct token loaded in `rebuilt_app/.env`.
  - Active process is running this app (`python src/app.py` from `rebuilt_app`).
3. Channel/server permissions:
  - `View Channel`
  - `Read Message History`
  - `Send Messages`
  - `Attach Files` (required for PDF upload)
  - `Embed Links` (recommended for job link previews)
4. Resume context files:
  - At least one `.md` or `.txt` source file in `rebuilt_app/resumes/` for explicit cache context.

## Strict Preconditions for `$resume`

The command handler currently expects:

1. Command must run in a guild text channel (not DM).
2. Caller must be server owner (`guild.owner_id`).
3. `$resume` message must be a reply to a job post message.
4. Replied job post message must contain a parseable URL.

## Expected Discord UX

1. User (server owner) replies to a job listing message with `$resume`.
2. Bot posts status summary:
   - model
   - cache status
   - parsed/scraped context
3. Bot posts tailored resume output.
4. If rewritten TeX is extracted, bot uploads the rewritten `.tex` file as a reply attachment.
5. If compile succeeds, bot uploads PDF.
6. If compile fails, bot reports failure and points to compile log context.

## Failure Modes

Typical error paths are handled and reported:

- No reply target message.
- Replied message missing parsable URL/title.
- Gemini API key missing.
- Resume source files missing in `rebuilt_app/resumes/`.
- Job posting scrape failure.
- Gemini generation failure/empty response.
- LaTeX environment not installed or missing packages.
- LaTeX compile error (with log excerpt and profile log path).
- `$resume` sent without replying to a job message (`Reply to a job post message with $resume.`).

## Known Caveats

1. Resume generation can still run even if explicit cache setup reports missing/error states.
  - Intended canonical context is the explicit cache plus local resume files.
  - If cache setup fails, generation may still proceed with reduced grounding.
2. Owner profile inference is startup/path based.
  - If profile folders are renamed/changed at runtime, restart may be required for clean behavior.

## Troubleshooting Runbook

If commands appear in channel but bot does not respond:

1. Confirm process and lock:
  - `rebuilt_app/.bot.pid`
  - `rebuilt_app/.bot.lock`
2. Confirm active process command line points to this app:
  - should run `src/app.py` under `rebuilt_app`.
3. Verify Message Content Intent is enabled in Discord Dev Portal for this bot.
4. Verify channel permission overrides did not block `Send Messages` / `Attach Files`.
5. Run a control command:
  - `$status` should return watcher state summary.
6. Run `$resume` correctly:
  - reply to a watcher job post message that includes a URL.
7. If LaTeX fails:
  - inspect profile `template.log` in `src/services/resumes/resumes_cache/<profile_key>/template.log`.
8. If `$commands` does not list `$resume` or `$resume` appears to do nothing:
  - the active process may be running older code.
  - restart runtime from `rebuilt_app` with `run.bat forcerun`.
  - re-check `$commands` and confirm `$resume` appears in the command list.

## Verification Checklist

Use this quick smoke test after any `$resume` changes:

1. Send `$status` in target channel and verify a status response appears.
2. Reply to a known watcher message containing a valid job URL with `$resume`.
3. Verify bot sends summary lines (`model`, `cache status`, scraped title/company/location).
4. Verify tailored content message is returned.
5. If Gemini output includes `<latex>...</latex>`, verify PDF upload succeeds.
6. Confirm per-profile files exist in `resumes_cache/<profile_key>/` and `template.log` updates after compile attempts.
7. Negative check: send `$resume` without replying to a job post and verify the bot returns `Reply to a job post message with $resume.`.

## What This System Is For

This is not a general resume builder. It is a Discord-driven, profile-aware tailoring pipeline for job-specific adaptation, with optional LaTeX-to-PDF output and explicit Gemini context caching.
