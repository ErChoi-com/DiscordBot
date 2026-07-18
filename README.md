# Rebuilt Discord Scraper Bot

Modular rebuild of the original Discord scraper bot.

## Quick Start
1. Install deps:
   - `pip install -U -r requirements.txt`
2. Create `.env` in this folder:
   - `discordtoken=YOUR_DISCORD_BOT_TOKEN`
   - `MAIN_USER_PROFILE=OPTIONAL_OWNER_PROFILE_FOLDER_NAME`
   - `openRouter=OPTIONAL_OPENROUTER_KEY`
   - `GOOGLE_AI_STUDIO_API_KEY=OPTIONAL_GEMINI_KEY`
      - `geminiAPI=OPTIONAL_GEMINI_KEY`
      - `GOOGLE_AI_STUDIO_API_KEY=OPTIONAL_GEMINI_KEY`
   - `GEMINI_MODEL=OPTIONAL_MODEL_NAME` (defaults to `gemini-2.5-flash`)
   - `GEMINI_RESUME_CACHE_TTL_SECONDS=OPTIONAL_SECONDS` (defaults to `86400`)
   - `JOBSPY_PYTHON_EXE=OPTIONAL_PYTHON_PATH`
   - `JOBSPY_PROXIES=OPTIONAL_PROXY_LIST` (comma/semicolon/newline separated)
   - `REDDIT_PROXIES=OPTIONAL_PROXY_LIST` (comma/semicolon/newline separated)
3. Run:
   - `python src/app.py`

## User Commands

Text commands use the `.` prefix; every command also has `/`-style aliases
(source of truth: `_COMMAND_ALIASES` in `src/commands/handlers.py` — run
`.cmd` in Discord for the live cheatsheet).

General:
- `.cmd` (`/commands`) — command cheatsheet
- `.hi` (`/hello`), `.more` (`/continue`) — hello / continue paginated output
- `.scrape <url>` or `.scrape <url> | <css selector>`
- `.cfg` / `.scrapecfg` (`/settings`) — scrape settings panel
- `.st` / `.watch` (`/status`) — watcher status
- `.health` (`.health all` for every channel) — scrape health dashboard

Job watching:
- `.job` / `.jobs` (`/jobsettings`, `/jobsinit`) — job watcher settings
- `.jtest` (`/jobbanktest`) — Job Bank listing test
- `.jobtest` (`/jobpipelinetest`) — full pipeline test with metrics
- `.jfilters [query|clear]` (`/jobbankfilters`) — extra native filters

Reddit watching:
- `.reddit` / `.rset` (`/redditsettings`) — reddit watcher settings
- `.rclear` (`/clearredditseen`), `.rreset` (`/resetredditseen`)

Resume (owner-only, reply to a watcher job post):
- `.resumebuild` (`/resumebuild`, `/res`) — tailored resume PDF; flags:
  `--aggressive`, `--strongaggressive`
- `.resumecoverbuild` (`/coverbuild`, `/cover`) — cover letter built from what
  the resume left out
- `.resumecheck` — compile the current template as a baseline check

## Notes
- Job watching uses JobSpy and now defaults to `All supported sites` for the installed JobSpy version.
- If you set `JOBSPY_PYTHON_EXE`, install the same `python-jobspy` version into that interpreter too.
- Proxy lists rotate automatically per request attempt when `JOBSPY_PROXIES` or `REDDIT_PROXIES` are set.
- For `/resume`, only the server owner can invoke the command.
- For `/resume`, local source files live in `resumes/` as `.md` or `.txt` files.
- Resume prompt/template scaffolding now lives under `src/services/resumes/resumes_cache/<discord_user_id>/` and auto-seeds `baseinfo.txt`, `instructions.txt`, `template.tex`, and `template.log` for each profile.
- If the owner already has a named folder such as `src/services/resumes/resumes_cache/xboxsignout._`, set `MAIN_USER_PROFILE` to that folder name or leave it unset when it is the only seeded profile folder and the bot will infer it.
- If Gemini returns a full LaTeX document inside `<latex>...</latex>`, the bot will compile it and upload the resulting PDF back to Discord.
- Gemini explicit cache is not the same thing as the local resume folder. Local source files are read from `resumes/`, while runtime cache metadata is stored in `.resume_cache/resume_explicit_cache.json` and mapped to a remote Gemini cache object.
- LaTeX resume compilation uses the Python package `pdflatex` as a wrapper only. You still need a real TeX distribution with `pdflatex`, `kpsewhich`, and the template dependencies such as `XCharter`, `geometry`, `enumitem`, `hyperref`, `titlesec`, `comment`, and `glyphtounicode`.
