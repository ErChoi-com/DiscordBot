from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

LATEX_WRAPPER_PACKAGE = "pdflatex"
RESUMES_CACHE_ROOT = Path(__file__).resolve().parent / "resumes_cache"
TEMPLATE_PATH = RESUMES_CACHE_ROOT / "template.tex"
PROFILE_ALLOWED_FILE_NAMES = ("baseinfo.txt", "instructions.txt", "template.tex")
EXAMPLE_PROFILE_KEY = "example"
LLM_PROVIDER_SWITCH_ORDER = ("gemini", "gemini-flash", "groq", "openrouter")
MAX_PROFILE_NAME_PART_LEN = 80
MAX_PROFILE_ID_PART_LEN = 80


def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
	raw = os.getenv(name)
	if raw is None:
		return default
	try:
		value = int(raw)
	except ValueError:
		return default
	return max(minimum, min(value, maximum))


# Keep per-engine compile timeout high enough for complex templates while
# preventing individual runs from stalling too long.
LATEX_ENGINE_TIMEOUT_SECONDS = _bounded_env_int(
	"LATEX_ENGINE_TIMEOUT_SECONDS",
	75,
	minimum=30,
	maximum=180,
)

# Keep total compile budget generous for engine fallback + rerun pass,
# but bounded so command handling remains responsive.
LATEX_TOTAL_TIMEOUT_SECONDS = _bounded_env_int(
	"LATEX_TOTAL_TIMEOUT_SECONDS",
	180,
	minimum=60,
	maximum=420,
)

LATEX_AUTOFIX_MAX_RETRIES = _bounded_env_int(
	"LATEX_AUTOFIX_MAX_RETRIES",
	2,
	minimum=0,
	maximum=4,
)


@dataclass(frozen=True, slots=True)
class LLMProviderCapabilities:
	"""Immutable constraints and feature flags for a resume generation provider.

	These are design-time constants that describe what a provider supports
	and what its hard limits are. Callers must respect them before dispatch.
	"""

	# True only for Gemini — OpenRouter and Groq do not support the
	# cachedContents pre-upload mechanism.  Passing a cache_name to a
	# provider that lacks this support must be suppressed.
	supports_context_cache: bool

	# Upper bound on prompt characters before the provider rejects or
	# throttles the request.
	# - Groq free tier: ~6 000 tokens/min and 500 000 tokens/day; a resume
	#   prompt with template + job description can easily exceed one minute's
	#   quota in a single call, so we keep it well under that ceiling.
	# - OpenRouter: depends on the routed model; 120 000 chars is conservative
	#   for gpt-4o-mini (128 k token context window).
	# - Gemini 2.5 Flash: 1 M token context window; 800 000 chars is safe.
	max_prompt_chars: int

	# Per-request timeout in seconds.  Groq responds fast on small inputs;
	# Gemini may take longer on large cached contexts.
	request_timeout_seconds: int


# The authoritative capability registry — one entry per named provider in
# LLM_PROVIDER_SWITCH_ORDER.  Keep this in sync whenever a new provider is
# added to that tuple.
PROVIDER_CAPABILITIES: dict[str, LLMProviderCapabilities] = {
	"gemini": LLMProviderCapabilities(
		supports_context_cache=True,
		max_prompt_chars=800_000,
		request_timeout_seconds=60,
	),
	"gemini-flash": LLMProviderCapabilities(
		supports_context_cache=False,
		max_prompt_chars=800_000,
		request_timeout_seconds=60,
	),
	"openrouter": LLMProviderCapabilities(
		supports_context_cache=False,
		max_prompt_chars=120_000,
		request_timeout_seconds=45,
	),
	"groq": LLMProviderCapabilities(
		supports_context_cache=False,
		max_prompt_chars=24_000,
		request_timeout_seconds=30,
	),
}


DOCUMENTCLASS_PATTERN = re.compile(r"\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}")
USEPACKAGE_PATTERN = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}")
INPUT_PATTERN = re.compile(r"\\input\{([^}]+)\}")
LATEX_TAG_PATTERN = re.compile(r"<latex>(.*?)</latex>", re.IGNORECASE | re.DOTALL)
LATEX_FENCE_PATTERN = re.compile(r"```(?:latex|tex)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
LATEX_DOCUMENT_PATTERN = re.compile(r"(\\documentclass[\s\S]*?\\end\{document\})", re.IGNORECASE)
UNESCAPED_AMPERSAND_PATTERN = re.compile(r"(?<!\\)&")
PROFILE_KEY_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
PROFILE_ID_SUFFIX_PATTERN = re.compile(r"^(?P<prefix>.+)-(?P<id>\d+)$")
XELATEX_HINT_PATTERN = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*fontspec[^}]*\}|\\setmainfont\{|\\newfontfamily\{", re.IGNORECASE)
LUALATEX_HINT_PATTERN = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*unicode-math[^}]*\}|\\directlua\{|\\begin\{luacode\}", re.IGNORECASE)
LATEX_RERUN_HINT_PATTERN = re.compile(r"Rerun to get cross-references right|Label\(s\) may have changed", re.IGNORECASE)
LATEX_ALIGNMENT_ERROR_PATTERN = re.compile(r"Misplaced alignment tab character\s*&", re.IGNORECASE)
LATEX_BRACE_ERROR_PATTERN = re.compile(
	r"Runaway argument\?|File ended while scanning use of|Missing \} inserted|Too many \}|Paragraph ended before",
	re.IGNORECASE,
)
LATEX_MISSING_STYLE_PATTERN = re.compile(r"File\s+`([^`]+\.sty)'\s+not\s+found|File\s+([^\s]+\.sty)\s+not\s+found", re.IGNORECASE)
GLYPHTOUNICODE_INPUT_PATTERN = re.compile(r"^\s*\\input\{glyphtounicode\}\s*$", re.IGNORECASE)
PDF_GLYPH_UNICODE_PATTERN = re.compile(r"^\s*\\pdfglyphtounicode\b.*$", re.IGNORECASE)
PDF_GENTOUNICODE_PATTERN = re.compile(r"^\s*\\pdfgentounicode\b.*$", re.IGNORECASE)
# Repair cases like "\textbf{Honours and Scholarship\section..." while
# preserving valid style openings such as "\textbf{\scshape Name}".
INLINE_TEXT_COMMAND_RUNAWAY_PATTERN = re.compile(
	r"(\\text(?:bf|it|sc|tt|sf|md)\{[^\\}\n][^}\n]*)(\\[A-Za-z@]+)"
)
EMPTY_TEXTCMD_SMALL_ARG_PATTERN = re.compile(r"(\\text(?:bf|it))\{\}\s*\\small\s*([^}\n]+)\}")
EMPTY_TEXTCMD_STYLE_ARG_PATTERN = re.compile(
	r"(\\text(?:bf|it))\{\}\s*\\(scshape|itshape|bfseries)\s*([^\n]+?)\}\}+"
)
BROKEN_ITEM_SECTION_PATTERN = re.compile(
	r"(?m)^(\s*\\item\s+\\small\{\\textbf\{[^\n]+?)\}?\s*\\section\{([^}\n]+)\}"
)
SMALL_ITEMIZE_BLOCK_PATTERN = re.compile(
	r"(\{\s*\\small\s*\\begin\{itemize\}(?:\[[^\]]*\])?[\s\S]*?\\end\{itemize\})(?!\s*\})",
	re.DOTALL,
)
ORPHAN_LOWER_FRAGMENT_PATTERN = re.compile(r"^[a-z]{2,12}$")
DOUBLE_SLASH_CONTROL_LINE_PATTERN = re.compile(
	r"(?m)^(?P<indent>\s*)\\\\(?P<cmd>resumeItemListStart|resumeItemListEnd|resumeSubHeadingListStart|resumeSubHeadingListEnd|resumeSubheading|resumeItem|section|vspace)\b"
)
BRACED_RESUME_END_MACRO_PATTERN = re.compile(
	r"\{\s*\\(?P<cmd>resumeItemListEnd|resumeSubHeadingListEnd)\s*\}"
)
CHKTEX_LINE_PATTERN = re.compile(r"^(?P<line>\d+):(?P<col>\d+):(?P<kind>[^:]*):(?P<code>\d+):(?P<message>.*)$")
MAX_CHKTEX_FINDINGS = 12
COLOR_DVIPS_OPTION_CASE_PATTERN = re.compile(
	r"(?P<prefix>\\usepackage(?:\[[^\]]*?)?)dvipsNames(?P<suffix>[^\]]*\]\{color\})"
)


@dataclass(slots=True)
class LatexCompileEnvironment:
	wrapper_package: str
	wrapper_installed: bool
	pdflatex_path: str | None
	kpsewhich_path: str | None
	xelatex_path: str | None = None
	lualatex_path: str | None = None
	required_files: list[str] = field(default_factory=list)
	missing_files: list[str] = field(default_factory=list)

	@property
	def ready(self) -> bool:
		return bool(
			(self.pdflatex_path or self.xelatex_path or self.lualatex_path)
			and not self.missing_files
		)


@dataclass(slots=True)
class LatexCompileResult:
	status: str
	message: str
	pdf_bytes: bytes | None = None
	pdf_name: str | None = None
	log_excerpt: str | None = None
	log_path: Path | None = None
	repairs_applied: list[str] = field(default_factory=list)
	repair_attempts: int = 0
	lint_findings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LatexLintResult:
	latex_document: str
	repairs_applied: list[str] = field(default_factory=list)
	findings: list[str] = field(default_factory=list)


def profile_cache_dir(user_id: int | str, cache_root: Path = RESUMES_CACHE_ROOT) -> Path:
	return cache_root / str(user_id)


def _sanitize_profile_key_component(value: str | None) -> str:
	text = (value or "").strip()
	if not text:
		return "user"
	# Keep the full username shape (including leading/trailing . and _) while
	# replacing filesystem-unfriendly characters with '-'.
	sanitized = PROFILE_KEY_SANITIZE_PATTERN.sub("-", text)
	if not sanitized:
		return "user"
	return sanitized[:MAX_PROFILE_NAME_PART_LEN]


def _sanitize_profile_id_component(value: int | str) -> str:
	text = str(value).strip()
	if not text:
		return "0"
	sanitized = PROFILE_KEY_SANITIZE_PATTERN.sub("-", text).strip("-._")
	if not sanitized:
		return "0"
	return sanitized[:MAX_PROFILE_ID_PART_LEN]


def discord_profile_key(user_id: int | str, username: str | None) -> str:
	name_part = _sanitize_profile_key_component(username).lower()
	return name_part


def _legacy_name_from_id_suffixed_key(key: str, user_id: int | str) -> str | None:
	id_suffix = f"-{_sanitize_profile_id_component(user_id)}"
	if not key.endswith(id_suffix):
		return None
	base = key[: -len(id_suffix)].strip("-._")
	if not base:
		return None
	return _sanitize_profile_key_component(base).lower()


def _existing_profile_dirs_for_id(user_id: int | str, cache_root: Path) -> list[Path]:
	if not cache_root.exists() or not cache_root.is_dir():
		return []
	id_suffix = f"-{_sanitize_profile_id_component(user_id)}"
	return [
		path
		for path in sorted(cache_root.iterdir())
		if path.is_dir() and path.name.endswith(id_suffix)
	]


def resolve_discord_profile_key(
	user_id: int | str,
	username: str | None,
	cache_root: Path = RESUMES_CACHE_ROOT,
) -> str:
	preferred_key = discord_profile_key(user_id, username)
	preferred_dir = profile_cache_dir(preferred_key, cache_root)
	# Enforce: only allow username-based key, never fallback to legacy/ID-based/other keys.
	if preferred_dir.exists() and preferred_dir.is_dir():
		return preferred_key
	# If the username-based folder does not exist, fail explicitly.
	raise FileNotFoundError(
		f"Resume cache for username '{preferred_key}' does not exist. "
		f"Expected at: {preferred_dir}. "
		"No fallback to legacy or ID-based cache is allowed. "
		"Please create the folder and required files."
	)


def _extract_user_id_from_legacy_profile_key(profile_key: str) -> int | None:
	key = profile_key.strip()
	if not key:
		return None
	if key.isdigit():
		try:
			return int(key)
		except ValueError:
			return None

	match = PROFILE_ID_SUFFIX_PATTERN.fullmatch(key)
	if not match:
		return None
	try:
		return int(match.group("id"))
	except ValueError:
		return None


def migrate_legacy_profile_keys_with_usernames(
	username_by_user_id: dict[int, str],
	cache_root: Path = RESUMES_CACHE_ROOT,
) -> list[tuple[str, str]]:
	"""Rename legacy ID-based profile folders to username-only keys.

	This only migrates directories that encode a numeric Discord ID either as
	a pure numeric folder name (for example ``12345``) or with an ID suffix
	(for example ``old-name-12345``). The destination folder is always derived
	from the user's unique Discord username (``user.name``), sanitized through
	``discord_profile_key``.
	"""
	if not cache_root.exists() or not cache_root.is_dir():
		return []

	migrations: list[tuple[str, str]] = []
	for path in sorted(cache_root.iterdir()):
		if not path.is_dir():
			continue
		if path.name == EXAMPLE_PROFILE_KEY:
			continue

		user_id = _extract_user_id_from_legacy_profile_key(path.name)
		if user_id is None:
			continue

		username = username_by_user_id.get(user_id)
		if not isinstance(username, str) or not username.strip():
			continue

		target_key = discord_profile_key(user_id, username)
		if target_key == path.name:
			continue

		target_path = profile_cache_dir(target_key, cache_root)
		if target_path.exists():
			# Avoid destructive merges; keep existing and skip this rename.
			continue

		try:
			path.rename(target_path)
		except OSError:
			continue

		migrations.append((path.name, target_key))

	return migrations


def profile_cache_seed_ready(profile_dir: Path) -> bool:
	return all((profile_dir / file_name).exists() for file_name in ("baseinfo.txt", "instructions.txt", "template.tex"))


def infer_owner_profile_key(cache_root: Path = RESUMES_CACHE_ROOT) -> str | None:
	if not cache_root.exists():
		return None
	all_profile_dirs = [
		path
		for path in sorted(cache_root.iterdir())
		if path.is_dir() and path.name != EXAMPLE_PROFILE_KEY
	]
	seeded_profile_dirs = [path for path in all_profile_dirs if profile_cache_seed_ready(path)]
	if len(seeded_profile_dirs) == 1:
		return seeded_profile_dirs[0].name
	if len(all_profile_dirs) == 1:
		return all_profile_dirs[0].name
	return None


def resolve_profile_seed_dir(cache_root: Path = RESUMES_CACHE_ROOT, seed_profile_key: int | str | None = None) -> Path:
	if seed_profile_key is not None:
		candidate = profile_cache_dir(seed_profile_key, cache_root)
		if profile_cache_seed_ready(candidate):
			return candidate

	example_dir = profile_cache_dir(EXAMPLE_PROFILE_KEY, cache_root)
	if profile_cache_seed_ready(example_dir):
		return example_dir

	for file_name in ("baseinfo.txt", "instructions.txt", "template.tex"):
		if not (cache_root / file_name).exists():
			break
	else:
		return cache_root

	inferred = infer_owner_profile_key(cache_root)
	if inferred is not None:
		return profile_cache_dir(inferred, cache_root)

	return cache_root


def ensure_profile_cache(
	user_id: int | str,
	cache_root: Path = RESUMES_CACHE_ROOT,
	seed_profile_key: int | str | None = None,
) -> Path:
	cache_root.mkdir(parents=True, exist_ok=True)
	example_dir = profile_cache_dir(EXAMPLE_PROFILE_KEY, cache_root)
	missing_example_files = [
		file_name
		for file_name in PROFILE_ALLOWED_FILE_NAMES
		if not (example_dir / file_name).exists()
	]
	if missing_example_files:
		missing_display = ", ".join(missing_example_files)
		raise FileNotFoundError(
			f"Missing example seed files in '{example_dir}': {missing_display}"
		)

	profile_dir = profile_cache_dir(user_id, cache_root)
	profile_dir.mkdir(parents=True, exist_ok=True)
	_purge_unexpected_profile_entries(profile_dir)


	for file_name in PROFILE_ALLOWED_FILE_NAMES:
		target_path = profile_dir / file_name
		if target_path.exists():
			continue
		seed_path = example_dir / file_name
		# Copy seed files as raw bytes so profile templates stay byte-identical to example.
		shutil.copyfile(seed_path, target_path)

	_purge_unexpected_profile_entries(profile_dir)

	return profile_dir


def _purge_unexpected_profile_entries(profile_dir: Path) -> None:
	allowed = set(PROFILE_ALLOWED_FILE_NAMES)
	for entry in profile_dir.iterdir():
		if entry.is_file() and entry.name in allowed:
			continue
		try:
			if entry.is_dir():
				shutil.rmtree(entry)
			else:
				entry.unlink()
		except OSError:
			pass


def read_resume_template(template_path: Path = TEMPLATE_PATH) -> str:
	return template_path.read_text(encoding="utf-8")


def extract_template_requirements(template_text: str) -> list[str]:
	required: list[str] = []

	document_class = DOCUMENTCLASS_PATTERN.search(template_text)
	if document_class:
		required.append(f"{document_class.group(1).strip()}.cls")

	for match in USEPACKAGE_PATTERN.finditer(template_text):
		for package_name in match.group(1).split(","):
			name = package_name.strip()
			if name:
				required.append(f"{name}.sty")

	for match in INPUT_PATTERN.finditer(template_text):
		name = match.group(1).strip()
		if not name:
			continue
		required.append(name if Path(name).suffix else f"{name}.tex")

	return sorted(dict.fromkeys(required))


def recommended_latex_python_package() -> str:
	return LATEX_WRAPPER_PACKAGE


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
	try:
		return subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
	except subprocess.TimeoutExpired as exc:
		return subprocess.CompletedProcess(
			command,
			124,
			exc.stdout or "",
			exc.stderr or f"Command timed out after {exc.timeout} seconds.",
		)
	except OSError as exc:
		return subprocess.CompletedProcess(command, 1, "", str(exc))


def _find_latex_executable(command_name: str) -> str | None:
	resolved = shutil.which(command_name)
	if resolved:
		return resolved

	# Windows installs (notably MiKTeX) are often available but not on PATH.
	if os.name != "nt":
		return None

	exe_name = command_name if command_name.lower().endswith(".exe") else f"{command_name}.exe"
	local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
	program_files = Path(os.environ.get("ProgramFiles", ""))
	program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", ""))

	candidate_dirs = [
		local_app_data / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64",
		local_app_data / "MiKTeX" / "miktex" / "bin" / "x64",
		program_files / "MiKTeX" / "miktex" / "bin" / "x64",
		program_files_x86 / "MiKTeX" / "miktex" / "bin" / "x64",
	]

	for directory in candidate_dirs:
		candidate = directory / exe_name
		if candidate.exists() and candidate.is_file():
			return str(candidate)

	return None


def check_template_compile_environment(
	template_path: Path = TEMPLATE_PATH,
	command_runner: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_command,
) -> LatexCompileEnvironment:
	template_text = read_resume_template(template_path)
	required_files = extract_template_requirements(template_text)
	wrapper_installed = importlib.util.find_spec(LATEX_WRAPPER_PACKAGE) is not None
	pdflatex_path = _find_latex_executable("pdflatex")
	kpsewhich_path = _find_latex_executable("kpsewhich")
	xelatex_path = _find_latex_executable("xelatex")
	lualatex_path = _find_latex_executable("lualatex")

	missing_files: list[str] = []
	if kpsewhich_path:
		for required_file in required_files:
			try:
				result = command_runner([kpsewhich_path, required_file])
			except (subprocess.TimeoutExpired, OSError):
				# Fail open when package probing is unstable so compilation can still be attempted.
				missing_files = []
				kpsewhich_path = None
				break
			if result.returncode == 124:
				missing_files = []
				kpsewhich_path = None
				break
			if result.returncode != 0 or not result.stdout.strip():
				missing_files.append(required_file)

	return LatexCompileEnvironment(
		wrapper_package=LATEX_WRAPPER_PACKAGE,
		wrapper_installed=wrapper_installed,
		pdflatex_path=pdflatex_path,
		kpsewhich_path=kpsewhich_path,
		xelatex_path=xelatex_path,
		lualatex_path=lualatex_path,
		required_files=required_files,
		missing_files=missing_files,
	)


def _preferred_latex_engines(template_text: str) -> list[str]:
	if XELATEX_HINT_PATTERN.search(template_text):
		return ["xelatex", "lualatex", "pdflatex"]
	if LUALATEX_HINT_PATTERN.search(template_text):
		return ["lualatex", "xelatex", "pdflatex"]
	return ["pdflatex", "xelatex", "lualatex"]


def _available_latex_engines(environment: LatexCompileEnvironment, template_text: str) -> list[tuple[str, str]]:
	engine_paths: dict[str, str | None] = {
		"pdflatex": getattr(environment, "pdflatex_path", None),
		"xelatex": getattr(environment, "xelatex_path", None),
		"lualatex": getattr(environment, "lualatex_path", None),
	}
	ordered_names = _preferred_latex_engines(template_text)
	engines: list[tuple[str, str]] = []
	for engine_name in ordered_names:
		engine_path = engine_paths.get(engine_name)
		if engine_path:
			engines.append((engine_name, engine_path))
	return engines


def _latex_needs_rerun(stdout: str, stderr: str) -> bool:
	combined = f"{stdout}\n{stderr}"
	return bool(LATEX_RERUN_HINT_PATTERN.search(combined))


def extract_latex_document(text: str) -> str | None:
	# 1. <latex>...</latex> tags
	match = LATEX_TAG_PATTERN.search(text)
	if match:
		latex_text = match.group(1).strip()
		# Validate that the tag content is actually a LaTeX document, not a
		# placeholder or truncated response (e.g. "..." or plain text).
		if latex_text and "\\documentclass" in latex_text and "\\end{document}" in latex_text:
			return latex_text
		# Fall through if tag content is not a valid LaTeX document.

	# 2. Code fences (```latex, ```tex, ``` ```, or ```json wrapping a JSON payload)
	for fence in LATEX_FENCE_PATTERN.finditer(text):
		raw_content = fence.group(1)
		candidate = raw_content.strip()

		# Strip an optional language-identifier line added by the LLM
		# (e.g. "json", "latex", "tex") that the fence pattern includes in
		# group(1) when the identifier is not one of the expected latex/tex values.
		nl_pos = candidate.find("\n")
		if nl_pos != -1:
			first_line = candidate[:nl_pos].strip().lower()
			if first_line in ("json", "latex", "tex", ""):
				candidate = candidate[nl_pos + 1:].strip()

		# If the fence content is a JSON object, extract the LaTeX document from
		# the appropriate field rather than returning the raw JSON blob (which
		# only incidentally contains \documentclass as a string value).
		if candidate.startswith("{"):
			try:
				parsed_fence = json.loads(candidate)
				if isinstance(parsed_fence, dict):
					for key in ("rewritten_tex", "latex", "latex_document"):
						val = parsed_fence.get(key)
						if isinstance(val, str) and "\\documentclass" in val:
							return val.strip()
			except json.JSONDecodeError:
				pass
			# JSON parse failed; fall back to LATEX_DOCUMENT_PATTERN inside fence
			inner = LATEX_DOCUMENT_PATTERN.search(candidate)
			if inner:
				extracted = inner.group(1).strip()
				if extracted:
					return extracted
		elif "\\documentclass" in candidate and "\\end{document}" in candidate:
			return candidate

	# 3. Direct LaTeX document anywhere in the full text
	inline_match = LATEX_DOCUMENT_PATTERN.search(text)
	if inline_match:
		candidate = inline_match.group(1).strip()
		return candidate or None

	return None


def sanitize_pdf_stem(text: str) -> str:
	cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
	return cleaned[:80] or "tailored-resume"


def _normalize_escaped_latex_document(latex_document: str) -> str:
	"""Normalize JSON-escaped LaTeX payloads into compilable TeX.

	Handles both fully-escaped documents (\\documentclass...) and mixed
	escaped/unescaped payloads where some embedded sequences are escaped.
	"""
	if not latex_document:
		return latex_document

	probe = latex_document.strip()
	# Check for fully-escaped document markers
	looks_fully_escaped = (
		probe.startswith("\\\\documentclass")
		or "\\\\begin{document}" in probe
		or "\\\\end{document}" in probe
	)
	
	# Check for embedded escaped sequences in content (e.g., URL...\n}).
	# Require a non-whitespace predecessor so macro-like usages such as " \n"
	# are less likely to be treated as JSON escape artifacts.
	has_escaped_sequences = bool(re.search(r'(?<=\S)\\[ntr](?![a-zA-Z@])', latex_document))

	if not looks_fully_escaped and not has_escaped_sequences:
		return latex_document

	# Protect four-consecutive-backslash sequences (encoded LaTeX \\ line-break)
	# BEFORE the command-leader marker step so that \\\\\n is not mis-read as
	# the start of a \n control sequence.  Four backslashes in a fully-escaped
	# document always represent exactly one LaTeX \\ line-break.
	line_break_marker = "\uE001"
	normalized = latex_document.replace("\\\\\\\\", line_break_marker)

	# Protect double-escaped TeX command leaders (e.g. ``\\newif``) so
	# global ``\\n`` decoding does not turn them into accidental newlines.
	marker = "\uE000"
	normalized = re.sub(r"\\\\(?=[A-Za-z@])", marker, normalized)

	# Decode JSON-style escaped control characters.
	normalized = normalized.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")

	# Collapse remaining double backslashes (line breaks, escaped literals).
	normalized = normalized.replace("\\\\", "\\")
	# Restore markers: command leaders first, then line-breaks (which expand to \\).
	normalized = normalized.replace(marker, "\\")
	return normalized.replace(line_break_marker, "\\\\")


def _close_unbalanced_latex_braces(latex_document: str) -> str:
	"""Append missing closing braces for lightly malformed model/template output."""
	balance = 0
	escaped = False
	for char in latex_document:
		if escaped:
			escaped = False
			continue
		if char == "\\":
			escaped = True
			continue
		if char == "{":
			balance += 1
		elif char == "}" and balance > 0:
			balance -= 1
	if balance <= 0:
		return latex_document
	return latex_document + ("}" * balance)


def _close_runaway_inline_text_commands(latex_document: str) -> str:
	"""Close inline text-format commands when another macro starts before closing brace."""
	if "\\text" not in latex_document:
		return latex_document

	updated = latex_document
	while True:
		next_text, count = INLINE_TEXT_COMMAND_RUNAWAY_PATTERN.subn(r"\1}\2", updated)
		if count == 0:
			break
		updated = next_text
	return updated


def _normalize_empty_text_command_small_args(latex_document: str) -> str:
	"""Repair malformed patterns like \\textbf{}\\small #2} into \\textbf{\\small #2}."""
	if "\\text" not in latex_document:
		return latex_document
	updated = EMPTY_TEXTCMD_SMALL_ARG_PATTERN.sub(r"\1{\\small \2}", latex_document)
	updated = EMPTY_TEXTCMD_STYLE_ARG_PATTERN.sub(r"\1{\\\2 \3}", updated)
	return updated


def _normalize_broken_item_section_collisions(latex_document: str) -> str:
	"""Close list scopes when a malformed bullet line runs directly into a section."""
	if "\\item" not in latex_document or "\\section{" not in latex_document:
		return latex_document

	def _repl(match: re.Match[str]) -> str:
		item_prefix = match.group(1)
		section_name = match.group(2)
		closing = "}" if item_prefix.rstrip().endswith("}") else "}}"
		return (
			f"{item_prefix}{closing}"
			"\n  \\resumeItemListEnd"
			"\n  \\resumeSubHeadingListEnd"
			f"\n\\section{{{section_name}}}"
		)

	return BROKEN_ITEM_SECTION_PATTERN.sub(_repl, latex_document)


def _normalize_unclosed_resume_lists(latex_document: str) -> str:
	"""Close open resume list helper macros before new sections or document end."""
	if "\\resumeSubHeadingListStart" not in latex_document and "\\resumeItemListStart" not in latex_document:
		return latex_document

	lines = latex_document.splitlines()
	sanitized: list[str] = []
	subheading_depth = 0
	item_depth = 0

	def _flush_open_lists() -> None:
		nonlocal subheading_depth, item_depth
		while item_depth > 0:
			sanitized.append("  \\resumeItemListEnd")
			item_depth -= 1
		while subheading_depth > 0:
			sanitized.append("  \\resumeSubHeadingListEnd")
			subheading_depth -= 1

	for line in lines:
		stripped = line.strip()
		if stripped.startswith("\\section{") or stripped.startswith("\\end{document}"):
			_flush_open_lists()
		if stripped.startswith("\\resumeSubheading") and subheading_depth == 0:
			sanitized.append("  \\resumeSubHeadingListStart")
			subheading_depth += 1

		sanitized.append(line)

		subheading_depth += stripped.count("\\resumeSubHeadingListStart")
		subheading_depth -= min(subheading_depth, stripped.count("\\resumeSubHeadingListEnd"))
		item_depth += stripped.count("\\resumeItemListStart")
		item_depth -= min(item_depth, stripped.count("\\resumeItemListEnd"))

	return "\n".join(sanitized)


def _close_small_itemize_group_brace(latex_document: str) -> str:
	"""Close malformed `{\\small \\begin{itemize} ... \\end{itemize}` groups."""
	if "\\begin{itemize}" not in latex_document or "\\small" not in latex_document:
		return latex_document
	return SMALL_ITEMIZE_BLOCK_PATTERN.sub(r"\1}", latex_document)


def _drop_orphan_lowercase_fragments(latex_document: str) -> str:
	"""Remove obvious stray text fragments introduced by malformed edits.

	Conservative rule: only drop a single lowercase word (2-12 chars) when
	it is between two non-empty LaTeX command lines.
	"""
	if not latex_document:
		return latex_document

	lines = latex_document.splitlines()
	if len(lines) < 3:
		return latex_document

	def _neighbor_is_command(index: int, step: int) -> bool:
		cursor = index + step
		while 0 <= cursor < len(lines):
			neighbor = lines[cursor].strip()
			if neighbor:
				return neighbor.startswith("\\")
			cursor += step
		return False

	kept: list[str] = []
	for i, line in enumerate(lines):
		stripped = line.strip()
		if ORPHAN_LOWER_FRAGMENT_PATTERN.fullmatch(stripped):
			if _neighbor_is_command(i, -1) and _neighbor_is_command(i, 1):
				continue
		kept.append(line)

	return "\n".join(kept)


def _normalize_double_slash_control_lines(latex_document: str) -> str:
	"""Normalize accidental escaped control commands at line starts.

	Conservative scope: only known resume/template commands and only at line start,
	so explicit line-break `\\` usage inside content is preserved.
	"""
	if "\\\\" not in latex_document:
		return latex_document
	return DOUBLE_SLASH_CONTROL_LINE_PATTERN.sub(r"\g<indent>\\\g<cmd>", latex_document)


def _normalize_braced_resume_end_macros(latex_document: str) -> str:
	"""Unwrap malformed forms like `{\resumeItemListEnd}` into `\resumeItemListEnd`."""
	if "resumeItemListEnd" not in latex_document and "resumeSubHeadingListEnd" not in latex_document:
		return latex_document
	return BRACED_RESUME_END_MACRO_PATTERN.sub(r"\\\g<cmd>", latex_document)


def _normalize_color_package_option_case(latex_document: str) -> str:
	"""Normalize known case-sensitive color package options emitted by model output."""
	if "dvipsNames" not in latex_document or "\\usepackage" not in latex_document:
		return latex_document
	return COLOR_DVIPS_OPTION_CASE_PATTERN.sub(r"\g<prefix>dvipsnames\g<suffix>", latex_document)


def _remove_dangling_single_line_backslashes(latex_document: str) -> str:
	"""Remove accidental single trailing backslashes at end-of-line.

	Conservative behavior: preserves intentional LaTeX line breaks (``\\``)
	and skips comment lines.
	"""
	if "\\" not in latex_document:
		return latex_document

	updated_lines: list[str] = []
	for line in latex_document.splitlines():
		trimmed = line.rstrip()
		if trimmed.endswith("\\") and not trimmed.endswith("\\\\") and not trimmed.lstrip().startswith("%"):
			trimmed = trimmed[:-1]
		updated_lines.append(trimmed)

	return "\n".join(updated_lines)


def lint_latex_document_for_compile(latex_document: str) -> LatexLintResult:
	"""Apply high-confidence LaTeX lint repairs and report structural findings.

	Repairs are intentionally narrow to avoid over-correcting valid templates.
	"""
	if not latex_document:
		return LatexLintResult(latex_document=latex_document)

	updated = latex_document
	repairs: list[str] = []
	findings: list[str] = []

	next_text = _normalize_double_slash_control_lines(updated)
	if next_text != updated:
		repairs.append("normalized escaped control commands")
		updated = next_text

	next_text = _normalize_braced_resume_end_macros(updated)
	if next_text != updated:
		repairs.append("unwrapped braced resume list end macros")
		updated = next_text

	next_text = _normalize_color_package_option_case(updated)
	if next_text != updated:
		repairs.append("normalized color package option casing")
		updated = next_text

	next_text = _remove_dangling_single_line_backslashes(updated)
	if next_text != updated:
		repairs.append("removed dangling single trailing backslashes")
		updated = next_text

	if "\\documentclass" not in updated:
		findings.append("missing \\documentclass declaration")
	if "\\begin{document}" not in updated:
		findings.append("missing \\begin{document}")
	if "\\end{document}" not in updated:
		findings.append("missing \\end{document}")

	if "\\\\resume" in updated or "\\\\section" in updated or "\\\\vspace" in updated:
		findings.append("contains suspicious double-backslash control commands")
	if "{\\resumeItemListEnd}" in updated or "{\\resumeSubHeadingListEnd}" in updated:
		findings.append("contains braced resume list terminator macros")

	return LatexLintResult(
		latex_document=updated,
		repairs_applied=repairs,
		findings=findings,
	)


def _run_chktex(cwd: Path, tex_name: str, chktex_path: str) -> tuple[list[str], str | None]:
	"""Run chktex if available and return compact finding summaries."""
	command = [
		chktex_path,
		"-q",
		"-I0",
		"-f%l:%c:%k:%n:%m\\n",
		tex_name,
	]
	try:
		result = subprocess.run(
			command,
			capture_output=True,
			text=True,
			timeout=20,
			check=False,
			cwd=str(cwd),
		)
	except (subprocess.TimeoutExpired, OSError):
		return [], None

	output = (result.stdout or "").strip()
	if not output:
		return [], None

	parsed: list[str] = []
	for line in output.splitlines():
		match = CHKTEX_LINE_PATTERN.match(line.strip())
		if not match:
			continue
		kind = (match.group("kind") or "msg").strip() or "msg"
		code = match.group("code")
		msg = (match.group("message") or "").strip()
		parsed.append(f"L{match.group('line')}:{match.group('col')} [{kind} {code}] {msg}")
		if len(parsed) >= MAX_CHKTEX_FINDINGS:
			break

	if not parsed:
		return [], None

	return parsed, output[-2000:]


def sanitize_latex_document_for_compile(
	latex_document: str,
	normalize_json_escaped_latex: bool = True,
) -> str:
	"""Escape common model-output mistakes while preserving tabular alignment syntax."""
	if not latex_document:
		return latex_document

	if normalize_json_escaped_latex:
		latex_document = _normalize_escaped_latex_document(latex_document)

	latex_document = _close_runaway_inline_text_commands(latex_document)
	latex_document = _normalize_empty_text_command_small_args(latex_document)
	latex_document = _normalize_broken_item_section_collisions(latex_document)
	latex_document = _close_small_itemize_group_brace(latex_document)
	latex_document = _drop_orphan_lowercase_fragments(latex_document)
	latex_document = _normalize_unclosed_resume_lists(latex_document)

	# Correct malformed \end{document>} and similar LLM typos.
	latex_document = _fix_malformed_end_document(latex_document)

	# Insert a newline between any % separator comment and a structural LaTeX
	# command that the LLM concatenated onto the same line.  Only triggered when
	# the comment text contains 3+ separator chars (e.g. "---") so legitimately
	# commented-out \begin/\end blocks are not accidentally activated.
	latex_document = _split_comment_concatenated_commands(latex_document)

	# Close any unclosed \resumeSubHeadingListStart / \resumeItemListStart macros
	# that the LLM omitted due to output truncation.
	latex_document = _close_unclosed_resume_list_macros(latex_document)

	# When the LLM omits real newlines, % comment markers at brace-depth 0 would
	# comment out the entire rest of the single-line document.  Insert newlines
	# before such markers so each comment is isolated to its own line.
	latex_document = _split_single_line_latex_comments(latex_document)

	# Fix LLM over-escaping of math-mode dollar signs (e.g. \$\vcenter → $\vcenter).
	latex_document = _fix_over_escaped_math_dollars(latex_document)

	# Escape bare % signs that appear mid-line; unescaped % starts a LaTeX comment
	# and will eat the rest of the line including closing braces.
	latex_document = _escape_unescaped_percent_in_content(latex_document)

	# Keep compile resilient to minor brace mismatches in cached profile/template text.
	latex_document = _close_unbalanced_latex_braces(latex_document)

	# Strip XML/JSON artifacts that sometimes appear after \end{document} (e.g. </latex>, quotes).
	latex_document = _strip_post_end_document_artifacts(latex_document)

	return _escape_unescaped_ampersands_outside_tabular(latex_document)


def _escape_unescaped_ampersands_outside_tabular(latex_document: str) -> str:
	lines = latex_document.splitlines()
	sanitized_lines: list[str] = []
	in_tabular = False

	for line in lines:
		stripped = line.strip()
		if stripped.startswith("\\begin{tabular"):
			in_tabular = True
		elif stripped.startswith("\\end{tabular"):
			in_tabular = False

		if in_tabular:
			sanitized_lines.append(line)
			continue

		sanitized_lines.append(UNESCAPED_AMPERSAND_PATTERN.sub(r"\\&", line))

	return "\n".join(sanitized_lines)


# Matches \$ immediately followed by a backslash-command that appears in math-mode
# bullet style definitions.  The LLM sometimes over-escapes the opening math $ to
# \\$ in fully-escaped payloads, producing \$ after decoding instead of the bare $
# that LaTeX expects for entering inline math.
_END_DOCUMENT_CORRUPTION_PATTERN = re.compile(r"\\end\{document[^}]*\}?", re.IGNORECASE)


def _fix_malformed_end_document(latex_document: str) -> str:
	"""Correct \\end{document>}, \\end{document>} and similar LLM typos to \\end{document}.

	LLMs sometimes embed a stray > or other character inside \\end{document},
	often as a leftover from markdown rendering.  The closing brace may be
	present but after the stray char (\\end{document>}), or absent entirely
	when the LLM truncated its output (\\end{document>).  Both forms are
	handled by making the closing brace optional in the regex.

	Fast path: return immediately if no character other than '}' follows
	'document' inside the \\end{...} group.
	"""
	if not re.search(r"\\end\{document[^}]", latex_document, re.IGNORECASE):
		return latex_document
	return _END_DOCUMENT_CORRUPTION_PATTERN.sub(r"\\end{document}", latex_document)


# Regex: a % SEPARATOR comment (contains 3+ consecutive dash/equals/etc. chars)
# immediately followed by a structural LaTeX command, with no intervening newline
# or backslash between the separator text and the command.
#
# The requirement for 3+ separator chars distinguishes section-divider comments
# like "%----------HEADING----------\begin{center}" from legitimately-commented-out
# code like "%\begin{itemize}" or "%\end{itemize}", which must NOT be activated.
_COMMENT_CONCATENATED_CMD_PATTERN = re.compile(
	r"(%[^\n\\]*[-=*#+]{3,}[^\n\\]*)(\\(?:begin|end|section|subsection|subsubsection"
	r"|paragraph|vspace|hspace|noindent|centering|renewcommand|newcommand)\s*(?=[\{\[\\]))",
	re.MULTILINE,
)


def _split_comment_concatenated_commands(latex_document: str) -> str:
	"""Insert a newline between a % separator comment and a LaTeX command on the same line.

	LLMs sometimes omit the newline between a section-separator comment and the
	following structural command, e.g.:

	    %----------HEADING----------\\begin{center}

	should be:

	    %----------HEADING----------
	    \\begin{center}

	Without the newline the % makes everything after it a LaTeX comment, so
	\\begin{center} never executes while a later \\end{center} still does →
	fatal "\\begin{document} ended by \\end{center}" error.

	To avoid activating intentionally-commented-out code like "%\\begin{itemize}"
	or "%\\end{itemize}", the split is only applied when the comment text contains
	3+ consecutive separator characters (dashes, equals, etc.) before the command.
	"""
	if "%" not in latex_document:
		return latex_document
	return _COMMENT_CONCATENATED_CMD_PATTERN.sub(r"\1\n\2", latex_document)


def _close_unclosed_resume_list_macros(latex_document: str) -> str:
	"""Insert missing \\resumeSubHeadingListEnd / \\resumeItemListEnd before \\end{document}.

	When the LLM truncates its output it sometimes omits the closing resume-list
	macros.  Each macro expands to \\end{itemize}, so a missing one leaves an
	unclosed \\begin{itemize} that causes a fatal compile error.  This function
	counts the open/close calls for each macro pair and appends the missing
	closers (inner list first, then outer) immediately before \\end{document}.
	"""
	if "\\end{document}" not in latex_document:
		return latex_document

	missing_item = (
		latex_document.count("\\resumeItemListStart")
		- latex_document.count("\\resumeItemListEnd")
	)
	missing_sub = (
		latex_document.count("\\resumeSubHeadingListStart")
		- latex_document.count("\\resumeSubHeadingListEnd")
	)

	if missing_item <= 0 and missing_sub <= 0:
		return latex_document

	# Close inner list before outer list.
	insert = ""
	if missing_item > 0:
		insert += "\\resumeItemListEnd\n" * missing_item
	if missing_sub > 0:
		insert += "\\resumeSubHeadingListEnd\n" * missing_sub

	return latex_document.replace("\\end{document}", insert + "\\end{document}", 1)


def _strip_post_end_document_artifacts(latex_document: str) -> str:
	"""Remove anything the LLM appended after \\end{document}.

	LLMs sometimes leak JSON structure or XML tags (e.g. </latex>, closing quotes)
	after the document's \\end{document}.  pdflatex/xelatex ignore such content
	but some MiKTeX versions emit spurious errors.  Truncating at the closing
	tag keeps the compiled output deterministic.
	"""
	end_tag = "\\end{document}"
	pos = latex_document.find(end_tag)
	if pos == -1:
		return latex_document
	return latex_document[: pos + len(end_tag)] + "\n"


def _split_single_line_latex_comments(latex_document: str) -> str:
	"""Insert real newlines before % LaTeX comments in near-single-line documents.

	When the LLM returns a document with no actual newlines (only % comment
	markers inline), pdflatex interprets everything from the first % to end-of-line
	as a comment — which in a single-line document means the entire rest of the
	file, including \\begin{document}.  This pre-pass inserts \\n before each %
	at brace-depth 0 so each comment is safely contained to its own line.
	"""
	if latex_document.count("\n") >= 2:
		return latex_document
	result: list[str] = []
	brace_depth = 0
	i = 0
	while i < len(latex_document):
		ch = latex_document[i]
		if ch == "\\" and i + 1 < len(latex_document):
			result.append(ch)
			result.append(latex_document[i + 1])
			i += 2
			continue
		if ch == "{":
			brace_depth += 1
		elif ch == "}" and brace_depth > 0:
			brace_depth -= 1
		elif ch == "%" and brace_depth == 0:
			if result and result[-1] != "\n":
				result.append("\n")
			result.append("%")
			i += 1
			while i < len(latex_document) and latex_document[i] not in ("\n", "\r"):
				result.append(latex_document[i])
				i += 1
			result.append("\n")
			continue
		result.append(ch)
		i += 1
	return "".join(result)


_MATH_DOLLAR_OVERCORRECT_PATTERN = re.compile(
	r"\\\$\\(vcenter|hbox|vbox|bullet|cdot|tiny|small|Bigg|bigg|mathbf|mathit|textstyle)\b"
)


def _fix_over_escaped_math_dollars(latex_document: str) -> str:
	"""Convert \\$\\<mathcmd> → $\\<mathcmd> produced by LLM over-escaping."""
	if r"\$" not in latex_document:
		return latex_document
	return _MATH_DOLLAR_OVERCORRECT_PATTERN.sub(r"$\\\1", latex_document)


def _escape_unescaped_percent_in_content(latex_document: str) -> str:
	"""Escape bare % signs that appear inside open LaTeX argument groups.

	A bare % outside any brace group starts a LaTeX comment and is intentional
	(e.g. ``\\fancyhf{} % clear fields`` or a full-line comment).  A bare % that
	appears while brace_depth > 0 (i.e. inside a macro argument) starts a comment
	that swallows the rest of the line, including closing braces, causing
	runaway-argument / "File ended while scanning" errors.
	"""
	if "%" not in latex_document:
		return latex_document

	result_lines: list[str] = []
	brace_depth: int = 0
	for line in latex_document.splitlines():
		out: list[str] = []
		i = 0
		while i < len(line):
			ch = line[i]
			if ch == "\\" and i + 1 < len(line):
				# Backslash + next character are one TeX token.
				out.append(ch)
				out.append(line[i + 1])
				i += 2
				continue
			if ch == "{":
				brace_depth += 1
				out.append(ch)
				i += 1
				continue
			if ch == "}" and brace_depth > 0:
				brace_depth -= 1
				out.append(ch)
				i += 1
				continue
			if ch == "%":
				if brace_depth > 0:
					# Inside a macro argument: % must be a literal percent, not a comment.
					out.append("\\%")
				else:
					# Outside any argument: intentional LaTeX comment — preserve verbatim.
					out.append(line[i:])
					break
				i += 1
				continue
			out.append(ch)
			i += 1
		result_lines.append("".join(out))

	return "\n".join(result_lines)


def _classify_latex_compile_failure(
	latex_document: str,
	stdout_text: str,
	stderr_text: str,
) -> str | None:
	combined = "\n".join(part for part in (stderr_text, stdout_text) if part)
	if not combined.strip():
		return None

	if LATEX_ALIGNMENT_ERROR_PATTERN.search(combined):
		return "ampersand"
	if LATEX_MISSING_STYLE_PATTERN.search(combined):
		return "missing-style"
	if LATEX_BRACE_ERROR_PATTERN.search(combined):
		return "brace"
	if "Emergency stop" in combined and ("\\resumeItemList" in latex_document or "\\resumeSubHeadingList" in latex_document):
		return "list-structure"
	if "!" in combined or "latex error" in combined.lower() or "emergency stop" in combined.lower():
		return "generic-latex-error"
	return None


def _extract_missing_style_files(stdout_text: str, stderr_text: str) -> list[str]:
	combined = "\n".join(part for part in (stderr_text, stdout_text) if part)
	if not combined.strip():
		return []
	missing: list[str] = []
	for match in LATEX_MISSING_STYLE_PATTERN.finditer(combined):
		entry = (match.group(1) or match.group(2) or "").strip()
		if entry and entry not in missing:
			missing.append(entry)
	return missing


def _apply_targeted_latex_auto_fix(
	latex_document: str,
	failure_kind: str,
	stdout_text: str = "",
	stderr_text: str = "",
) -> tuple[str, list[str]]:
	if not latex_document:
		return latex_document, []

	if failure_kind == "ampersand":
		updated = _escape_unescaped_ampersands_outside_tabular(latex_document)
		if updated != latex_document:
			return updated, ["escaped unescaped ampersands"]
		return latex_document, []

	if failure_kind == "brace":
		updated = latex_document
		repairs: list[str] = []

		next_text = _close_runaway_inline_text_commands(updated)
		if next_text != updated:
			repairs.append("closed runaway inline text commands")
			updated = next_text

		next_text = _normalize_empty_text_command_small_args(updated)
		if next_text != updated:
			repairs.append("normalized malformed text command payloads")
			updated = next_text

		next_text = _normalize_broken_item_section_collisions(updated)
		if next_text != updated:
			repairs.append("separated broken item/section collisions")
			updated = next_text

		next_text = _close_small_itemize_group_brace(updated)
		if next_text != updated:
			repairs.append("closed small itemize group braces")
			updated = next_text

		next_text = _normalize_unclosed_resume_lists(updated)
		if next_text != updated:
			repairs.append("closed unbalanced resume list blocks")
			updated = next_text

		next_text = _close_unbalanced_latex_braces(updated)
		if next_text != updated:
			repairs.append("balanced unclosed braces")
			updated = next_text

		return updated, repairs

	if failure_kind == "list-structure":
		updated = latex_document
		repairs: list[str] = []
		next_text = _normalize_unclosed_resume_lists(updated)
		if next_text != updated:
			repairs.append("closed unbalanced resume list blocks")
			updated = next_text
		next_text = _close_small_itemize_group_brace(updated)
		if next_text != updated:
			repairs.append("closed small itemize group braces")
			updated = next_text
		return updated, repairs

	if failure_kind == "missing-style":
		missing_files = _extract_missing_style_files(stdout_text, stderr_text)
		if not missing_files:
			return _apply_general_latex_repairs(latex_document)
		updated, removed = _strip_missing_style_usepackages(latex_document, missing_files)
		if not removed or updated == latex_document:
			return _apply_general_latex_repairs(latex_document)
		removed_label = ", ".join(sorted(set(removed)))
		return updated, [f"stripped missing style packages: {removed_label}"]

	if failure_kind == "generic-latex-error":
		return _apply_general_latex_repairs(latex_document)

	return _apply_general_latex_repairs(latex_document)


def _apply_general_latex_repairs(latex_document: str) -> tuple[str, list[str]]:
	"""Apply conservative broad-spectrum structural cleanups for unknown LaTeX errors."""
	updated = latex_document
	repairs: list[str] = []

	next_text = _close_runaway_inline_text_commands(updated)
	if next_text != updated:
		repairs.append("closed runaway inline text commands")
		updated = next_text

	next_text = _normalize_empty_text_command_small_args(updated)
	if next_text != updated:
		repairs.append("normalized malformed text command payloads")
		updated = next_text

	next_text = _normalize_broken_item_section_collisions(updated)
	if next_text != updated:
		repairs.append("separated broken item/section collisions")
		updated = next_text

	next_text = _close_small_itemize_group_brace(updated)
	if next_text != updated:
		repairs.append("closed small itemize group braces")
		updated = next_text

	next_text = _normalize_unclosed_resume_lists(updated)
	if next_text != updated:
		repairs.append("closed unbalanced resume list blocks")
		updated = next_text

	next_text = _close_unbalanced_latex_braces(updated)
	if next_text != updated:
		repairs.append("balanced unclosed braces")
		updated = next_text

	next_text = _escape_unescaped_ampersands_outside_tabular(updated)
	if next_text != updated:
		repairs.append("escaped unescaped ampersands")
		updated = next_text

	return updated, repairs


def _strip_missing_style_usepackages(latex_document: str, missing_files: list[str]) -> tuple[str, list[str]]:
	"""Remove \\usepackage entries for missing .sty files from an in-memory LaTeX document."""
	missing_styles = {
		Path(entry).stem.strip().casefold()
		for entry in missing_files
		if str(entry).strip().lower().endswith(".sty")
	}
	if not missing_styles:
		return latex_document, []

	removed_packages: list[str] = []
	updated_lines: list[str] = []
	for line in latex_document.splitlines():
		match = USEPACKAGE_PATTERN.search(line)
		if not match:
			updated_lines.append(line)
			continue

		declared = [pkg.strip() for pkg in match.group(1).split(",") if pkg.strip()]
		if not declared:
			updated_lines.append(line)
			continue

		kept = [pkg for pkg in declared if pkg.casefold() not in missing_styles]
		removed_now = [pkg for pkg in declared if pkg.casefold() in missing_styles]
		if not removed_now:
			updated_lines.append(line)
			continue

		removed_packages.extend(removed_now)
		if not kept:
			# Drop the full line only when every package on this declaration is unavailable.
			continue

		updated_lines.append(line[: match.start(1)] + ",".join(kept) + line[match.end(1):])

	return "\n".join(updated_lines), removed_packages


def _sanitize_engine_specific_latex(latex_document: str, engine_name: str) -> tuple[str, list[str]]:
	"""Drop pdfTeX-specific unicode directives when compiling with non-pdfTeX engines."""
	if engine_name == "pdflatex":
		return latex_document, []

	removed_tokens: list[str] = []
	updated_lines: list[str] = []
	for line in latex_document.splitlines():
		if GLYPHTOUNICODE_INPUT_PATTERN.match(line):
			removed_tokens.append("input{glyphtounicode}")
			continue
		if PDF_GLYPH_UNICODE_PATTERN.match(line):
			removed_tokens.append("pdfglyphtounicode")
			continue
		if PDF_GENTOUNICODE_PATTERN.match(line):
			removed_tokens.append("pdfgentounicode")
			continue
		updated_lines.append(line)

	return "\n".join(updated_lines), removed_tokens


def compile_latex_to_pdf(
	latex_document: str,
	job_title: str,
	template_path: Path = TEMPLATE_PATH,
	log_path: Path | None = None,
	normalize_json_escaped_latex: bool = True,
	command_runner: Callable[[list[str], Path], subprocess.CompletedProcess[str]] | None = None,
	auto_fix_on_failure: bool = True,
	max_auto_fix_retries: int = LATEX_AUTOFIX_MAX_RETRIES,
) -> LatexCompileResult:
	environment = check_template_compile_environment(template_path)
	if not (environment.pdflatex_path or environment.xelatex_path or environment.lualatex_path):
		missing = "no LaTeX engine found (pdflatex/xelatex/lualatex)"
		return LatexCompileResult(
			status="unavailable",
			message=f"LaTeX compile environment is not ready: {missing}",
			log_path=log_path,
		)

	runner = command_runner or _run_latex_command
	pdf_stem = sanitize_pdf_stem(job_title)
	resolved_log_path = log_path or template_path.with_suffix(".log")
	with tempfile.TemporaryDirectory(prefix="rebuilt_resume_") as temp_dir:
		workdir = Path(temp_dir)
		tex_path = workdir / f"{pdf_stem}.tex"
		lint_result = lint_latex_document_for_compile(latex_document)
		sanitized_tex = sanitize_latex_document_for_compile(
			lint_result.latex_document,
			normalize_json_escaped_latex=normalize_json_escaped_latex,
		)
		sanitized_tex, removed_packages = _strip_missing_style_usepackages(
			sanitized_tex,
			list(getattr(environment, "missing_files", []) or []),
		)
		tex_path.write_text(sanitized_tex, encoding="utf-8")

		engines = _available_latex_engines(environment, sanitized_tex)
		if not engines:
			return LatexCompileResult(
				status="unavailable",
				message="LaTeX compile environment is not ready: no LaTeX engine found (pdflatex/xelatex/lualatex)",
				log_path=resolved_log_path,
			)

		attempt_logs: list[str] = []
		lint_findings = list(lint_result.findings)
		repairs_applied: list[str] = list(lint_result.repairs_applied)
		if lint_result.repairs_applied:
			attempt_logs.append(
				"[lint:internal] Applied preflight lint repairs: "
				+ ", ".join(lint_result.repairs_applied)
			)
		if lint_result.findings:
			attempt_logs.append(
				"[lint:internal] Findings: " + "; ".join(lint_result.findings)
			)

		chktex_path = _find_latex_executable("chktex")
		if chktex_path:
			chktex_findings, chktex_excerpt = _run_chktex(workdir, tex_path.name, chktex_path)
			if chktex_findings:
				lint_findings.extend([f"chktex: {item}" for item in chktex_findings])
				attempt_logs.append("[lint:chktex] " + " | ".join(chktex_findings))
			elif chktex_excerpt:
				attempt_logs.append(f"[lint:chktex]\n{chktex_excerpt}")

		if removed_packages:
			removed_label = ", ".join(sorted(set(removed_packages)))
			attempt_logs.append(
				f"[preflight] Missing style packages were stripped for this compile attempt: {removed_label}"
			)
		last_excerpt: str | None = None
		repair_attempts = 0
		deadline = time.monotonic() + LATEX_TOTAL_TIMEOUT_SECONDS
		stop_all_engines = False
		bounded_retries = max(0, min(max_auto_fix_retries, 4))
		for engine_name, engine_path in engines:
			if time.monotonic() >= deadline:
				last_excerpt = "Compilation budget exhausted before trying remaining engines."
				break

			engine_tex, removed_engine_tokens = _sanitize_engine_specific_latex(sanitized_tex, engine_name)
			if removed_engine_tokens:
				removed_label = ", ".join(sorted(set(removed_engine_tokens)))
				attempt_logs.append(
					f"[{engine_name}:preflight] Stripped engine-incompatible directives: {removed_label}"
				)

			command = [
				engine_path,
				"-interaction=nonstopmode",
				"-halt-on-error",
				tex_path.name,
			]
			retry_count = 0
			while True:
				if time.monotonic() >= deadline:
					last_excerpt = "Compilation budget exhausted before trying remaining engines."
					stop_all_engines = True
					break

				tex_path.write_text(engine_tex, encoding="utf-8")
				attempt_tag = engine_name if retry_count == 0 else f"{engine_name}:retry{retry_count}"
				try:
					result = runner(command, workdir)
				except subprocess.TimeoutExpired as exc:
					stderr_text = exc.stderr or f"{engine_name} timed out after {exc.timeout} seconds."
					stdout_text = exc.stdout or ""
					attempt_logs.append(f"[{attempt_tag}]\n{stdout_text}\n{stderr_text}".strip())
					last_excerpt = (stderr_text or stdout_text)[-1500:] if (stderr_text or stdout_text) else None
					stop_all_engines = True
					break
				except OSError as exc:
					attempt_logs.append(f"[{attempt_tag}]\n{exc}".strip())
					last_excerpt = str(exc)[-1500:]
					break

				stdout_text = result.stdout or ""
				stderr_text = result.stderr or ""
				attempt_logs.append(f"[{attempt_tag}]\n{stdout_text}\n{stderr_text}".strip())
				if result.returncode == 124:
					last_excerpt = ((stderr_text or "").strip() or (stdout_text or "").strip())[-1500:] or None
					stop_all_engines = True
					break

				final_returncode = result.returncode
				final_stdout = stdout_text
				final_stderr = stderr_text
				if final_returncode == 0 and _latex_needs_rerun(final_stdout, final_stderr):
					if time.monotonic() >= deadline:
						last_excerpt = "Compilation budget exhausted before rerun pass."
						stop_all_engines = True
						break
					try:
						second_pass = runner(command, workdir)
					except subprocess.TimeoutExpired as exc:
						stderr_text_2 = exc.stderr or f"{engine_name} second pass timed out after {exc.timeout} seconds."
						stdout_text_2 = exc.stdout or ""
						attempt_logs.append(f"[{attempt_tag}:pass2]\\n{stdout_text_2}\\n{stderr_text_2}".strip())
						last_excerpt = (stderr_text_2 or stdout_text_2)[-1500:] if (stderr_text_2 or stdout_text_2) else None
						stop_all_engines = True
						break
					except OSError as exc:
						attempt_logs.append(f"[{attempt_tag}:pass2]\\n{exc}".strip())
						last_excerpt = str(exc)[-1500:]
						break

					stdout_text_2 = second_pass.stdout or ""
					stderr_text_2 = second_pass.stderr or ""
					attempt_logs.append(f"[{attempt_tag}:pass2]\n{stdout_text_2}\n{stderr_text_2}".strip())
					final_returncode = second_pass.returncode
					final_stdout = stdout_text_2
					final_stderr = stderr_text_2
					if final_returncode == 124:
						last_excerpt = ((final_stderr or "").strip() or (final_stdout or "").strip())[-1500:] or None
						stop_all_engines = True
						break

				if final_returncode == 0:
					pdf_path = tex_path.with_suffix(".pdf")
					if not pdf_path.exists():
						last_excerpt = f"{engine_name} completed without producing a PDF."[-1500:]
						break

					_write_compile_log(resolved_log_path, "\n\n".join(attempt_logs), "")
					return LatexCompileResult(
						status="ok",
						message=f"Compiled LaTeX resume to PDF using {engine_name}.",
						pdf_bytes=pdf_path.read_bytes(),
						pdf_name=f"{pdf_stem}.pdf",
						log_path=resolved_log_path,
						repairs_applied=repairs_applied.copy(),
						repair_attempts=repair_attempts,
						lint_findings=lint_findings,
					)

				last_excerpt = ((final_stderr or "").strip() or (final_stdout or "").strip())[-1500:] or None
				can_retry_with_fix = auto_fix_on_failure and retry_count < bounded_retries
				if not can_retry_with_fix:
					break

				failure_kind = _classify_latex_compile_failure(engine_tex, final_stdout, final_stderr)
				if failure_kind is None:
					break

				fixed_tex, applied_repairs = _apply_targeted_latex_auto_fix(
					engine_tex,
					failure_kind,
					stdout_text=final_stdout,
					stderr_text=final_stderr,
				)
				if not applied_repairs or fixed_tex == engine_tex:
					break

				retry_count += 1
				repair_attempts += 1
				for repair_label in applied_repairs:
					if repair_label not in repairs_applied:
						repairs_applied.append(repair_label)
				attempt_logs.append(
					f"[{engine_name}:repair#{retry_count}] Applied auto-fixes ({failure_kind}): {', '.join(applied_repairs)}"
				)
				engine_tex = fixed_tex

			if stop_all_engines:
				break

		_write_compile_log(resolved_log_path, "\n\n".join(attempt_logs), "")
		failure_message = "LaTeX compilation failed across all available engines."
		if last_excerpt and "timed out" in last_excerpt.lower():
			failure_message = "LaTeX compilation timed out across all available engines."
		return LatexCompileResult(
			status="error",
			message=failure_message,
			log_excerpt=last_excerpt,
			log_path=resolved_log_path,
			repairs_applied=repairs_applied,
			repair_attempts=repair_attempts,
			lint_findings=lint_findings,
		)


def _run_latex_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
	try:
		return subprocess.run(
			command,
			capture_output=True,
			text=True,
			timeout=LATEX_ENGINE_TIMEOUT_SECONDS,
			check=False,
			cwd=str(cwd),
		)
	except subprocess.TimeoutExpired as exc:
		return subprocess.CompletedProcess(
			command,
			124,
			exc.stdout or "",
			exc.stderr or f"LaTeX compile timed out after {exc.timeout} seconds.",
		)
	except OSError as exc:
		return subprocess.CompletedProcess(command, 1, "", str(exc))


def _write_compile_log(log_path: Path, stdout: str, stderr: str) -> None:
	try:
		log_path.parent.mkdir(parents=True, exist_ok=True)
		parts = []
		if stdout.strip():
			parts.append(stdout.rstrip())
		if stderr.strip():
			parts.append(stderr.rstrip())
		log_path.write_text("\n\n".join(parts), encoding="utf-8")
	except OSError:
		pass
