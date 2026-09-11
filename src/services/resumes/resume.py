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

from services import capacity, platform_support
from services.resumes.ats_check import TYPE3_UNREADABLE_RATIO, audit_pdf_ats

LATEX_WRAPPER_PACKAGE = "pdflatex"
RESUMES_CACHE_ROOT = Path(__file__).resolve().parent / "resumes_cache"
TEMPLATE_PATH = RESUMES_CACHE_ROOT / "template.tex"
PROFILE_ALLOWED_FILE_NAMES = ("baseinfo.txt", "instructions.txt", "template.tex")
# Optional per-profile files (structured pipeline). Never required, never
# seeded from the example profile, and never purged.
PROFILE_OPTIONAL_FILE_NAMES = ("structured_config.json",)
EXAMPLE_PROFILE_KEY = "example"
LLM_PROVIDER_SWITCH_ORDER = ("gemini", "gemini-flash", "openrouter", "groq")
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
# Stretched on slower hardware: pdflatex is single-threaded and CPU-bound, so a
# budget tuned on a desktop can cut off a legitimate compile on a small host.
LATEX_ENGINE_TIMEOUT_SECONDS = int(capacity.timeout(_bounded_env_int(
	"LATEX_ENGINE_TIMEOUT_SECONDS",
	75,
	minimum=30,
	maximum=180,
)))

# Keep total compile budget generous for engine fallback + rerun pass,
# but bounded so command handling remains responsive.
# Must stretch with the per-engine budget above, otherwise the total would cut
# off the engine fallback + rerun pass that the per-engine budget now allows.
LATEX_TOTAL_TIMEOUT_SECONDS = int(capacity.timeout(_bounded_env_int(
	"LATEX_TOTAL_TIMEOUT_SECONDS",
	180,
	minimum=60,
	maximum=420,
)))

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

# Per-MODEL overrides, keyed by model-id prefix (so ":free" suffixes match).
# A provider slot's defaults are tuned for the model that historically ran
# there; swapping the slot's model must re-tune its limits without touching
# the other providers' behavior.
MODEL_CAPABILITY_OVERRIDES: tuple[tuple[str, LLMProviderCapabilities], ...] = (
	(
		"nvidia/nemotron-3-super",
		LLMProviderCapabilities(
			supports_context_cache=False,
			max_prompt_chars=120_000,
			# Observed live generations: 75-250s (mean ~150s). The 45s
			# openrouter default was tuned for gpt-oss-120b and would time
			# out nearly every nemotron call before it finished.
			request_timeout_seconds=300,
		),
	),
)


def resolve_provider_capabilities(
	provider_name: str, model: str | None = None
) -> LLMProviderCapabilities | None:
	"""Capabilities for a provider slot, honoring per-model overrides first."""
	if model:
		for prefix, capabilities in MODEL_CAPABILITY_OVERRIDES:
			if model.startswith(prefix):
				return capabilities
	return PROVIDER_CAPABILITIES.get(provider_name)


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
LUA_ONLY_HINT_PATTERN = re.compile(r"\\directlua\{|\\begin\{luacode\}", re.IGNORECASE)
LATEX_RERUN_HINT_PATTERN = re.compile(r"Rerun to get cross-references right|Label\(s\) may have changed", re.IGNORECASE)
LATEX_PAGE_COUNT_PATTERN = re.compile(r"Output written on [^(]*\((\d+)\s+pages?", re.IGNORECASE)
LATEX_ALIGNMENT_ERROR_PATTERN = re.compile(r"Misplaced alignment tab character\s*&", re.IGNORECASE)
LATEX_BRACE_ERROR_PATTERN = re.compile(
	r"Runaway argument\?|File ended while scanning use of|Missing \} inserted|Too many \}|Paragraph ended before",
	re.IGNORECASE,
)
LATEX_MISSING_STYLE_PATTERN = re.compile(r"File\s+`([^`]+\.sty)'\s+not\s+found|File\s+([^\s]+\.sty)\s+not\s+found", re.IGNORECASE)
# A font the engine cannot load or build.  Distinct from a missing .sty: the
# package is installed, but its font files are absent and cannot be generated
# (e.g. MiKTeX "Sorry, but miktex-makemf did not succeed").  The error names a
# font file, never the package that asked for it, so nothing was stripped and
# the build died -- even though dropping the font package and falling back to
# the default typeface yields a perfectly readable resume.
LATEX_FONT_FAILURE_PATTERN = re.compile(
	r"pdfTeX error[^\n]*\(file\s+[^)]+\)\s*:\s*Font"
	r"|Font\s+\S+\s+not\s+loadable"
	r"|miktex-makemf\s+did\s+not\s+succeed"
	r"|Font\s+shape\s+`[^']+'\s+undefined",
	re.IGNORECASE,
)
LATEX_FONT_FILE_PATTERN = re.compile(r"\(file\s+([^)]+?)\s*\)\s*:\s*Font", re.IGNORECASE)
# Font packages common in resume templates.  Used only as a fallback when the
# failing font name cannot be matched back to a declared package.
KNOWN_FONT_PACKAGES = frozenset(
	{
		"arev", "avant", "bookman", "charter", "chancery", "cmbright", "concrete",
		"courier", "crimson", "ebgaramond", "fbb", "fourier", "helvet", "kpfonts",
		"lato", "libertine", "libertinus", "lmodern", "mathpazo", "mathptmx",
		"newpxtext", "newtxtext", "nimbusserif", "opensans", "palatino", "roboto",
		"sourcecodepro", "sourcesanspro", "sourceserifpro", "tgadventor",
		"tgbonum", "tgheros", "tgpagella", "tgtermes", "times", "utopia",
		"xcharter",
	}
)
# Ordering for "did the salvage retry actually help?".  Only ever used to
# compare two renders of the same document, never as a quality gate.
ATS_QUALITY_RANK = {"unreadable": 0, "unknown": 1, "degraded": 2, "ok": 3}

GLYPHTOUNICODE_INPUT_PATTERN = re.compile(r"^\s*\\input\{glyphtounicode\}\s*$", re.IGNORECASE)
PDF_GLYPH_UNICODE_PATTERN = re.compile(r"^\s*\\pdfglyphtounicode\b.*$", re.IGNORECASE)
PDF_GENTOUNICODE_PATTERN = re.compile(r"^\s*\\pdfgentounicode\b.*$", re.IGNORECASE)
RESUME_SUBHEADING_MACRO_PATTERN = re.compile(r"\\resumeSubheading\s*(?=\{)")
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
	page_count: int | None = None
	# Post-compile ATS readability audit of the produced PDF.  "unknown" means
	# the bytes could not be parsed (stubs in tests), and is treated as a
	# pass-through rather than a failure.
	ats_status: str = "unknown"
	ats_findings: list[str] = field(default_factory=list)
	ats_degradations: list[str] = field(default_factory=list)


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
	allowed = set(PROFILE_ALLOWED_FILE_NAMES) | set(PROFILE_OPTIONAL_FILE_NAMES)
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


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
	try:
		return subprocess.run(
			command,
			capture_output=True,
			text=True,
			encoding="utf-8",
			errors="replace",
			stdin=subprocess.DEVNULL,  # same reason as _run_latex_command
			timeout=15,
			check=False,
			**platform_support.no_window_kwargs(),
		)
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
	# Delegates to the platform shim: PATH first, then the install locations that
	# are commonly off PATH (MiKTeX on Windows, TeX Live on Linux -- the latter
	# matters because systemd hands a service a minimal PATH).
	return platform_support.find_latex_executable(command_name)


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


def _template_engine_family(template_text: str) -> str:
	"""Classify a template by the font machinery it uses.

	This is not a preference, it is a hard constraint.  A ``fontspec`` template
	cannot compile under pdflatex at all ("Fatal Package fontspec Error"), and a
	Type1-font template (XCharter, ``[T1]{fontenc}``) compiled under xelatex
	silently degrades to Type 3 bitmap fonts whose text extracts as glyph names
	-- a PDF that looks perfect and is unreadable to an ATS.  Each engine is
	correct for its own family and broken for the other, so the families must
	never be crossed as a fallback.
	"""
	if LUA_ONLY_HINT_PATTERN.search(template_text):
		return "luatex"
	if XELATEX_HINT_PATTERN.search(template_text) or LUALATEX_HINT_PATTERN.search(template_text):
		return "unicode"
	return "pdftex"


def _preferred_latex_engines(template_text: str) -> list[str]:
	family = _template_engine_family(template_text)
	if family == "luatex":
		# \directlua / luacode is lualatex-only; xelatex cannot run it.
		return ["lualatex"]
	if family == "unicode":
		# fontspec/unicode-math load system fonts and embed them with a
		# ToUnicode map under either Unicode engine.  Both are in-family.
		return ["xelatex", "lualatex"]
	# pdftex family: pdflatex only.  Falling back to a Unicode engine here is
	# the cross-family case that produces silently unreadable PDFs.
	return ["pdflatex"]


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


def _fix_html_comment_tags(latex_document: str) -> str:
	"""Replace HTML-style </comment> and </comment} closing tags with \\end{comment}.

	LLMs (especially Gemini) occasionally emit HTML-style closing tags for LaTeX
	comment environments. Both </comment> and </comment} are invalid LaTeX and
	leave the comment block unclosed, causing a 'Runaway argument' fatal error.
	"""
	import re
	fixed = re.sub(r"</comment[}>]?", r"\\end{comment}", latex_document)
	return fixed


def _remove_dangling_single_line_backslashes(latex_document: str) -> str:
	"""Remove accidental single trailing backslashes at end-of-line.

	Conservative behavior: preserves intentional LaTeX line breaks (``\\``)
	and skips comment lines.
	"""
	if "\\" not in latex_document:
		return latex_document

	updated_lines: list[str] = []
	removed_any = False
	for line in latex_document.splitlines():
		trimmed = line.rstrip()
		if trimmed.endswith("\\") and not trimmed.endswith("\\\\") and not trimmed.lstrip().startswith("%"):
			updated_lines.append(trimmed[:-1])
			removed_any = True
		else:
			# Leave untouched lines byte-identical: whitespace-only rstrip
			# changes here made the caller report this repair on every
			# document, even when no backslash was removed.
			updated_lines.append(line)

	if not removed_any:
		return latex_document
	return "\n".join(updated_lines)


MARKDOWN_BOLD_PATTERN = re.compile(r"\*\*([^*\n]+)\*\*")
BARE_TEXT_COMMAND_PATTERN = re.compile(r"(?<![\\a-zA-Z])(textbf|textit|texttt)\{")
SMART_QUOTE_TRANSLATION = str.maketrans({
	"“": "``",
	"”": "''",
	"‘": "`",
	"’": "'",
	"„": "``",
})


def _convert_markdown_bold_to_textbf(latex_document: str) -> str:
	"""LLMs slip markdown emphasis into LaTeX output; `**x**` renders as
	literal asterisks in the PDF instead of bold text."""
	return MARKDOWN_BOLD_PATTERN.sub(r"\\textbf{\1}", latex_document)


def _restore_bare_text_command_backslashes(latex_document: str) -> str:
	"""A `textbf{`/`textit{` whose backslash was eaten (JSON escaping, model
	error) renders as literal 'textbf{...}' text in the PDF."""
	return BARE_TEXT_COMMAND_PATTERN.sub(r"\\\1{", latex_document)


def _normalize_smart_quotes(latex_document: str) -> str:
	"""Map Unicode curly quotes to LaTeX quote ligatures so the PDF shows
	proper typography regardless of engine/inputenc handling."""
	return latex_document.translate(SMART_QUOTE_TRANSLATION)


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

	next_text = _fix_html_comment_tags(updated)
	if next_text != updated:
		repairs.append("replaced HTML </comment> tags with \\end{comment}")
		updated = next_text

	next_text = _close_unclosed_comment_environments(updated)
	if next_text != updated:
		repairs.append("closed unclosed \\begin{comment} environments")
		updated = next_text

	next_text = _remove_dangling_single_line_backslashes(updated)
	if next_text != updated:
		repairs.append("removed dangling single trailing backslashes")
		updated = next_text

	next_text = _convert_markdown_bold_to_textbf(updated)
	if next_text != updated:
		repairs.append("converted markdown **bold** to \\textbf")
		updated = next_text

	next_text = _restore_bare_text_command_backslashes(updated)
	if next_text != updated:
		repairs.append("restored missing backslashes on text commands")
		updated = next_text

	next_text = _normalize_smart_quotes(updated)
	if next_text != updated:
		repairs.append("normalized smart quotes to LaTeX quote ligatures")
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
	# Suppressed warnings are template-level false positives on every render:
	# 1 = "command terminated with space" (\raggedright at line end),
	# 8 = "wrong length of dash" (phone numbers, date ranges),
	# 27 = "could not execute LaTeX command" (\input{glyphtounicode}).
	command = [
		chktex_path,
		"-q",
		"-I0",
		"-n1",
		"-n8",
		"-n27",
		"-f%l:%c:%k:%n:%m\\n",
		tex_name,
	]
	try:
		result = subprocess.run(
			command,
			capture_output=True,
			text=True,
			encoding="utf-8",
			errors="replace",
			stdin=subprocess.DEVNULL,  # same reason as _run_latex_command
			timeout=20,
			check=False,
			cwd=str(cwd),
			**platform_support.no_window_kwargs(),
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

	# Remove duplicate preambles before all other processing.
	latex_document = _remove_duplicate_begin_document(latex_document)

	if normalize_json_escaped_latex:
		latex_document = _normalize_escaped_latex_document(latex_document)

	latex_document = _close_runaway_inline_text_commands(latex_document)
	latex_document = _normalize_empty_text_command_small_args(latex_document)
	latex_document = _normalize_broken_item_section_collisions(latex_document)
	latex_document = _close_small_itemize_group_brace(latex_document)
	latex_document = _drop_orphan_lowercase_fragments(latex_document)
	latex_document = _normalize_unclosed_resume_lists(latex_document)

	# Ensure \resumeSubheading always has exactly 4 argument groups.
	latex_document = _fix_resume_subheading_arg_count(latex_document)

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

	# Fix LLM writing </comment> (HTML) instead of \end{comment} (LaTeX environment).
	latex_document = _fix_html_comment_closing_tags(latex_document)

	# Close any \begin{comment} blocks that the LLM left without a matching \end{comment}.
	latex_document = _close_unclosed_comment_environments(latex_document)

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


def _parse_brace_group(text: str, pos: int) -> tuple[str, int] | None:
	"""Parse one {}-delimited group starting at text[pos]. Returns (group_text, end_pos) or None."""
	if pos >= len(text) or text[pos] != "{":
		return None
	depth = 0
	i = pos
	while i < len(text):
		ch = text[i]
		if ch == "\\" and i + 1 < len(text):
			i += 2
			continue
		if ch == "{":
			depth += 1
		elif ch == "}":
			depth -= 1
			if depth == 0:
				return text[pos : i + 1], i + 1
		i += 1
	return None


def _fix_resume_subheading_arg_count(latex_document: str) -> str:
	"""Pad \\resumeSubheading to exactly 4 {{}} groups when the LLM emitted fewer.

	\\resumeSubheading is defined with [4] arguments: {Title}{Dates}{Company}{Location}.
	LLMs often omit the final argument (location), causing LaTeX to consume the
	next brace group as the missing arg and producing broken table layout or a
	runaway-argument error.  When 1-3 groups are detected, empty groups are appended.
	"""
	if "\\resumeSubheading" not in latex_document:
		return latex_document

	result: list[str] = []
	last_end = 0

	for match in RESUME_SUBHEADING_MACRO_PATTERN.finditer(latex_document):
		macro_end = match.end()

		pos = macro_end
		# Skip horizontal whitespace only; a newline ends the call for our purposes.
		while pos < len(latex_document) and latex_document[pos] in (" ", "\t"):
			pos += 1

		args: list[str] = []
		while pos < len(latex_document) and latex_document[pos] == "{":
			parsed = _parse_brace_group(latex_document, pos)
			if parsed is None:
				break
			group, pos = parsed
			args.append(group)
			while pos < len(latex_document) and latex_document[pos] in (" ", "\t"):
				pos += 1

		result.append(latex_document[last_end:macro_end])

		if 1 <= len(args) < 4:
			result.append("".join(args) + "{}" * (4 - len(args)))
			last_end = pos
		else:
			# 0 args (pattern shouldn't fire) or already 4+ args: leave untouched
			last_end = macro_end

	result.append(latex_document[last_end:])
	return "".join(result)


def _remove_duplicate_begin_document(latex_document: str) -> str:
	"""Strip a duplicate preamble when the LLM accidentally regenerated the document.

	Some LLMs restart their output mid-generation, producing two full preambles.
	Having two \\documentclass declarations causes pdflatex to reject the file.
	When multiple non-commented \\documentclass lines are detected, content before
	the last one is discarded.
	"""
	lines = latex_document.splitlines(keepends=True)
	docclass_indices = [
		i
		for i, line in enumerate(lines)
		if "\\documentclass" in line and not line.lstrip().startswith("%")
	]
	if len(docclass_indices) < 2:
		return latex_document
	return "".join(lines[docclass_indices[-1] :])


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
	"""Insert missing list-environment closers before \\end{document}.

	Handles two cases:
	  1. Resume-macro pairs (\\resumeSubHeadingListStart/End, \\resumeItemListStart/End)
	     — the LLM truncated before emitting the closing macro.
	  2. Bare \\begin{itemize} / \\end{itemize} pairs — for LLMs that write explicit
	     environments instead of macros.  In macro-based templates the preamble
	     \\newcommand definitions contribute equally to both sides, so the net
	     explicit-itemize count remains correct.
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
	missing_explicit = (
		latex_document.count("\\begin{itemize}")
		- latex_document.count("\\end{itemize}")
	)

	if missing_item <= 0 and missing_sub <= 0 and missing_explicit <= 0:
		return latex_document

	insert = ""
	if missing_item > 0:
		insert += "\\resumeItemListEnd\n" * missing_item
	if missing_sub > 0:
		insert += "\\resumeSubHeadingListEnd\n" * missing_sub
	if missing_explicit > 0:
		insert += "\\end{itemize}\n" * missing_explicit

	return latex_document.replace("\\end{document}", insert + "\\end{document}", 1)


_HTML_COMMENT_CLOSE_PATTERN = re.compile(r"</comment\s*>", re.IGNORECASE)


def _fix_html_comment_closing_tags(latex_document: str) -> str:
	"""Replace HTML-style </comment> closing tags with \\end{comment}.

	LLMs trained on mixed HTML/LaTeX data sometimes write </comment> instead of
	the correct LaTeX environment closer \\end{comment}, leaving comment blocks
	unclosed and hiding the rest of the document from pdflatex.
	"""
	if "</comment" not in latex_document.lower():
		return latex_document
	return _HTML_COMMENT_CLOSE_PATTERN.sub("\\\\end{comment}", latex_document)


def _close_unclosed_comment_environments(latex_document: str) -> str:
	"""Insert missing \\end{comment} before the next \\begin{comment} or \\end{document}.

	LLMs sometimes omit the closing tag entirely (rather than writing </comment>),
	leaving \\begin{comment} blocks open. This causes pdflatex to fail with
	'File ended while scanning use of \\next'. The fix inserts \\end{comment}
	immediately before any \\begin{comment} or \\end{document} line that would
	otherwise be swallowed by an unclosed block.
	"""
	if r"\begin{comment}" not in latex_document:
		return latex_document
	lines = latex_document.splitlines()
	open_count = 0
	result: list[str] = []
	for line in lines:
		stripped = line.strip()
		if stripped == r"\begin{comment}":
			if open_count > 0:
				result.append(r"\end{comment}")
				open_count = 0
			open_count += 1
			result.append(line)
		elif stripped == r"\end{comment}":
			if open_count > 0:
				open_count -= 1
			result.append(line)
		elif stripped == r"\end{document}" and open_count > 0:
			result.append(r"\end{comment}")
			open_count = 0
			result.append(line)
		else:
			result.append(line)
	if open_count > 0:
		result.append(r"\end{comment}")
	return "\n".join(result)


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
					# Trailing % at end of a line is the standard LaTeX whitespace-
					# suppression idiom used in multi-line macro definitions, e.g.:
					#   \newcommand{\cmd}[4]{%
					#     \vspace{\skip}%
					#     content
					#   }
					# It must NOT be escaped; it is a comment that eats the newline.
					# Only escape % when there is non-whitespace content after it on the
					# same line (i.e. it is a mid-argument literal percent sign).
					rest = line[i + 1 :]
					if not rest.strip():
						out.append(line[i:])
						break
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
	if LATEX_FONT_FAILURE_PATTERN.search(combined):
		return "font-unavailable"
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


def _declared_usepackage_names(latex_document: str) -> list[str]:
	names: list[str] = []
	for match in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", latex_document):
		for raw in match.group(1).split(","):
			name = raw.strip()
			if name and name not in names:
				names.append(name)
	return names


def _font_match_key(text: str) -> str:
	"""Reduce a font or package name to a comparable stem.

	The engine reports a font *file* (``SourceSans3-It-tlf-t1--base``) while the
	preamble names a *package* (``sourcesanspro``).  Lowercasing and dropping
	digits and separators turns the first token of the former into
	``sourcesans``, which is a substring of the latter -- enough to connect them
	without hardcoding a mapping table.
	"""
	return re.sub(r"[^a-z]", "", text.lower())


def _strip_font_packages_for_failed_font(
	latex_document: str,
	stdout_text: str,
	stderr_text: str,
) -> tuple[str, list[str]]:
	"""Drop the font package whose font the engine could not load or build.

	Removing it makes the document fall back to the engine's default typeface.
	The resume then looks different from what its author designed, but it
	compiles and -- verified against a real failing profile -- extracts as clean
	Type 1 text.  A readable resume in the wrong font beats no resume at all.
	"""
	combined = "\n".join(part for part in (stderr_text, stdout_text) if part)
	declared = _declared_usepackage_names(latex_document)
	if not declared:
		return latex_document, []

	targets: list[str] = []
	for match in LATEX_FONT_FILE_PATTERN.finditer(combined):
		# "SourceSans3-It-tlf-t1--base" -> "SourceSans3"
		head = re.split(r"[-_.]", match.group(1).strip())[0]
		key = _font_match_key(head)
		if len(key) < 4:
			continue
		for name in declared:
			if key in _font_match_key(name) and name not in targets:
				targets.append(name)

	if not targets:
		# Nothing matched by name: fall back to dropping any declared package
		# that is a known font package.
		targets = [name for name in declared if _font_match_key(name) in KNOWN_FONT_PACKAGES]

	if not targets:
		return latex_document, []

	updated, removed = _strip_missing_style_usepackages(
		latex_document, [f"{name}.sty" for name in targets]
	)
	if not removed or updated == latex_document:
		return latex_document, []
	removed_label = ", ".join(sorted(set(removed)))
	return updated, [f"dropped unavailable font packages: {removed_label}"]


FONTENC_T1_PATTERN = re.compile(r"^[^%\n]*\\usepackage\s*\[[^\]]*\bT1\b[^\]]*\]\s*\{fontenc\}", re.MULTILINE)
LMODERN_PATTERN = re.compile(r"^[^%\n]*\\usepackage(?:\s*\[[^\]]*\])?\s*\{lmodern\}", re.MULTILINE)


def _ensure_type1_fallback_font(latex_document: str) -> tuple[str, list[str]]:
	"""Give a T1-encoded document a Type 1 font it can actually use.

	``\\usepackage[T1]{fontenc}`` with no T1 Type 1 font installed does not
	fail.  MiKTeX generates EC bitmap (``.pk``) fonts on the fly, so the engine
	exits 0 and every glyph in the PDF becomes a Type 3 bitmap with no
	ToUnicode map -- it prints correctly and extracts as nothing.  Latin Modern
	is the T1-encoded Type 1 companion to the default typeface, so loading it
	keeps the document looking essentially unchanged while making the text
	real, selectable characters.
	"""
	if not FONTENC_T1_PATTERN.search(latex_document) or LMODERN_PATTERN.search(latex_document):
		return latex_document, []

	insertion = "\\usepackage{lmodern}\n"
	match = FONTENC_T1_PATTERN.search(latex_document)
	line_start = latex_document.rfind("\n", 0, match.start()) + 1
	updated = latex_document[:line_start] + insertion + latex_document[line_start:]
	if updated == latex_document:
		return latex_document, []
	return updated, ["loaded lmodern so T1 text renders as Type 1 instead of bitmaps"]


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

	if failure_kind == "font-unavailable":
		updated, repairs = _strip_font_packages_for_failed_font(
			latex_document, stdout_text, stderr_text
		)
		if repairs:
			return updated, repairs
		return _apply_general_latex_repairs(latex_document)

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
			# Engines are constrained to the template's family, so this can fire
			# even when some other engine is installed.  Say which one is needed
			# rather than implying nothing is available -- compiling with the
			# wrong-family engine is what produces unreadable PDFs.
			family = _template_engine_family(sanitized_tex)
			needed = ", ".join(_preferred_latex_engines(sanitized_tex))
			return LatexCompileResult(
				status="unavailable",
				message=(
					f"LaTeX compile environment is not ready: this template needs a "
					f"{family}-family engine ({needed}), which is not installed."
				),
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
			# A compile can succeed and still be unreadable (see the salvage
			# block in the success branch).  At most one salvage per engine.
			bitmap_salvage_done = False
			unreadable_fallback: tuple[bytes, int | None, object, list[str]] | None = None
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

					page_count: int | None = None
					# pdflatex hard-wraps its own stdout at a fixed column width
					# regardless of word boundaries, so a long output filename can
					# split "(1 page, ...)" across a line break (e.g. "...(1 p\nage,
					# ...)"), silently defeating the pattern below. Strip newlines
					# before matching since none are semantically meaningful here.
					flattened_stdout = re.sub(r"\r?\n", "", final_stdout or "")
					page_match = LATEX_PAGE_COUNT_PATTERN.search(flattened_stdout)
					if page_match:
						page_count = int(page_match.group(1))

					pdf_bytes = pdf_path.read_bytes()
					if not pdf_bytes:
						# pdflatex reports "No pages of output" and leaves a
						# zero-byte .pdf behind rather than deleting it, so
						# exists() is not enough.  Returning "ok" here would hand
						# the caller an empty attachment and call it a success.
						last_excerpt = f"{engine_name} produced an empty PDF (no pages of output)."[-1500:]
						break

					# Exit code 0 says the compile ran, not that an ATS can read
					# the result.  Audit the actual bytes.  Report-only for now:
					# the status is recorded and logged, never used to reject a
					# PDF, so this cannot regress a build that works today.
					ats_audit = audit_pdf_ats(pdf_bytes)
					if ats_audit.status != "unknown":
						attempt_logs.append(f"[{engine_name}:ats] {ats_audit.summary()}")

					if unreadable_fallback is not None:
						# This is the render produced *after* dropping the font
						# packages.  If it is no more readable than what the
						# author designed, their typography wins -- a salvage
						# that does not salvage anything is pure loss.
						prev_bytes, prev_pages, prev_audit, prev_labels = unreadable_fallback
						unreadable_fallback = None
						if ATS_QUALITY_RANK.get(ats_audit.status, 1) <= ATS_QUALITY_RANK.get(
							getattr(prev_audit, "status", "unknown"), 1
						):
							attempt_logs.append(
								f"[{engine_name}:ats] dropping font packages did not improve "
								f"readability; keeping the original render"
							)
							pdf_bytes, page_count, ats_audit = prev_bytes, prev_pages, prev_audit
							for label in prev_labels:
								if label in repairs_applied:
									repairs_applied.remove(label)
					elif (
						# Gate on the bitmap signal itself, not on "unreadable".
						# A short resume also audits as unreadable, and dropping
						# the author's font would not add a single character to
						# it -- it would just lose their typography for nothing.
						max(ats_audit.type3_char_ratio, ats_audit.no_tounicode_char_ratio)
						>= TYPE3_UNREADABLE_RATIO
						and not bitmap_salvage_done
						and retry_count < bounded_retries
						and time.monotonic() < deadline
					):
						# When a font package's Type 1 files are not installed,
						# MiKTeX quietly generates PK bitmaps instead of failing.
						# The engine exits 0, the PDF looks perfect, and every
						# glyph is a Type 3 bitmap with no ToUnicode map -- an
						# ATS reads nothing.  There is no error text to classify;
						# the only evidence is in the bytes we just audited.
						bitmap_salvage_done = True
						salvaged_tex, salvage_repairs = _strip_font_packages_for_failed_font(
							engine_tex, "", ""
						)
						salvaged_tex, lmodern_repairs = _ensure_type1_fallback_font(salvaged_tex)
						salvage_repairs = list(salvage_repairs) + lmodern_repairs
						if salvage_repairs and salvaged_tex != engine_tex:
							unreadable_fallback = (
								pdf_bytes, page_count, ats_audit, list(salvage_repairs)
							)
							retry_count += 1
							repair_attempts += 1
							for label in salvage_repairs:
								if label not in repairs_applied:
									repairs_applied.append(label)
							attempt_logs.append(
								f"[{engine_name}:repair#{retry_count}] Applied auto-fixes "
								f"(unreadable-bitmap-fonts): {', '.join(salvage_repairs)}"
							)
							engine_tex = salvaged_tex
							continue

					_write_compile_log(resolved_log_path, "\n\n".join(attempt_logs), "")
					return LatexCompileResult(
						status="ok",
						message=f"Compiled LaTeX resume to PDF using {engine_name}.",
						pdf_bytes=pdf_bytes,
						pdf_name=f"{pdf_stem}.pdf",
						log_path=resolved_log_path,
						repairs_applied=repairs_applied.copy(),
						repair_attempts=repair_attempts,
						lint_findings=lint_findings,
						page_count=page_count,
						ats_status=ats_audit.status,
						ats_findings=list(ats_audit.findings),
						ats_degradations=list(ats_audit.degradations),
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
		# Decode explicitly as UTF-8 with replacement.  text=True uses the
		# locale codec (cp1252 on Windows), which raises UnicodeDecodeError in
		# subprocess's reader thread when an engine echoes a non-ASCII source
		# line.  The exception is swallowed by that thread, so stdout arrives
		# empty and the failure classifier sees nothing to work with.
		return subprocess.run(
			command,
			capture_output=True,
			text=True,
			encoding="utf-8",
			errors="replace",
			# Never inherit stdin.  -interaction=nonstopmode only covers TeX's
			# own error prompts; MiKTeX's package installer is a separate prompt
			# that reads stdin, and an inherited stdin lets it block forever --
			# past this timeout, because the kill-then-drain path waits on a
			# pipe the stalled child still holds.  Observed as a full-suite hang
			# that burned 9.8 CPU-hours in compile_latex_to_pdf.  With DEVNULL
			# the prompt reads EOF and the engine fails fast instead.
			stdin=subprocess.DEVNULL,
			timeout=LATEX_ENGINE_TIMEOUT_SECONDS,
			check=False,
			cwd=str(cwd),
			**platform_support.no_window_kwargs(),
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
