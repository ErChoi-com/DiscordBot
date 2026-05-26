from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

LATEX_WRAPPER_PACKAGE = "pdflatex"
RESUMES_CACHE_ROOT = Path(__file__).resolve().parent / "resumes_cache"
TEMPLATE_PATH = RESUMES_CACHE_ROOT / "template.tex"
PROFILE_ALLOWED_FILE_NAMES = ("baseinfo.txt", "instructions.txt", "template.tex")
EXAMPLE_PROFILE_KEY = "example"
LLM_PROVIDER_SWITCH_ORDER = ("gemini", "groq", "openrouter")
MAX_PROFILE_NAME_PART_LEN = 80
MAX_PROFILE_ID_PART_LEN = 80


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


@dataclass(slots=True)
class LatexCompileEnvironment:
	wrapper_package: str
	wrapper_installed: bool
	pdflatex_path: str | None
	kpsewhich_path: str | None
	required_files: list[str] = field(default_factory=list)
	missing_files: list[str] = field(default_factory=list)

	@property
	def ready(self) -> bool:
		return bool(
			self.wrapper_installed
			and self.pdflatex_path
			and self.kpsewhich_path
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
	return subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)


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

	missing_files: list[str] = []
	if kpsewhich_path:
		for required_file in required_files:
			result = command_runner([kpsewhich_path, required_file])
			if result.returncode != 0 or not result.stdout.strip():
				missing_files.append(required_file)

	return LatexCompileEnvironment(
		wrapper_package=LATEX_WRAPPER_PACKAGE,
		wrapper_installed=wrapper_installed,
		pdflatex_path=pdflatex_path,
		kpsewhich_path=kpsewhich_path,
		required_files=required_files,
		missing_files=missing_files,
	)


def extract_latex_document(text: str) -> str | None:
	match = LATEX_TAG_PATTERN.search(text)
	if match:
		latex_text = match.group(1).strip()
		return latex_text or None

	for fence in LATEX_FENCE_PATTERN.finditer(text):
		candidate = fence.group(1).strip()
		if "\\documentclass" in candidate and "\\end{document}" in candidate:
			return candidate

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

	# Protect double-escaped TeX command leaders (e.g. ``\\newif``) so
	# global ``\\n`` decoding does not turn them into accidental newlines.
	marker = "\uE000"
	normalized = re.sub(r"\\\\(?=[A-Za-z@])", marker, latex_document)

	# Decode JSON-style escaped control characters.
	normalized = normalized.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")

	# Collapse remaining double backslashes (line breaks, escaped literals).
	normalized = normalized.replace("\\\\", "\\")
	return normalized.replace(marker, "\\")


def sanitize_latex_document_for_compile(
	latex_document: str,
	normalize_json_escaped_latex: bool = True,
) -> str:
	"""Escape common model-output mistakes while preserving tabular alignment syntax."""
	if not latex_document:
		return latex_document

	if normalize_json_escaped_latex:
		latex_document = _normalize_escaped_latex_document(latex_document)

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


def compile_latex_to_pdf(
	latex_document: str,
	job_title: str,
	template_path: Path = TEMPLATE_PATH,
	log_path: Path | None = None,
	normalize_json_escaped_latex: bool = True,
	command_runner: Callable[[list[str], Path], subprocess.CompletedProcess[str]] | None = None,
) -> LatexCompileResult:
	environment = check_template_compile_environment(template_path)
	if not environment.ready:
		missing = ", ".join(environment.missing_files) or "pdflatex or kpsewhich not found"
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
		tex_path.write_text(
			sanitize_latex_document_for_compile(
				latex_document,
				normalize_json_escaped_latex=normalize_json_escaped_latex,
			),
			encoding="utf-8",
		)

		command = [
			environment.pdflatex_path or "pdflatex",
			"-interaction=nonstopmode",
			"-halt-on-error",
			tex_path.name,
		]
		result = runner(command, workdir)
		_write_compile_log(resolved_log_path, result.stdout or "", result.stderr or "")
		if result.returncode != 0:
			stderr = (result.stderr or "").strip()
			stdout = (result.stdout or "").strip()
			log_excerpt = (stderr or stdout)[-1500:] if (stderr or stdout) else None
			return LatexCompileResult(
				status="error",
				message="LaTeX compilation failed.",
				log_excerpt=log_excerpt,
				log_path=resolved_log_path,
			)

		pdf_path = tex_path.with_suffix(".pdf")
		if not pdf_path.exists():
			return LatexCompileResult(
				status="error",
				message="LaTeX compilation completed without producing a PDF.",
				log_path=resolved_log_path,
			)

		return LatexCompileResult(
			status="ok",
			message="Compiled LaTeX resume to PDF.",
			pdf_bytes=pdf_path.read_bytes(),
			pdf_name=f"{pdf_stem}.pdf",
			log_path=resolved_log_path,
		)


def _run_latex_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
	return subprocess.run(command, capture_output=True, text=True, timeout=60, check=False, cwd=str(cwd))


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
