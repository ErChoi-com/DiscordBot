import sys
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load_config
from services.resumes.cache import ResumeExplicitCacheManager, load_resume_source_bundle
from services.resumes.configkey import DEFAULT_GEMINI_MODEL, GeminiSettings, load_gemini_settings
from services.resumes.listing import (
    ScrapedJobPosting,
    build_resume_rewrite_prompt,
    extract_job_context_from_message,
    generate_resume_rewrite,
    scrape_job_posting,
)
from services.resumes.resume import (
    check_template_compile_environment,
    compile_latex_to_pdf,
    discord_profile_key,
    ensure_profile_cache,
    extract_latex_document,
    extract_template_requirements,
    migrate_legacy_profile_keys_with_usernames,
    resolve_discord_profile_key,
    sanitize_latex_document_for_compile,
)


def test_extract_job_context_from_watcher_message() -> None:
    message = (
        "[Glassdoor] Junior Python Developer\n"
        "https://example.com/jobs/123\n"
        "Apply: https://example.com/apply/123"
    )

    job = extract_job_context_from_message(message)
    assert job is not None
    assert job.title == "Junior Python Developer"
    assert job.posting_url == "https://example.com/jobs/123"
    assert job.apply_url == "https://example.com/apply/123"


def test_load_gemini_settings_accepts_geminiapi_env(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_AI_STUDIO_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("geminiAPI", "env-key")

    settings = load_gemini_settings()

    assert settings.api_key == "env-key"


def test_load_gemini_settings_uses_config_values() -> None:
    class _Config:
        gemini_api_key = "config-key"
        gemini_model = "gemini-2.5-flash"
        gemini_resume_cache_ttl_seconds = 7200

    settings = load_gemini_settings(_Config())

    assert settings.api_key == "config-key"
    assert settings.model == "gemini-2.5-flash"
    assert settings.resume_cache_ttl_seconds == 7200


def test_load_gemini_settings_prefers_existing_config_key(monkeypatch) -> None:
    class _Config:
        gemini_api_key = "config-key"
        gemini_model = "gemini-2.5-flash"

    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEY", "env-key")

    settings = load_gemini_settings(_Config())

    assert settings.api_key == "config-key"


def test_load_gemini_settings_uses_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEY", "env-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("GEMINI_RESUME_CACHE_TTL_SECONDS", "5400")

    settings = load_gemini_settings()

    assert settings.api_key == "env-key"
    assert settings.model == "gemini-2.5-flash-lite"
    assert settings.resume_cache_ttl_seconds == 5400


def test_load_config_reads_main_user_id(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("MAIN_USER_ID=123456789\n", encoding="utf-8")

    config = load_config(tmp_path)

    assert config.main_user_id == 123456789


def test_load_config_defaults_profile_to_example_even_with_named_owner_profile(tmp_path: Path) -> None:
    cache_root = tmp_path / "src" / "services" / "resumes" / "resumes_cache" / "xboxsignout._"
    cache_root.mkdir(parents=True)
    (cache_root / "baseinfo.txt").write_text("base info", encoding="utf-8")
    (cache_root / "instructions.txt").write_text("instructions", encoding="utf-8")
    (cache_root / "template.tex").write_text("\\documentclass{article}", encoding="utf-8")

    config = load_config(tmp_path)

    assert config.main_user_profile_key == "example"


def test_load_config_defaults_profile_to_example_with_empty_named_owner_folder(tmp_path: Path) -> None:
    cache_root = tmp_path / "src" / "services" / "resumes" / "resumes_cache" / "xboxsignout._"
    cache_root.mkdir(parents=True)

    config = load_config(tmp_path)

    assert config.main_user_profile_key == "example"


def test_extract_job_context_requires_a_url() -> None:
    assert extract_job_context_from_message("[Glassdoor] Python Developer") is None


def test_default_model_constant_is_stable() -> None:
    assert DEFAULT_GEMINI_MODEL == "gemini-2.5-flash"


def test_load_resume_source_bundle_reads_resume_folder(tmp_path: Path) -> None:
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "profile.md").write_text("Candidate profile", encoding="utf-8")

    bundle = load_resume_source_bundle(resumes_dir)

    assert bundle is not None
    combined_text, _digest, source_files = bundle
    assert [path.name for path in source_files] == ["profile.md"]
    assert "Candidate profile" in combined_text


def test_ensure_profile_cache_creates_seeded_user_folder(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    example_dir = cache_root / "example"
    example_dir.mkdir(parents=True)
    (example_dir / "baseinfo.txt").write_text("base info", encoding="utf-8")
    (example_dir / "instructions.txt").write_text("instructions", encoding="utf-8")
    (example_dir / "template.tex").write_text("\\documentclass{article}", encoding="utf-8")

    profile_dir = ensure_profile_cache(12345, cache_root=cache_root)

    assert profile_dir == cache_root / "12345"
    assert (profile_dir / "baseinfo.txt").read_text(encoding="utf-8") == "base info"
    assert (profile_dir / "instructions.txt").read_text(encoding="utf-8") == "instructions"
    assert (profile_dir / "template.tex").read_text(encoding="utf-8") == "\\documentclass{article}"


def test_ensure_profile_cache_ignores_named_owner_when_example_exists(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    owner_dir = cache_root / "xboxsignout._"
    example_dir = cache_root / "example"
    owner_dir.mkdir(parents=True)
    example_dir.mkdir(parents=True)
    (owner_dir / "baseinfo.txt").write_text("owner base info", encoding="utf-8")
    (owner_dir / "instructions.txt").write_text("owner instructions", encoding="utf-8")
    (owner_dir / "template.tex").write_text("owner template", encoding="utf-8")
    (example_dir / "baseinfo.txt").write_text("example base info", encoding="utf-8")
    (example_dir / "instructions.txt").write_text("example instructions", encoding="utf-8")
    (example_dir / "template.tex").write_text("example template", encoding="utf-8")

    profile_dir = ensure_profile_cache(12345, cache_root=cache_root, seed_profile_key="xboxsignout._")

    assert profile_dir == cache_root / "12345"
    assert (profile_dir / "baseinfo.txt").read_text(encoding="utf-8") == "example base info"
    assert (profile_dir / "instructions.txt").read_text(encoding="utf-8") == "example instructions"
    assert (profile_dir / "template.tex").read_text(encoding="utf-8") == "example template"


def test_ensure_profile_cache_prefers_example_folder_over_owner_seed(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    owner_dir = cache_root / "xboxsignout._"
    example_dir = cache_root / "example"
    owner_dir.mkdir(parents=True)
    example_dir.mkdir(parents=True)
    (owner_dir / "baseinfo.txt").write_text("owner base info", encoding="utf-8")
    (owner_dir / "instructions.txt").write_text("owner instructions", encoding="utf-8")
    (owner_dir / "template.tex").write_text("owner template", encoding="utf-8")
    (example_dir / "baseinfo.txt").write_text("example base info", encoding="utf-8")
    (example_dir / "instructions.txt").write_text("example instructions", encoding="utf-8")
    (example_dir / "template.tex").write_text("example template", encoding="utf-8")

    profile_dir = ensure_profile_cache(12345, cache_root=cache_root, seed_profile_key="xboxsignout._")

    assert profile_dir == cache_root / "12345"
    assert (profile_dir / "baseinfo.txt").read_text(encoding="utf-8") == "example base info"
    assert (profile_dir / "instructions.txt").read_text(encoding="utf-8") == "example instructions"
    assert (profile_dir / "template.tex").read_text(encoding="utf-8") == "example template"


def test_ensure_profile_cache_purges_unexpected_entries(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    example_dir = cache_root / "example"
    example_dir.mkdir(parents=True)
    (example_dir / "baseinfo.txt").write_text("base info", encoding="utf-8")
    (example_dir / "instructions.txt").write_text("instructions", encoding="utf-8")
    (example_dir / "template.tex").write_text("\\documentclass{article}", encoding="utf-8")

    profile_dir = cache_root / "12345"
    profile_dir.mkdir()
    (profile_dir / "template.log").write_text("old log", encoding="utf-8")
    (profile_dir / "old.pdf").write_bytes(b"%PDF")
    (profile_dir / "nested").mkdir()

    ensure_profile_cache(12345, cache_root=cache_root)

    assert sorted(path.name for path in profile_dir.iterdir()) == ["baseinfo.txt", "instructions.txt", "template.tex"]


def test_ensure_profile_cache_raises_when_example_missing_required_files(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    example_dir = cache_root / "example"
    example_dir.mkdir(parents=True)
    (example_dir / "baseinfo.txt").write_text("base info", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        ensure_profile_cache(12345, cache_root=cache_root)


def test_ensure_profile_cache_copies_template_as_exact_bytes(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    example_dir = cache_root / "example"
    example_dir.mkdir(parents=True)
    (example_dir / "baseinfo.txt").write_text("base info", encoding="utf-8")
    (example_dir / "instructions.txt").write_text("instructions", encoding="utf-8")

    # Use explicit CRLF and mixed bytes to verify no newline/encoding normalization occurs.
    template_bytes = b"\\documentclass{article}\r\n% keep exact bytes\r\n\\begin{document}\r\n\\end{document}\r\n"
    (example_dir / "template.tex").write_bytes(template_bytes)

    profile_dir = ensure_profile_cache(12345, cache_root=cache_root)

    assert (profile_dir / "template.tex").read_bytes() == template_bytes


def test_ensure_profile_cache_preserves_existing_template(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    example_dir = cache_root / "example"
    example_dir.mkdir(parents=True)
    (example_dir / "baseinfo.txt").write_text("base info", encoding="utf-8")
    (example_dir / "instructions.txt").write_text("instructions", encoding="utf-8")
    (example_dir / "template.tex").write_text("example template", encoding="utf-8")

    profile_dir = cache_root / "12345"
    profile_dir.mkdir()
    (profile_dir / "template.tex").write_text("custom template", encoding="utf-8")

    ensure_profile_cache(12345, cache_root=cache_root)

    assert (profile_dir / "template.tex").read_text(encoding="utf-8") == "custom template"


def test_discord_profile_key_uses_username_prefix() -> None:
    assert discord_profile_key(12345, "Ernest Choi") == "ernest-choi"


def test_discord_profile_key_truncates_long_username_component() -> None:
    long_username = "A" * 400

    key = discord_profile_key(12345, long_username)

    assert len(key) <= 80


def test_discord_profile_key_preserves_trailing_dot_underscore() -> None:
    assert discord_profile_key(12345, "xboxsignout._") == "xboxsignout._"


def test_resolve_discord_profile_key_migrates_legacy_numeric_dir(tmp_path: Path) -> None:

    def test_resolve_discord_profile_key_fails_if_username_folder_missing(tmp_path: Path) -> None:
        cache_root = tmp_path / "resumes_cache"
        # No folder for the username-based key
        with pytest.raises(FileNotFoundError) as excinfo:
            resolve_discord_profile_key(555, "Alpha User", cache_root)
        assert "Resume cache for username" in str(excinfo.value)
        # Create a legacy numeric folder, should still fail
        legacy_dir = cache_root / "555"
        legacy_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            resolve_discord_profile_key(555, "Alpha User", cache_root)
    cache_root = tmp_path / "resumes_cache"
    legacy_dir = cache_root / "12345"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "baseinfo.txt").write_text("base", encoding="utf-8")

    key = resolve_discord_profile_key(12345, "Ernest Choi", cache_root)

    assert key == "ernest-choi"
    assert not legacy_dir.exists()
    assert (cache_root / key / "baseinfo.txt").read_text(encoding="utf-8") == "base"


def test_resolve_discord_profile_key_reuses_existing_username_based_dir_on_rename(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    old_key = "oldname-12345"
    old_dir = cache_root / old_key
    old_dir.mkdir(parents=True)
    (old_dir / "baseinfo.txt").write_text("base", encoding="utf-8")

    key = resolve_discord_profile_key(12345, "New Name", cache_root)

    assert key == "new-name"
    assert not old_dir.exists()
    assert (cache_root / key / "baseinfo.txt").read_text(encoding="utf-8") == "base"


def test_resolve_discord_profile_key_uses_existing_username_based_dir_when_preferred_collides_with_file(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    existing_dir = cache_root / "legacy-name-12345"
    existing_dir.mkdir(parents=True)
    preferred_key = "new-name"
    preferred_path = cache_root / preferred_key
    preferred_path.parent.mkdir(parents=True, exist_ok=True)
    preferred_path.write_text("collision", encoding="utf-8")

    key = resolve_discord_profile_key(12345, "New Name", cache_root)

    assert key == "legacy-name"


def test_migrate_legacy_profile_keys_with_usernames_renames_numeric_and_id_suffix_dirs(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    numeric_dir = cache_root / "555"
    suffixed_dir = cache_root / "old-user-777"
    numeric_dir.mkdir(parents=True)
    suffixed_dir.mkdir(parents=True)

    mapping = {555: "Alpha User", 777: "Beta.User"}
    migrations = migrate_legacy_profile_keys_with_usernames(mapping, cache_root)

    assert (cache_root / "alpha-user").exists()
    assert (cache_root / "beta.user").exists()
    assert sorted(migrations) == [("555", "alpha-user"), ("old-user-777", "beta.user")]


def test_migrate_legacy_profile_keys_with_usernames_skips_when_target_exists(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    legacy_dir = cache_root / "old-user-777"
    target_dir = cache_root / "beta"
    legacy_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)

    migrations = migrate_legacy_profile_keys_with_usernames({777: "beta"}, cache_root)

    assert migrations == []
    assert legacy_dir.exists()
    assert target_dir.exists()


class _FakeCache:
    def __init__(self, name: str, expire_time: datetime | None = None) -> None:
        self.name = name
        self.expire_time = expire_time or datetime(2030, 1, 1, tzinfo=timezone.utc)


class _FakeCachesApi:
    def __init__(self) -> None:
        self.created: list[tuple[str, object]] = []
        self.updated: list[tuple[str, object]] = []
        self.deleted: list[str] = []
        self.remote: dict[str, _FakeCache] = {}

    def create(self, model: str, config: object) -> _FakeCache:
        cache = _FakeCache(name=f"cachedContents/{len(self.created) + 1}")
        self.created.append((model, config))
        self.remote[cache.name] = cache
        return cache

    def get(self, name: str) -> _FakeCache:
        return self.remote[name]

    def update(self, name: str, config: object) -> _FakeCache:
        self.updated.append((name, config))
        return self.remote[name]

    def delete(self, name: str) -> None:
        self.deleted.append(name)
        self.remote.pop(name, None)


class _FakeClient:
    def __init__(self, caches_api: _FakeCachesApi) -> None:
        self.caches = caches_api


def test_resume_cache_manager_creates_local_dirs_and_record(tmp_path: Path) -> None:
    caches_api = _FakeCachesApi()
    manager = ResumeExplicitCacheManager(
        profiles_dir=tmp_path / "resumes",
        cache_dir=tmp_path / ".resume_cache",
        client_factory=lambda api_key: _FakeClient(caches_api),
    )
    (tmp_path / "resumes").mkdir()
    (tmp_path / "resumes" / "profile.md").write_text("Senior Python developer", encoding="utf-8")

    status = manager.ensure_cache(GeminiSettings(api_key="test-key", model="gemini-2.5-flash"))

    assert status.status == "created"
    assert manager.profiles_dir.exists()
    assert manager.cache_dir.exists()
    assert manager.record_path.exists()
    payload = json.loads(manager.record_path.read_text(encoding="utf-8"))
    assert payload["cache_name"] == "cachedContents/1"
    assert payload["source_files"] == ["profile.md"]
    assert len(caches_api.created) == 1


def test_resume_cache_manager_reuses_existing_remote_cache(tmp_path: Path) -> None:
    caches_api = _FakeCachesApi()
    manager = ResumeExplicitCacheManager(
        profiles_dir=tmp_path / "resumes",
        cache_dir=tmp_path / ".resume_cache",
        client_factory=lambda api_key: _FakeClient(caches_api),
    )
    (tmp_path / "resumes").mkdir()
    (tmp_path / "resumes" / "profile.md").write_text("Senior Python developer", encoding="utf-8")

    first = manager.ensure_cache(GeminiSettings(api_key="test-key", model="gemini-2.5-flash"))
    second = manager.ensure_cache(GeminiSettings(api_key="test-key", model="gemini-2.5-flash"))

    assert first.status == "created"
    assert second.status == "reused"
    assert second.cache_name == "cachedContents/1"
    assert len(caches_api.created) == 1


def test_resume_cache_manager_reports_missing_profile(tmp_path: Path) -> None:
    manager = ResumeExplicitCacheManager(
        profiles_dir=tmp_path / "resumes",
        cache_dir=tmp_path / ".resume_cache",
        client_factory=lambda api_key: _FakeClient(_FakeCachesApi()),
    )

    status = manager.ensure_cache(GeminiSettings(api_key="test-key", model="gemini-2.5-flash"))

    assert status.status == "missing-profile"
    assert manager.profiles_dir.exists()
    assert manager.cache_dir.exists()


def test_extract_template_requirements_collects_class_packages_and_inputs() -> None:
    template = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage{geometry,hyperref}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\input{glyphtounicode}\n"
    )

    requirements = extract_template_requirements(template)

    assert requirements == ["article.cls", "fontenc.sty", "geometry.sty", "glyphtounicode.tex", "hyperref.sty"]


def test_check_template_compile_environment_reports_missing_binary(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    monkeypatch.setattr("services.resumes.resume.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("services.resumes.resume._find_latex_executable", lambda name: None)

    status = check_template_compile_environment(template_path)

    assert status.wrapper_installed is True
    assert status.pdflatex_path is None
    assert status.kpsewhich_path is None
    assert status.ready is False
    assert status.required_files == ["article.cls"]


def test_check_template_compile_environment_detects_missing_tex_files(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text(
        "\\documentclass{article}\n"
        "\\usepackage{XCharter,geometry}\n"
        "\\input{glyphtounicode}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("services.resumes.resume.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(
        "services.resumes.resume.shutil.which",
        lambda name: f"C:/tex/{name}.exe" if name in {"pdflatex", "kpsewhich"} else None,
    )

    def _runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        required_file = command[-1]
        if required_file == "XCharter.sty":
            return subprocess.CompletedProcess(command, 1, "", "not found")
        return subprocess.CompletedProcess(command, 0, f"C:/texmf/{required_file}\n", "")

    status = check_template_compile_environment(template_path, command_runner=_runner)

    assert status.wrapper_installed is True
    assert status.pdflatex_path == "C:/tex/pdflatex.exe"
    assert status.kpsewhich_path == "C:/tex/kpsewhich.exe"
    assert status.missing_files == ["XCharter.sty"]
    assert status.ready is False


def test_extract_latex_document_reads_tagged_block() -> None:
    text = "summary text <latex>\\documentclass{article}\\begin{document}Hi\\end{document}</latex> more text"

    latex = extract_latex_document(text)

    assert latex == "\\documentclass{article}\\begin{document}Hi\\end{document}"


def test_extract_latex_document_reads_fenced_tex_block() -> None:
    text = """analysis\n```tex\n\\documentclass{article}\n\\begin{document}\nHi\n\\end{document}\n```"""

    latex = extract_latex_document(text)

    assert latex is not None
    assert "\\documentclass{article}" in latex
    assert "\\end{document}" in latex


def test_extract_latex_document_reads_inline_document() -> None:
    text = "Intro text \\documentclass{article}\\begin{document}Hi\\end{document} trailing"

    latex = extract_latex_document(text)

    assert latex == "\\documentclass{article}\\begin{document}Hi\\end{document}"


def test_compile_latex_to_pdf_returns_pdf_bytes(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    class _ReadyEnvironment:
        ready = True
        pdflatex_path = "C:/tex/pdflatex.exe"

    monkeypatch.setattr("services.resumes.resume.check_template_compile_environment", lambda path: _ReadyEnvironment())

    def _runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        tex_name = command[-1]
        pdf_path = cwd / Path(tex_name).with_suffix(".pdf")
        pdf_path.write_bytes(b"%PDF-1.4\n")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    result = compile_latex_to_pdf(
        latex_document="\\documentclass{article}\\begin{document}Hi\\end{document}",
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
    )

    assert result.status == "ok"
    assert result.pdf_bytes == b"%PDF-1.4\n"
    assert result.pdf_name == "Python-Developer.pdf"
    assert result.log_path == tmp_path / "template.log"


def test_sanitize_latex_document_for_compile_escapes_ampersands_outside_tabular() -> None:
    source = (
        "\\section{Profile}\n"
        "Data & Analytics summary\n"
        "\\begin{tabular}{lr}\n"
        "A & B \\\\\n"
        "\\end{tabular}\n"
        "Already escaped \\& should remain."
    )

    sanitized = sanitize_latex_document_for_compile(source)

    assert "Data \\& Analytics summary" in sanitized
    assert "A & B" in sanitized
    assert "Already escaped \\& should remain." in sanitized


def test_sanitize_latex_document_for_compile_normalizes_json_escaped_latex() -> None:
    source = "\\\\documentclass{article}\\n\\\\begin{document}\\nResearch & Analytics\\n\\\\end{document}"

    sanitized = sanitize_latex_document_for_compile(source)

    assert sanitized.startswith("\\documentclass{article}\n")
    assert "\\begin{document}" in sanitized
    assert "\\end{document}" in sanitized
    assert "Research \\& Analytics" in sanitized
    assert "\\n" not in sanitized


def test_sanitize_latex_document_for_compile_can_skip_json_unescape() -> None:
    source = "\\\\documentclass{article}\\n\\\\begin{document}\\nResearch & Analytics\\n\\\\end{document}"

    sanitized = sanitize_latex_document_for_compile(
        source,
        normalize_json_escaped_latex=False,
    )

    assert "\\\\documentclass{article}" in sanitized
    assert "\\n" in sanitized


def test_sanitize_latex_document_for_compile_does_not_corrupt_newif_in_raw_tex() -> None:
    source = "\\newif\\ifhastitlesec\n\\ifhastitlesectrue"

    sanitized = sanitize_latex_document_for_compile(source)

    assert "\\newif\\ifhastitlesec" in sanitized
    assert "\newif" not in sanitized


def test_sanitize_latex_document_for_compile_normalizes_escaped_newif_payload() -> None:
    source = "\\\\documentclass{article}\\n\\\\newif\\\\ifhastitlesec\\n\\\\begin{document}\\nOK\\n\\\\end{document}"

    sanitized = sanitize_latex_document_for_compile(source)

    assert "\\documentclass{article}" in sanitized
    assert "\\newif\\ifhastitlesec" in sanitized
    assert "\newif" not in sanitized


def test_sanitize_latex_document_for_compile_handles_mixed_escaped_embedded_tokens() -> None:
    """Mixed escaped/unescaped LaTeX with embedded \n tokens in URLs (actual failure case)."""
    source = "\\documentclass{article}\n\\usepackage{hyperref}\n\\href{https://example.com/Particle-Fluid-Simulation\\n}{Project}\n\\end{document}"

    sanitized = sanitize_latex_document_for_compile(source)

    # The \n token should be decoded to actual newline
    assert "Particle-Fluid-Simulation\n" in sanitized or "Particle-Fluid-Simulation}" in sanitized
    # Make sure it's not left as literal \n
    assert "Particle-Fluid-Simulation\\n" not in sanitized


def test_sanitize_latex_document_for_compile_does_not_decode_whitespace_prefixed_macro_n() -> None:
    source = "\\documentclass{article}\n\\begin{document}\nToken: \\n{}\n\\end{document}"

    sanitized = sanitize_latex_document_for_compile(source)

    # Keep explicit macro-like control sequence untouched in non-escaped docs.
    assert "Token: \\n{}" in sanitized

def test_compile_latex_to_pdf_sanitizes_unescaped_ampersands(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    class _ReadyEnvironment:
        ready = True
        pdflatex_path = "C:/tex/pdflatex.exe"

    monkeypatch.setattr("services.resumes.resume.check_template_compile_environment", lambda path: _ReadyEnvironment())

    captured_tex: dict[str, str] = {}

    def _runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        tex_name = command[-1]
        tex_path = cwd / tex_name
        captured_tex["value"] = tex_path.read_text(encoding="utf-8")
        pdf_path = cwd / Path(tex_name).with_suffix(".pdf")
        pdf_path.write_bytes(b"%PDF-1.4\n")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    result = compile_latex_to_pdf(
        latex_document="\\documentclass{article}\\begin{document}Research & Analytics\\end{document}",
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
    )

    assert result.status == "ok"
    assert "Research \\& Analytics" in captured_tex["value"]


class _FakeModelsApi:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)

        class _Response:
            text = self.response_text

        return _Response()


class _FakeGeminiClient:
    def __init__(self, models_api: _FakeModelsApi) -> None:
        self.models = models_api


def test_build_resume_rewrite_prompt_includes_job_and_baseinfo() -> None:
    job = extract_job_context_from_message("[Indeed] Python Developer\nhttps://example.com/job")
    assert job is not None

    scraped = ScrapedJobPosting(
        title="Python Developer - Example Co",
        company="Example Co",
        location="Toronto, ON",
        description="We are hiring a Python developer with API and SQL experience.",
        highlights=["3+ years Python experience", "Build REST APIs"],
        source_url="https://example.com/job",
    )
    prompt = build_resume_rewrite_prompt(job, scraped, "base info text", "latex instructions")

    assert "Requested title: Python Developer" in prompt
    assert "Company: Example Co" in prompt
    assert "base info text" in prompt
    assert "<latex>...</latex>" in prompt


def test_generate_resume_rewrite_uses_cache_name_when_available(tmp_path: Path) -> None:
    job = extract_job_context_from_message("[LinkedIn] Software Engineer\nhttps://example.com/job")
    assert job is not None
    baseinfo_file = tmp_path / "baseinfo.txt"
    baseinfo_file.write_text("Candidate has Python and backend experience.", encoding="utf-8")

    models_api = _FakeModelsApi('{"summary":"ok","targeted_bullets":["a"],"revised_profile":"b"}')
    fake_client = _FakeGeminiClient(models_api)

    result = generate_resume_rewrite(
        settings=GeminiSettings(api_key="test-key", model="gemini-2.5-flash"),
        job=job,
        cache_name="cachedContents/123",
        baseinfo_paths=[baseinfo_file],
        support_paths=[],
        template_path=tmp_path / "template.tex",
        scraper=lambda _: ScrapedJobPosting(
            title="Software Engineer",
            company="Example",
            location="Remote",
            description="Requirements: Python and SQL",
            highlights=["Requirements: Python and SQL"],
            source_url="https://example.com/job",
        ),
        client_factory=lambda _: fake_client,
    )

    assert result.status == "ok"
    assert result.rewritten_resume is not None
    assert len(models_api.calls) == 1
    assert models_api.calls[0].get("model") == "models/gemini-2.5-flash"
    assert "config" in models_api.calls[0]


def test_generate_resume_rewrite_extracts_latex_block(tmp_path: Path) -> None:
    job = extract_job_context_from_message("[LinkedIn] Software Engineer\nhttps://example.com/job")
    assert job is not None
    models_api = _FakeModelsApi("<latex>\\documentclass{article}\\begin{document}Hi\\end{document}</latex>")
    fake_client = _FakeGeminiClient(models_api)

    result = generate_resume_rewrite(
        settings=GeminiSettings(api_key="test-key", model="gemini-2.5-flash"),
        job=job,
        cache_name=None,
        baseinfo_paths=[],
        support_paths=[],
        template_path=tmp_path / "template.tex",
        scraper=lambda _: ScrapedJobPosting(
            title="Software Engineer",
            company="Example",
            location="Remote",
            description="Requirements: Python and SQL",
            highlights=["Requirements: Python and SQL"],
            source_url="https://example.com/job",
        ),
        client_factory=lambda _: fake_client,
    )

    assert result.status == "ok"
    assert result.latex_document == "\\documentclass{article}\\begin{document}Hi\\end{document}"


def test_generate_resume_rewrite_prefers_rewritten_tex_field(tmp_path: Path) -> None:
    job = extract_job_context_from_message("[LinkedIn] Software Engineer\nhttps://example.com/job")
    assert job is not None
    payload = {
        "summary": "ok",
        "targeted_bullets": ["a"],
        "revised_profile": "b",
        "rewritten_tex": "\\documentclass{article}\\n\\begin{document}Hi\\n\\end{document}",
    }
    models_api = _FakeModelsApi(json.dumps(payload))
    fake_client = _FakeGeminiClient(models_api)

    result = generate_resume_rewrite(
        settings=GeminiSettings(api_key="test-key", model="gemini-2.5-flash"),
        job=job,
        cache_name=None,
        baseinfo_paths=[],
        support_paths=[],
        template_path=tmp_path / "template.tex",
        scraper=lambda _: ScrapedJobPosting(
            title="Software Engineer",
            company="Example",
            location="Remote",
            description="Requirements: Python and SQL",
            highlights=["Requirements: Python and SQL"],
            source_url="https://example.com/job",
        ),
        client_factory=lambda _: fake_client,
    )

    assert result.status == "ok"
    assert result.latex_document is not None
    assert "\\documentclass" in result.latex_document


def test_scrape_job_posting_falls_back_to_job_service_on_http_error(monkeypatch) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            raise requests.HTTPError("403 Client Error: Forbidden")

    monkeypatch.setattr("services.resumes.listing.requests.get", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(
        "services.resumes.listing.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.RequestException("graphql unavailable")),
    )
    monkeypatch.setattr("services.resumes.listing.job_site_from_url", lambda url: "indeed")
    monkeypatch.setattr(
        "services.resumes.listing.scrape_jobs_from_board_url",
        lambda url, max_items=5: [
            {
                "title": "Co-op/ Intern Engineer, Data & Analytics",
                "company": "Kinaxis",
                "location": "Ottawa, ON, CA",
                "apply_link": "https://example.com/apply",
                "contact_emails": ["jobs@example.com"],
                "contact_phones": ["(111) 222-3333"],
            }
        ],
    )

    result = scrape_job_posting("https://ca.indeed.com/viewjob?jk=abc")

    assert result.title == "Co-op/ Intern Engineer, Data & Analytics"
    assert result.company == "Kinaxis"
    assert result.location == "Ottawa, ON, CA"
    assert "Apply link:" in result.description
    assert result.source_url == "https://ca.indeed.com/viewjob?jk=abc"


def test_scrape_job_posting_raises_when_fallback_unavailable(monkeypatch) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            raise requests.HTTPError("403 Client Error: Forbidden")

    monkeypatch.setattr("services.resumes.listing.requests.get", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(
        "services.resumes.listing.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.RequestException("graphql unavailable")),
    )
    monkeypatch.setattr("services.resumes.listing.job_site_from_url", lambda url: None)

    try:
        scrape_job_posting("https://example.com/posting")
        assert False, "Expected scrape_job_posting to raise requests.HTTPError"
    except requests.HTTPError:
        pass


def test_scrape_job_posting_falls_back_to_exported_descriptions_when_board_scrape_is_empty(monkeypatch) -> None:
    class _Response:
        def raise_for_status(self) -> None:
            raise requests.HTTPError("403 Client Error: Forbidden")

    monkeypatch.setattr("services.resumes.listing.requests.get", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(
        "services.resumes.listing.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.RequestException("graphql unavailable")),
    )
    monkeypatch.setattr("services.resumes.listing.job_site_from_url", lambda url: "indeed")
    monkeypatch.setattr("services.resumes.listing.scrape_jobs_from_board_url", lambda url, max_items=5: [])
    monkeypatch.setattr(
        "services.resumes.listing.scrape_job_descriptions_from_all_sites",
        lambda **kwargs: [
            {
                "site": "indeed",
                "title": "Software Intern",
                "description": "Posting URL: https://ca.indeed.com/viewjob?jk=abc\nDescription: Build APIs.",
                "link": "https://ca.indeed.com/viewjob?jk=abc",
            }
        ],
    )

    result = scrape_job_posting("https://ca.indeed.com/viewjob?jk=abc")

    assert result.title == "Software Intern"
    assert "Build APIs." in result.description
    assert result.source_url == "https://ca.indeed.com/viewjob?jk=abc"


def test_scrape_job_posting_uses_indeed_graphql_fallback_when_available(monkeypatch) -> None:
    class _GetResponse:
        def raise_for_status(self) -> None:
            raise requests.HTTPError("403 Client Error: Forbidden")

    class _PostResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": {
                    "jobData": {
                        "title": "Co-op/ Intern Engineer, Data & Analytics",
                        "sourceEmployerName": "Kinaxis",
                        "jobLocation": {"long": {"label": "Ottawa, ON, CA"}},
                        "description": {
                            "text": "Responsibilities include Python, SQL, and data modeling in an agile team."
                        },
                    }
                }
            }

    monkeypatch.setattr("services.resumes.listing.requests.get", lambda *args, **kwargs: _GetResponse())
    monkeypatch.setattr("services.resumes.listing.requests.post", lambda *args, **kwargs: _PostResponse())
    monkeypatch.setattr(
        "services.resumes.listing._fallback_scrape_job_posting_with_job_service",
        lambda posting_url: (_ for _ in ()).throw(AssertionError("job service fallback should not be called")),
    )

    result = scrape_job_posting("https://ca.indeed.com/viewjob?jk=abc123")

    assert result.title == "Co-op/ Intern Engineer, Data & Analytics"
    assert result.company == "Kinaxis"
    assert result.location == "Ottawa, ON, CA"
    assert "Python, SQL" in result.description
    assert result.source_url == "https://ca.indeed.com/viewjob?jk=abc123"


# ── Provider fallback verification ────────────────────────────────────────────


def test_load_config_reads_groq_api_key(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("groqAPI=gsk_testkey123\n", encoding="utf-8")

    config = load_config(tmp_path)

    assert config.groq_api_key == "gsk_testkey123"


def test_load_config_reads_groq_api_key_env_var_name(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("GROQ_API_KEY=gsk_standard_key\n", encoding="utf-8")

    config = load_config(tmp_path)

    assert config.groq_api_key == "gsk_standard_key"


def test_load_gemini_settings_reads_openrouter_and_groq_from_config() -> None:
    class _Config:
        gemini_api_key = None
        gemini_model = "gemini-2.5-flash"
        gemini_resume_cache_ttl_seconds = 86400
        openrouter_key = "or-key-from-config"
        groq_api_key = "groq-key-from-config"

    settings = load_gemini_settings(_Config())

    assert settings.openrouter_api_key == "or-key-from-config"
    assert settings.groq_api_key == "groq-key-from-config"


def test_load_gemini_settings_falls_back_to_env_for_openrouter_and_groq(monkeypatch) -> None:
    for key in ("OPENROUTER_API_KEY", "openRouter", "GROQ_API_KEY", "groqAPI",
                "GOOGLE_AI_STUDIO_API_KEY", "GEMINI_API_KEY", "geminiAPI"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("openRouter", "or-env-key")
    monkeypatch.setenv("groqAPI", "groq-env-key")

    settings = load_gemini_settings()

    assert settings.openrouter_api_key == "or-env-key"
    assert settings.groq_api_key == "groq-env-key"


def _make_openai_compat_response(content: str) -> object:
    class _R:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": content}}]}

    return _R()


def test_generate_resume_rewrite_falls_back_to_openrouter_when_no_gemini_key(
    monkeypatch, tmp_path: Path
) -> None:
    job = extract_job_context_from_message("[LinkedIn] Software Engineer\nhttps://example.com/job")
    assert job is not None

    settings = GeminiSettings(
        api_key=None,
        model="gemini-2.5-flash",
        openrouter_api_key="or-test-key",
    )

    import services.resumes.listing as _lm
    from services.resumes.listing import OPENROUTER_CHAT_COMPLETIONS_URL

    post_calls: list[str] = []

    def _mock_post(url: str, **kwargs: object) -> object:
        post_calls.append(url)
        return _make_openai_compat_response(
            json.dumps({"summary": "ok", "targeted_bullets": ["a"], "revised_profile": "b"})
        )

    monkeypatch.setattr(_lm.requests, "post", _mock_post)

    result = generate_resume_rewrite(
        settings=settings,
        job=job,
        cache_name=None,
        baseinfo_paths=[],
        support_paths=[],
        template_path=tmp_path / "template.tex",
        scraper=lambda _: ScrapedJobPosting(
            title="Software Engineer",
            company="Example",
            location="Remote",
            description="Requirements: Python and SQL",
            highlights=["Requirements: Python and SQL"],
            source_url="https://example.com/job",
        ),
    )

    assert result.status == "ok", result.message
    assert "openrouter" in result.message
    assert len(post_calls) == 1
    assert post_calls[0] == OPENROUTER_CHAT_COMPLETIONS_URL


def test_generate_resume_rewrite_falls_back_to_groq_when_no_gemini_or_openrouter_key(
    monkeypatch, tmp_path: Path
) -> None:
    job = extract_job_context_from_message("[LinkedIn] Software Engineer\nhttps://example.com/job")
    assert job is not None

    settings = GeminiSettings(
        api_key=None,
        model="gemini-2.5-flash",
        groq_api_key="groq-test-key",
    )

    import services.resumes.listing as _lm
    from services.resumes.listing import GROQ_CHAT_COMPLETIONS_URL

    # Suppress env/dotenv so the OpenRouter candidate truly has no key.
    monkeypatch.setattr(_lm, "_openrouter_api_key", lambda: None)

    post_calls: list[str] = []

    def _mock_post(url: str, **kwargs: object) -> object:
        post_calls.append(url)
        return _make_openai_compat_response(
            json.dumps({"summary": "ok", "targeted_bullets": ["a"], "revised_profile": "b"})
        )

    monkeypatch.setattr(_lm.requests, "post", _mock_post)

    result = generate_resume_rewrite(
        settings=settings,
        job=job,
        cache_name=None,
        baseinfo_paths=[],
        support_paths=[],
        template_path=tmp_path / "template.tex",
        scraper=lambda _: ScrapedJobPosting(
            title="Software Engineer",
            company="Example",
            location="Remote",
            description="Requirements: Python and SQL",
            highlights=["Requirements: Python and SQL"],
            source_url="https://example.com/job",
        ),
    )

    assert result.status == "ok", result.message
    assert "groq" in result.message
    assert len(post_calls) == 1
    assert post_calls[0] == GROQ_CHAT_COMPLETIONS_URL


def test_generate_resume_rewrite_reports_error_when_all_providers_fail(
    monkeypatch, tmp_path: Path
) -> None:
    job = extract_job_context_from_message("[LinkedIn] Software Engineer\nhttps://example.com/job")
    assert job is not None

    settings = GeminiSettings(
        api_key="gemini-key",
        model="gemini-2.5-flash",
        openrouter_api_key="or-key",
        groq_api_key="groq-key",
    )

    import services.resumes.listing as _lm

    monkeypatch.setattr(_lm.requests, "post", lambda *a, **kw: (_ for _ in ()).throw(OSError("network down")))

    result = generate_resume_rewrite(
        settings=settings,
        job=job,
        cache_name=None,
        baseinfo_paths=[],
        support_paths=[],
        template_path=tmp_path / "template.tex",
        scraper=lambda _: ScrapedJobPosting(
            title="Software Engineer",
            company="Example",
            location="Remote",
            description="Requirements: Python and SQL",
            highlights=["Requirements: Python and SQL"],
            source_url="https://example.com/job",
        ),
        client_factory=lambda api_key: (_ for _ in ()).throw(RuntimeError("Gemini SDK unavailable")),
    )

    assert result.status == "error"
    assert "All resume providers failed" in result.message
