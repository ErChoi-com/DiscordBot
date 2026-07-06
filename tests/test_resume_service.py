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
    lint_latex_document_for_compile,
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
    assert DEFAULT_GEMINI_MODEL == "gemini-3.5-flash"


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

    profile_dir = ensure_profile_cache(12345, cache_root=cache_root)

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


def test_resolve_discord_profile_key_fails_if_username_folder_missing(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_discord_profile_key(555, "Alpha User", cache_root)
    assert "Resume cache for username" in str(excinfo.value)

    # Legacy numeric folders are intentionally ignored.
    legacy_dir = cache_root / "555"
    legacy_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        resolve_discord_profile_key(555, "Alpha User", cache_root)


def test_resolve_discord_profile_key_returns_username_folder_when_present(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    preferred_dir = cache_root / "ernest-choi"
    preferred_dir.mkdir(parents=True)
    (preferred_dir / "baseinfo.txt").write_text("base", encoding="utf-8")

    key = resolve_discord_profile_key(12345, "Ernest Choi", cache_root)

    assert key == "ernest-choi"
    assert (cache_root / key / "baseinfo.txt").read_text(encoding="utf-8") == "base"


def test_resolve_discord_profile_key_rejects_legacy_id_suffix_folder(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    old_key = "oldname-12345"
    old_dir = cache_root / old_key
    old_dir.mkdir(parents=True)
    (old_dir / "baseinfo.txt").write_text("base", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        resolve_discord_profile_key(12345, "New Name", cache_root)

    assert old_dir.exists()


def test_resolve_discord_profile_key_fails_when_preferred_key_collides_with_file(tmp_path: Path) -> None:
    cache_root = tmp_path / "resumes_cache"
    existing_dir = cache_root / "legacy-name-12345"
    existing_dir.mkdir(parents=True)
    preferred_key = "new-name"
    preferred_path = cache_root / preferred_key
    preferred_path.parent.mkdir(parents=True, exist_ok=True)
    preferred_path.write_text("collision", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        resolve_discord_profile_key(12345, "New Name", cache_root)


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


def test_check_template_compile_environment_tolerates_missing_kpsewhich_when_pdflatex_exists(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    monkeypatch.setattr("services.resumes.resume.importlib.util.find_spec", lambda name: object())

    def _find_exec(name: str) -> str | None:
        if name == "pdflatex":
            return "C:/tex/pdflatex.exe"
        return None

    monkeypatch.setattr("services.resumes.resume._find_latex_executable", _find_exec)

    status = check_template_compile_environment(template_path)

    assert status.pdflatex_path == "C:/tex/pdflatex.exe"
    assert status.kpsewhich_path is None
    assert status.missing_files == []
    assert status.ready is True


def test_check_template_compile_environment_marks_missing_when_kpsewhich_times_out(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    monkeypatch.setattr("services.resumes.resume.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(
        "services.resumes.resume._find_latex_executable",
        lambda name: f"C:/tex/{name}.exe" if name in {"pdflatex", "kpsewhich"} else None,
    )

    calls: list[list[str]] = []

    def _runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        raise subprocess.TimeoutExpired(command, timeout=15)

    status = check_template_compile_environment(template_path, command_runner=_runner)

    # Probe should fail open and stop further kpsewhich checks to avoid repeated delays.
    assert status.kpsewhich_path is None
    assert status.missing_files == []
    assert status.ready is True
    assert len(calls) == 1


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


def test_compile_latex_to_pdf_continues_when_missing_style_is_reported(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    class _PreflightEnvironment:
        ready = False
        pdflatex_path = "C:/tex/pdflatex.exe"
        xelatex_path = None
        lualatex_path = None
        missing_files = ["sourcesanspro.sty"]

    monkeypatch.setattr("services.resumes.resume.check_template_compile_environment", lambda path: _PreflightEnvironment())

    captured_tex: dict[str, str] = {}

    def _runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        tex_name = command[-1]
        tex_path = cwd / tex_name
        captured_tex["content"] = tex_path.read_text(encoding="utf-8")
        pdf_path = cwd / Path(tex_name).with_suffix(".pdf")
        pdf_path.write_bytes(b"%PDF-1.4\n")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    result = compile_latex_to_pdf(
        latex_document="\\documentclass{article}\n\\usepackage{sourcesanspro}\n\\begin{document}Hi\\end{document}",
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
    )

    assert result.status == "ok"
    assert "sourcesanspro" not in captured_tex["content"]


def test_compile_latex_to_pdf_handles_runner_timeout(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    class _ReadyEnvironment:
        ready = True
        pdflatex_path = "C:/tex/pdflatex.exe"

    monkeypatch.setattr("services.resumes.resume.check_template_compile_environment", lambda path: _ReadyEnvironment())

    def _runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout=60)

    result = compile_latex_to_pdf(
        latex_document="\\documentclass{article}\\begin{document}Hi\\end{document}",
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
    )

    assert result.status == "error"
    assert "timed out" in result.message.lower()
    assert result.log_path == tmp_path / "template.log"


def test_compile_latex_to_pdf_falls_back_to_xelatex_when_pdflatex_fails(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    class _ReadyEnvironment:
        ready = True
        pdflatex_path = "C:/tex/pdflatex.exe"
        xelatex_path = "C:/tex/xelatex.exe"
        lualatex_path = None

    monkeypatch.setattr("services.resumes.resume.check_template_compile_environment", lambda path: _ReadyEnvironment())

    calls: list[str] = []

    def _runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        engine = command[0]
        calls.append(engine)
        tex_name = command[-1]
        pdf_path = cwd / Path(tex_name).with_suffix(".pdf")
        if engine.endswith("pdflatex.exe"):
            return subprocess.CompletedProcess(command, 1, "", "pdftex error")
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
    assert "xelatex" in result.message.lower()
    assert calls[:2] == ["C:/tex/pdflatex.exe", "C:/tex/xelatex.exe"]


def test_compile_latex_to_pdf_stops_after_engine_timeout(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    class _ReadyEnvironment:
        ready = True
        pdflatex_path = "C:/tex/pdflatex.exe"
        xelatex_path = "C:/tex/xelatex.exe"
        lualatex_path = None

    monkeypatch.setattr("services.resumes.resume.check_template_compile_environment", lambda path: _ReadyEnvironment())

    calls: list[str] = []

    def _runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append(command[0])
        return subprocess.CompletedProcess(command, 124, "", "timed out")

    result = compile_latex_to_pdf(
        latex_document="\\documentclass{article}\\begin{document}Hi\\end{document}",
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
    )

    assert result.status == "error"
    assert "timed out" in result.message.lower()
    # Should not try fallback engines after a timeout-sized failure.
    assert calls == ["C:/tex/pdflatex.exe"]


def test_compile_latex_to_pdf_respects_total_budget(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    class _ReadyEnvironment:
        ready = True
        pdflatex_path = "C:/tex/pdflatex.exe"
        xelatex_path = "C:/tex/xelatex.exe"
        lualatex_path = None

    monkeypatch.setattr("services.resumes.resume.check_template_compile_environment", lambda path: _ReadyEnvironment())
    monkeypatch.setattr("services.resumes.resume.LATEX_TOTAL_TIMEOUT_SECONDS", 0)

    calls: list[str] = []

    def _runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append(command[0])
        return subprocess.CompletedProcess(command, 0, "ok", "")

    result = compile_latex_to_pdf(
        latex_document="\\documentclass{article}\\begin{document}Hi\\end{document}",
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
    )

    assert result.status == "error"
    assert "failed" in result.message.lower() or "timed out" in result.message.lower()
    assert calls == []


def test_compile_latex_to_pdf_prefers_xelatex_for_fontspec_templates(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    class _ReadyEnvironment:
        ready = True
        pdflatex_path = "C:/tex/pdflatex.exe"
        xelatex_path = "C:/tex/xelatex.exe"
        lualatex_path = "C:/tex/lualatex.exe"

    monkeypatch.setattr("services.resumes.resume.check_template_compile_environment", lambda path: _ReadyEnvironment())

    calls: list[str] = []

    def _runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        engine = command[0]
        calls.append(engine)
        tex_name = command[-1]
        pdf_path = cwd / Path(tex_name).with_suffix(".pdf")
        pdf_path.write_bytes(b"%PDF-1.4\n")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    result = compile_latex_to_pdf(
        latex_document="\\documentclass{article}\\usepackage{fontspec}\\begin{document}Hi\\end{document}",
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
    )

    assert result.status == "ok"
    assert calls[0] == "C:/tex/xelatex.exe"


def test_compile_latex_to_pdf_strips_pdflatex_only_unicode_directives_for_xelatex(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    class _ReadyEnvironment:
        ready = True
        pdflatex_path = None
        xelatex_path = "C:/tex/xelatex.exe"
        lualatex_path = None
        missing_files = []

    monkeypatch.setattr("services.resumes.resume.check_template_compile_environment", lambda path: _ReadyEnvironment())

    observed_tex: dict[str, str] = {}

    def _runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        tex_name = command[-1]
        tex_path = cwd / tex_name
        observed_tex["content"] = tex_path.read_text(encoding="utf-8")
        pdf_path = cwd / Path(tex_name).with_suffix(".pdf")
        pdf_path.write_bytes(b"%PDF-1.4\n")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    result = compile_latex_to_pdf(
        latex_document=(
            "\\documentclass{article}\n"
            "\\input{glyphtounicode}\n"
            "\\pdfgentounicode=1\n"
            "\\pdfglyphtounicode{A}{0041}\n"
            "\\begin{document}Hi\\end{document}\n"
        ),
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
    )

    assert result.status == "ok"
    assert "input{glyphtounicode}" not in observed_tex["content"]
    assert "\\pdfgentounicode" not in observed_tex["content"]
    assert "\\pdfglyphtounicode" not in observed_tex["content"]


def test_compile_latex_to_pdf_auto_fixes_missing_style_with_retry(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    class _ReadyEnvironment:
        ready = True
        pdflatex_path = "C:/tex/pdflatex.exe"

    monkeypatch.setattr("services.resumes.resume.check_template_compile_environment", lambda path: _ReadyEnvironment())

    observed_texts: list[str] = []

    def _runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        tex_path = cwd / command[-1]
        tex_text = tex_path.read_text(encoding="utf-8")
        observed_texts.append(tex_text)
        if len(observed_texts) == 1:
            return subprocess.CompletedProcess(command, 1, "", "! LaTeX Error: File `missingfont.sty' not found.")
        (cwd / Path(command[-1]).with_suffix(".pdf")).write_bytes(b"%PDF-1.4\n")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    result = compile_latex_to_pdf(
        latex_document="\\documentclass{article}\n\\usepackage{missingfont}\n\\begin{document}Hi\\end{document}",
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
        max_auto_fix_retries=2,
    )

    assert result.status == "ok"
    assert len(observed_texts) == 2
    assert "\\usepackage{missingfont}" in observed_texts[0]
    assert "\\usepackage{missingfont}" not in observed_texts[1]
    assert any("stripped missing style packages" in label for label in result.repairs_applied)
    assert result.repair_attempts == 1


def test_compile_latex_to_pdf_respects_auto_fix_retry_cap(monkeypatch, tmp_path: Path) -> None:
    template_path = tmp_path / "template.tex"
    template_path.write_text("\\documentclass{article}\n", encoding="utf-8")

    class _ReadyEnvironment:
        ready = True
        pdflatex_path = "C:/tex/pdflatex.exe"

    monkeypatch.setattr("services.resumes.resume.check_template_compile_environment", lambda path: _ReadyEnvironment())

    call_count = 0

    def _runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(command, 1, "", "! LaTeX Error: File `missingfont.sty' not found.")

    result = compile_latex_to_pdf(
        latex_document="\\documentclass{article}\n\\usepackage{missingfont}\n\\begin{document}Hi\\end{document}",
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
        max_auto_fix_retries=1,
    )

    assert result.status == "error"
    assert call_count == 2
    assert result.repair_attempts == 1


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


def test_sanitize_latex_document_for_compile_closes_unbalanced_braces() -> None:
    source = "\\textbf{Honours and Scholarship\\section{Publication}"

    sanitized = sanitize_latex_document_for_compile(source)

    assert sanitized.endswith("}")


def test_sanitize_latex_document_for_compile_closes_runaway_inline_text_command() -> None:
    source = "\\textbf{Honours and Scholarship\\section{Publication}"

    sanitized = sanitize_latex_document_for_compile(source)

    assert "\\textbf{Honours and Scholarship}\\section{Publication}" in sanitized


def test_sanitize_latex_document_for_compile_preserves_textbf_scshape_payload() -> None:
    source = "{\\fontsize{28}{34}\\selectfont \\textbf{\\scshape Ricky Nong}} \\\\ \\vspace{5pt}"

    sanitized = sanitize_latex_document_for_compile(source)

    assert "\\textbf{\\scshape Ricky Nong}} \\\\" in sanitized


def test_sanitize_latex_document_for_compile_normalizes_item_section_collision() -> None:
    source = (
        "\\resumeItemListStart\n"
        "\\item \\small{\\textbf{Honours and Scholarship\\section{Publication}\n"
        "\\resumeSubHeadingListStart"
    )

    sanitized = sanitize_latex_document_for_compile(source)

    assert "\\item \\small{\\textbf{Honours and Scholarship}" in sanitized
    assert "\\resumeItemListEnd" in sanitized
    assert "\\resumeSubHeadingListEnd" in sanitized
    assert "\\section{Publication}" in sanitized


def test_sanitize_latex_document_for_compile_closes_open_resume_lists_before_section() -> None:
    source = (
        "\\section{Work Experience}\n"
        "\\resumeSubHeadingListStart\n"
        "\\resumeSubheading{Role}{Dates}{Company}{City}\n"
        "\\resumeItemListStart\n"
        "\\resumeItem{Did work.}\n"
        "\\section{Extracurriculars}\n"
        "\\end{document}"
    )

    sanitized = sanitize_latex_document_for_compile(source)

    assert "\\resumeItemListEnd\n  \\resumeSubHeadingListEnd\n\\section{Extracurriculars}" in sanitized


def test_sanitize_latex_document_for_compile_closes_small_itemize_group_brace() -> None:
    source = (
        "{\\small \\begin{itemize}[leftmargin=0.15in]\n"
        "\\item {Publication detail.}\n"
        "\\end{itemize}\n"
        "\\section{Next}"
    )

    sanitized = sanitize_latex_document_for_compile(source)

    assert "\\end{itemize}}" in sanitized


def test_sanitize_latex_document_for_compile_drops_orphan_lower_fragment_between_commands() -> None:
    source = (
        "\\resumeItemListEnd\n"
        "ista\n"
        "\\section{Work Experience}"
    )

    sanitized = sanitize_latex_document_for_compile(source)

    assert "\nista\n" not in f"\n{sanitized}\n"
    assert "\\section{Work Experience}" in sanitized


def test_sanitize_latex_document_for_compile_keeps_lower_fragment_when_not_between_commands() -> None:
    source = (
        "\\section{Summary}\n"
        "ista\n"
        "plain text line"
    )

    sanitized = sanitize_latex_document_for_compile(source)

    assert "ista" in sanitized


def test_sanitize_latex_document_for_compile_starts_subheading_list_for_lonely_subheading() -> None:
    source = (
        "\\section{Extracurriculars}\n"
        "\\resumeSubheading{Role}{Dates}{Org}{City}\n"
        "\\resumeItemListStart\n"
        "\\resumeItem{Did work.}\n"
        "\\resumeItemListEnd\n"
        "\\end{document}"
    )

    sanitized = sanitize_latex_document_for_compile(source)

    assert "\\section{Extracurriculars}\n  \\resumeSubHeadingListStart\n\\resumeSubheading" in sanitized
    assert "\\resumeSubHeadingListEnd\n\\end{document}" in sanitized


def test_sanitize_latex_document_for_compile_normalizes_empty_textbf_small_pattern() -> None:
    source = "\\textbf{}\\small #2} \\\\"

    sanitized = sanitize_latex_document_for_compile(source)

    assert "\\textbf{\\small #2}" in sanitized


def test_sanitize_latex_document_for_compile_normalizes_empty_textbf_scshape_pattern() -> None:
    source = "\\textbf{}\\scshape Ricky Nong}} \\\\"

    sanitized = sanitize_latex_document_for_compile(source)

    assert "\\textbf{\\scshape Ricky Nong}" in sanitized


def test_lint_latex_document_for_compile_normalizes_double_slash_control_lines() -> None:
    source = "\\\\resumeItemListStart\n\\\\vspace{2pt}\n\\\\section{Next}"

    linted = lint_latex_document_for_compile(source)

    assert "\\resumeItemListStart" in linted.latex_document
    assert "\\vspace{2pt}" in linted.latex_document
    assert "\\section{Next}" in linted.latex_document
    assert "normalized escaped control commands" in linted.repairs_applied


def test_lint_latex_document_for_compile_unwraps_braced_resume_end_macros() -> None:
    source = "{\\resumeItemListEnd}\n{\\resumeSubHeadingListEnd}"

    linted = lint_latex_document_for_compile(source)

    assert "{\\resumeItemListEnd}" not in linted.latex_document
    assert "{\\resumeSubHeadingListEnd}" not in linted.latex_document
    assert "\\resumeItemListEnd" in linted.latex_document
    assert "\\resumeSubHeadingListEnd" in linted.latex_document
    assert "unwrapped braced resume list end macros" in linted.repairs_applied


def test_lint_latex_document_for_compile_normalizes_color_dvips_option_case() -> None:
    source = "\\usepackage[usenames,dvipsNames]{color}\n\\begin{document}\nOK\n\\end{document}"

    linted = lint_latex_document_for_compile(source)

    assert "dvipsNames" not in linted.latex_document
    assert "dvipsnames" in linted.latex_document
    assert "normalized color package option casing" in linted.repairs_applied


def test_lint_latex_document_for_compile_removes_dangling_single_trailing_backslashes() -> None:
    source = "\\vspace{-4pt}\\\n\\item text\\\n\\\\\n"

    linted = lint_latex_document_for_compile(source)

    lines = linted.latex_document.splitlines()
    assert lines[0] == "\\vspace{-4pt}"
    assert lines[1] == "\\item text"
    # Preserve intentional TeX line break token.
    assert lines[2] == "\\\\"
    assert "removed dangling single trailing backslashes" in linted.repairs_applied



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

def test_sanitize_latex_document_for_compile_decodes_four_backslash_newline_to_linebreak() -> None:
    """\\\\\\\\\\n (encoded LaTeX \\\\ line-break + newline) must decode to \\\\ + newline, not \\\\\\n."""
    # In a JSON-escaped doc: \\\\\n = four backslashes + literal \n
    # Four backslashes = encoded LaTeX \\ line-break
    # \n = JSON-encoded newline
    # Expected output: \\ (real LaTeX line-break) + actual newline
    source = "\\\\textbf{#1} & \\\\textbf{\\\\small #2} \\\\\\\\\n\\\\hline"

    sanitized = sanitize_latex_document_for_compile(source)

    # Should contain \\ + actual newline (not \\\n or \n as two chars)
    assert "\\\\\n" in sanitized
    # Must NOT contain the undefined \n control sequence (backslash followed by n as two chars after \\)
    lines = sanitized.splitlines()
    for line in lines:
        assert not line.endswith("\\n"), f"Literal \\\\n control sequence found at end of line: {line!r}"


def test_sanitize_latex_document_for_compile_fixes_over_escaped_math_dollars() -> None:
    """\\$\\vcenter (LLM over-escaped math dollar) must become $\\vcenter."""
    source = "\\documentclass{article}\n\\begin{document}\n$\\vcenter{\\hbox{\\$\\vcenter{x}}}$\n\\end{document}"

    sanitized = sanitize_latex_document_for_compile(source)

    assert "\\$\\vcenter" not in sanitized
    assert "$\\vcenter" in sanitized


def test_sanitize_latex_document_for_compile_escapes_percent_inside_arguments_only() -> None:
    """% inside macro arguments must be escaped; % outside (inline comments) must be preserved."""
    source = (
        "\\documentclass{article}\n"
        "\\fancyhf{} % clear all header and footer fields\n"
        "\\begin{document}\n"
        "\\resumeItem{Improved throughput by 3% for rehabilitation services}\n"
        "\\end{document}"
    )

    sanitized = sanitize_latex_document_for_compile(source)

    # % inside the resumeItem argument must be escaped
    assert "3\\% for rehabilitation" in sanitized
    # % in the inline LaTeX comment outside any braces must NOT be escaped
    assert "\\fancyhf{} % clear all header" in sanitized


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


def test_compile_latex_to_pdf_carries_internal_lint_findings(monkeypatch, tmp_path: Path) -> None:
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
        latex_document="\\documentclass{article}\\begin{document}Hi",
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
    )

    assert result.status == "ok"
    assert any("missing \\end{document}" in finding for finding in result.lint_findings)


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

    models_api = _FakeModelsApi('{"summary":"ok","targeted_bullets":["a"],"revised_profile":"b","rewritten_tex":"\\\\documentclass{article}\\\\begin{document}Hi\\\\end{document}"}')
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


def test_extract_latex_document_rejects_latex_tag_with_placeholder_content() -> None:
    """<latex>...</latex> whose content lacks \\documentclass must return None."""
    assert extract_latex_document("<latex>...</latex>") is None
    assert extract_latex_document("<latex>some plain text</latex>") is None


def test_extract_latex_document_handles_json_fenced_response() -> None:
    """A ```json fence containing a rewritten_tex field must return the LaTeX, not the JSON."""
    latex = "\\documentclass[11pt]{article}\n\\begin{document}\nHello\n\\end{document}"
    payload = json.dumps({"summary": "ok", "rewritten_tex": latex})
    fenced = f"```json\n{payload}\n```"
    result = extract_latex_document(fenced)
    assert result == latex


def test_fix_malformed_end_document_without_closing_brace() -> None:
    """\\end{document> (no closing brace) must be corrected to \\end{document}."""
    from services.resumes.resume import sanitize_latex_document_for_compile
    # Simulate the normalized (single-backslash) form the sanitizer receives
    bad = "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document>"
    fixed = sanitize_latex_document_for_compile(bad)
    assert "\\end{document}" in fixed
    assert "\\end{document>" not in fixed


def test_split_comment_concatenated_commands() -> None:
    """% separator comment immediately followed by \\begin/\\section must be split onto its own line."""
    from services.resumes.resume import sanitize_latex_document_for_compile
    # Simulate the LLM omitting the newline between a separator comment and the command
    bad = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "%----------HEADING----------\\begin{center}\n"
        "Title\n"
        "\\end{center}\n"
        "%-----------SECTION-----------\\section{Skills}\n"
        "stuff\n"
        "\\end{document}\n"
    )
    fixed = sanitize_latex_document_for_compile(bad)
    lines = fixed.splitlines()
    # The \begin{center} and \section must appear on their own lines (not after %)
    begin_lines = [l for l in lines if "\\begin{center}" in l]
    section_lines = [l for l in lines if "\\section{Skills}" in l]
    assert begin_lines, "\\begin{center} should still be present"
    assert not begin_lines[0].lstrip().startswith("%"), "\\begin{center} must not be on a comment line"
    assert section_lines, "\\section{Skills} should still be present"
    assert not section_lines[0].lstrip().startswith("%"), "\\section must not be on a comment line"


def test_split_comment_does_not_activate_commented_out_environments() -> None:
    """Legitimately commented-out \\begin/\\end blocks must NOT be activated."""
    from services.resumes.resume import sanitize_latex_document_for_compile
    # The LLM left an entire itemize block commented out — must stay commented
    bad = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "text\n"
        "%\\begin{itemize}\n"
        "%  \\item one\n"
        "%  \\item two\n"
        "%\\end{itemize}\n"
        "\\end{document}\n"
    )
    fixed = sanitize_latex_document_for_compile(bad)
    lines = fixed.splitlines()
    # Every line containing \begin{itemize} or \end{itemize} must start with %
    for line in lines:
        if "\\begin{itemize}" in line or "\\end{itemize}" in line:
            assert line.lstrip().startswith("%"), (
                f"itemize command should remain commented out, got: {line!r}"
            )


def test_close_unclosed_resume_list_macros_on_truncation() -> None:
    """Truncated output missing \\resumeSubHeadingListEnd must be auto-closed."""
    from services.resumes.resume import sanitize_latex_document_for_compile
    # Document body only (no \\newcommand definitions) so counting is unambiguous.
    # The LLM truncated before emitting \\resumeSubHeadingListEnd.
    truncated = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\resumeSubHeadingListStart\n"
        "\\resumeItemListStart\n"
        "text\n"
        "\\resumeItemListEnd\n"
        "\\end{document}\n"
    )
    assert truncated.count("\\resumeSubHeadingListEnd") == 0  # sanity check
    fixed = sanitize_latex_document_for_compile(truncated)
    # Exactly one closer must have been inserted
    assert fixed.count("\\resumeSubHeadingListEnd") == 1
    # It must appear before \end{document}
    end_pos = fixed.index("\\end{document}")
    close_pos = fixed.index("\\resumeSubHeadingListEnd")
    assert close_pos < end_pos, "\\resumeSubHeadingListEnd must appear before \\end{document}"


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


LINKEDIN_GUEST_API_HTML = """
<section>
  <h2 class="top-card-layout__title topcard__title">Energy Procurement Markets Co-op</h2>
  <div class="topcard__flavor-row">
    <a class="topcard__org-name-link" href="https://linkedin.com/company/powerco">PowerCo</a>
    <span class="topcard__flavor topcard__flavor--bullet">St Thomas, Ontario, Canada</span>
  </div>
  <div class="show-more-less-html__markup">
    <strong>Who We Are<br><br></strong>We build batteries.<br><br>
    <ul>
      <li>Analyze energy invoices for cost-saving opportunities.</li>
      <li>Support sourcing activities and stakeholder collaboration.</li>
    </ul>
  </div>
  <ul class="description__job-criteria-list">
    <li class="description__job-criteria-item">
      <h3 class="description__job-criteria-subheader">Seniority level</h3>
      <span class="description__job-criteria-text description__job-criteria-text--criteria">Internship</span>
    </li>
    <li class="description__job-criteria-item">
      <h3 class="description__job-criteria-subheader">Employment type</h3>
      <span class="description__job-criteria-text description__job-criteria-text--criteria">Full-time</span>
    </li>
  </ul>
</section>
"""


def _mock_linkedin_guest_response(status_code: int = 200, text: str = LINKEDIN_GUEST_API_HTML):
    class _Response:
        def raise_for_status(self) -> None:
            if status_code >= 400:
                raise requests.HTTPError(f"{status_code} error")

    response = _Response()
    response.text = text
    return response


def test_scrape_job_posting_uses_linkedin_guest_api_before_direct_page_fetch(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get(url, *args, **kwargs):
        calls.append(url)
        if "jobs-guest/jobs/api/jobPosting" in url:
            return _mock_linkedin_guest_response()
        raise AssertionError("direct LinkedIn page should not be fetched when guest API succeeds")

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    result = scrape_job_posting("https://www.linkedin.com/jobs/view/4436972760")

    assert calls == ["https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/4436972760"]
    assert result.title == "Energy Procurement Markets Co-op"
    assert result.company == "PowerCo"
    assert result.location == "St Thomas, Ontario, Canada"
    assert "We build batteries." in result.description
    assert "Analyze energy invoices" in result.description
    assert result.source_url == "https://www.linkedin.com/jobs/view/4436972760"


@pytest.mark.parametrize(
    "posting_url,expected_job_id",
    [
        ("https://www.linkedin.com/jobs/view/4436972760", "4436972760"),
        ("https://www.linkedin.com/jobs/view/4436972760/", "4436972760"),
        (
            "https://ca.linkedin.com/jobs/view/energy-procurement-markets-co-op-at-powerco-4436972760",
            "4436972760",
        ),
        ("https://www.linkedin.com/jobs/collections/recommended/?currentJobId=4436972760", "4436972760"),
        ("https://www.linkedin.com/jobs/view/4436972760?trk=x&utm_source=y", "4436972760"),
    ],
)
def test_scrape_job_posting_extracts_linkedin_job_id_from_url_shapes(monkeypatch, posting_url, expected_job_id) -> None:
    captured_urls: list[str] = []

    def fake_get(url, *args, **kwargs):
        captured_urls.append(url)
        return _mock_linkedin_guest_response()

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    result = scrape_job_posting(posting_url)

    assert captured_urls == [f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{expected_job_id}"]
    assert result.title == "Energy Procurement Markets Co-op"


def test_scrape_job_posting_linkedin_falls_back_when_no_job_id_in_url(monkeypatch) -> None:
    class _DirectPageResponse:
        def raise_for_status(self) -> None:
            return None

        text = "<html><head><title>LinkedIn Job Search</title></head><body><h1>Python Jobs</h1></body></html>"

    def fake_get(url, *args, **kwargs):
        assert "jobs-guest" not in url
        return _DirectPageResponse()

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    result = scrape_job_posting("https://www.linkedin.com/jobs/search/?keywords=python")

    assert result.title == "LinkedIn Job Search"


def test_scrape_job_posting_linkedin_falls_back_when_guest_api_returns_error(monkeypatch) -> None:
    call_order: list[str] = []

    def fake_get(url, *args, **kwargs):
        if "jobs-guest" in url:
            call_order.append("guest_api")
            return _mock_linkedin_guest_response(status_code=429)
        call_order.append("direct_page")

        class _Response:
            def raise_for_status(self) -> None:
                return None

            text = "<html><head><title>Direct Fallback Job</title></head><body><h1>Direct Fallback Job</h1></body></html>"

        return _Response()

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    result = scrape_job_posting("https://www.linkedin.com/jobs/view/4436972760")

    assert call_order == ["guest_api", "direct_page"]
    assert result.title == "Direct Fallback Job"


def test_scrape_job_posting_linkedin_falls_back_when_guest_api_html_is_empty_shell(monkeypatch) -> None:
    call_order: list[str] = []

    def fake_get(url, *args, **kwargs):
        if "jobs-guest" in url:
            call_order.append("guest_api")
            return _mock_linkedin_guest_response(text="<html><body>Job no longer available</body></html>")
        call_order.append("direct_page")

        class _Response:
            def raise_for_status(self) -> None:
                raise requests.HTTPError("blocked")

        return _Response()

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)
    monkeypatch.setattr("services.resumes.listing.browser_service.ensure_ready", lambda: False)
    monkeypatch.setattr("services.resumes.listing.job_site_from_url", lambda url: "linkedin")
    monkeypatch.setattr(
        "services.resumes.listing.scrape_jobs_from_board_url",
        lambda url, max_items=5: [
            {
                "title": "Board Fallback Job",
                "company": "Fallback Co",
                "location": "Remote",
            }
        ],
    )

    result = scrape_job_posting("https://www.linkedin.com/jobs/view/4436972760")

    assert call_order == ["guest_api", "direct_page"]
    assert result.title == "Board Fallback Job"


def test_scrape_linkedin_job_posting_with_guest_api_handles_missing_description(monkeypatch) -> None:
    from services.resumes.listing import _scrape_linkedin_job_posting_with_guest_api

    monkeypatch.setattr(
        "services.resumes.listing.requests.get",
        lambda *args, **kwargs: _mock_linkedin_guest_response(text="<html><body></body></html>"),
    )

    result = _scrape_linkedin_job_posting_with_guest_api(
        "https://www.linkedin.com/jobs/view/4436972760", 20, "test-agent"
    )

    assert result is None


def test_scrape_linkedin_job_posting_with_guest_api_handles_request_exception(monkeypatch) -> None:
    from services.resumes.listing import _scrape_linkedin_job_posting_with_guest_api

    monkeypatch.setattr(
        "services.resumes.listing.requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.SSLError("EOF occurred")),
    )

    result = _scrape_linkedin_job_posting_with_guest_api(
        "https://www.linkedin.com/jobs/view/4436972760", 20, "test-agent"
    )

    assert result is None


@pytest.mark.parametrize(
    "posting_url,expected",
    [
        ("https://www.linkedin.com/jobs/view/4436972760", "4436972760"),
        ("https://www.linkedin.com/jobs/view/4436972760/", "4436972760"),
        ("https://ca.linkedin.com/jobs/view/some-slug-4436972760", "4436972760"),
        ("https://www.linkedin.com/jobs/collections/recommended/?currentJobId=4436972760", "4436972760"),
        ("https://www.linkedin.com/jobs/collections/recommended/?jobId=4436972760", "4436972760"),
        ("https://www.linkedin.com/jobs/search/?keywords=python", None),
        ("https://www.linkedin.com/jobs/view/not-a-job-id", None),
    ],
)
def test_extract_linkedin_job_id(posting_url, expected) -> None:
    from services.resumes.listing import _extract_linkedin_job_id

    assert _extract_linkedin_job_id(posting_url) == expected


# ── Transient-error retry (SSLError/ConnectionError/Timeout) ─────────────────


def test_get_with_retry_recovers_from_transient_ssl_error_then_succeeds(monkeypatch) -> None:
    from services.resumes.listing import _get_with_retry

    attempts: list[int] = []

    class _Response:
        text = "ok"

    def fake_get(url, headers=None, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise requests.exceptions.SSLError("EOF occurred in violation of protocol")
        return _Response()

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)
    monkeypatch.setattr("services.resumes.listing.time.sleep", lambda _seconds: None)

    response = _get_with_retry("https://example.com/job", headers={}, timeout=10)

    assert len(attempts) == 3
    assert response.text == "ok"


def test_get_with_retry_gives_up_after_max_attempts_and_raises_original_error(monkeypatch) -> None:
    from services.resumes.listing import _get_with_retry

    attempts: list[int] = []

    def fake_get(url, headers=None, timeout=None):
        attempts.append(1)
        raise requests.exceptions.ConnectionError("connection reset")

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)
    monkeypatch.setattr("services.resumes.listing.time.sleep", lambda _seconds: None)

    with pytest.raises(requests.exceptions.ConnectionError):
        _get_with_retry("https://example.com/job", headers={}, timeout=10, max_attempts=3)

    assert len(attempts) == 3


def test_get_with_retry_does_not_retry_non_transient_errors(monkeypatch) -> None:
    from services.resumes.listing import _get_with_retry

    attempts: list[int] = []

    def fake_get(url, headers=None, timeout=None):
        attempts.append(1)
        raise requests.exceptions.TooManyRedirects("redirect loop")

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    with pytest.raises(requests.exceptions.TooManyRedirects):
        _get_with_retry("https://example.com/job", headers={}, timeout=10, max_attempts=3)

    assert len(attempts) == 1


def test_scrape_job_posting_recovers_from_transient_ssl_error_without_falling_back(monkeypatch) -> None:
    attempts: list[int] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        text = "<html><head><title>Recovered Job</title></head><body><h1>Recovered Job</h1></body></html>"

    def fake_get(url, headers=None, timeout=None):
        attempts.append(1)
        if len(attempts) < 2:
            raise requests.exceptions.SSLError("EOF occurred in violation of protocol")
        return _Response()

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)
    monkeypatch.setattr("services.resumes.listing.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("services.resumes.listing.job_site_from_url", lambda url: None)

    result = scrape_job_posting("https://example.com/careers/123")

    assert len(attempts) == 2
    assert result.title == "Recovered Job"


# ── Greenhouse direct-API fallback ────────────────────────────────────────────

GREENHOUSE_JOB_JSON = {
    "title": "Android Engineer",
    "company_name": "Robinhood",
    "location": {"name": "New York, NY"},
    "content": "<p>Build the mobile app.</p><ul><li>Requirement: 3+ years Kotlin experience.</li></ul>",
}


def _mock_json_response(status_code: int = 200, payload=None):
    class _Response:
        def raise_for_status(self) -> None:
            if status_code >= 400:
                raise requests.HTTPError(f"{status_code} error")

        def json(self):
            return payload

    return _Response()


def test_scrape_job_posting_uses_greenhouse_api_before_direct_page_fetch(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get(url, *args, **kwargs):
        calls.append(url)
        if "boards-api.greenhouse.io" in url:
            return _mock_json_response(payload=GREENHOUSE_JOB_JSON)
        raise AssertionError("direct Greenhouse page should not be fetched when the API succeeds")

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    result = scrape_job_posting("https://boards.greenhouse.io/robinhood/jobs/6669758")

    assert calls == ["https://boards-api.greenhouse.io/v1/boards/robinhood/jobs/6669758?content=true"]
    assert result.title == "Android Engineer"
    assert result.company == "Robinhood"
    assert result.location == "New York, NY"
    assert "Build the mobile app." in result.description
    assert result.source_url == "https://boards.greenhouse.io/robinhood/jobs/6669758"


def test_scrape_job_posting_greenhouse_job_boards_domain_variant(monkeypatch) -> None:
    def fake_get(url, *args, **kwargs):
        return _mock_json_response(payload=GREENHOUSE_JOB_JSON)

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    result = scrape_job_posting("https://job-boards.greenhouse.io/affirm/jobs/7767436003")

    assert result.title == "Android Engineer"


def test_scrape_job_posting_greenhouse_falls_back_when_api_returns_error(monkeypatch) -> None:
    call_order: list[str] = []

    def fake_get(url, *args, **kwargs):
        if "boards-api.greenhouse.io" in url:
            call_order.append("api")
            return _mock_json_response(status_code=404)
        call_order.append("direct_page")

        class _Response:
            def raise_for_status(self) -> None:
                return None

            text = "<html><head><title>Direct Greenhouse Page</title></head><body></body></html>"

        return _Response()

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    result = scrape_job_posting("https://boards.greenhouse.io/robinhood/jobs/6669758")

    assert call_order == ["api", "direct_page"]
    assert result.title == "Direct Greenhouse Page"


@pytest.mark.parametrize(
    "posting_url,expected",
    [
        ("https://boards.greenhouse.io/robinhood/jobs/6669758", ("robinhood", "6669758")),
        ("https://job-boards.greenhouse.io/affirm/jobs/7767436003", ("affirm", "7767436003")),
        ("https://boards.greenhouse.io/robinhood/jobs/6669758?t=gh_src=", ("robinhood", "6669758")),
        ("https://stripe.com/jobs/search?gh_jid=7954688", None),
        ("https://boards.greenhouse.io/robinhood", None),
    ],
)
def test_extract_greenhouse_job_ref(posting_url, expected) -> None:
    from services.resumes.listing import _extract_greenhouse_job_ref

    assert _extract_greenhouse_job_ref(posting_url) == expected


# ── Lever direct-API fallback ─────────────────────────────────────────────────

LEVER_POSTING_JSON = {
    "text": "Head of People & Culture",
    "categories": {"team": "Operations", "location": "UK - London"},
    "descriptionPlain": "Own the company's People strategy.",
    "lists": [
        {"text": "What you'll do", "content": "<li>Define and execute a People strategy.</li>"},
    ],
}


def test_scrape_job_posting_uses_lever_api_before_direct_page_fetch(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get(url, *args, **kwargs):
        calls.append(url)
        if "api.lever.co" in url:
            return _mock_json_response(payload=LEVER_POSTING_JSON)
        raise AssertionError("direct Lever page should not be fetched when the API succeeds")

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    posting_url = "https://jobs.lever.co/1inch/fdd44b03-06e7-4409-9102-5c3a52975852"
    result = scrape_job_posting(posting_url)

    assert calls == ["https://api.lever.co/v0/postings/1inch/fdd44b03-06e7-4409-9102-5c3a52975852?mode=json"]
    assert result.title == "Head of People & Culture"
    assert result.company == "1Inch"
    assert result.location == "UK - London"
    assert "Own the company's People strategy." in result.description
    assert "Define and execute a People strategy." in result.description
    assert result.source_url == posting_url


def test_scrape_job_posting_lever_apply_path_variant(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.resumes.listing.requests.get",
        lambda *args, **kwargs: _mock_json_response(payload=LEVER_POSTING_JSON),
    )

    result = scrape_job_posting("https://jobs.lever.co/1inch/fdd44b03-06e7-4409-9102-5c3a52975852/apply")

    assert result.title == "Head of People & Culture"


def test_scrape_job_posting_lever_falls_back_when_api_returns_error(monkeypatch) -> None:
    call_order: list[str] = []

    def fake_get(url, *args, **kwargs):
        if "api.lever.co" in url:
            call_order.append("api")
            return _mock_json_response(status_code=404)
        call_order.append("direct_page")

        class _Response:
            def raise_for_status(self) -> None:
                return None

            text = "<html><head><title>Direct Lever Page</title></head><body></body></html>"

        return _Response()

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    result = scrape_job_posting("https://jobs.lever.co/1inch/fdd44b03-06e7-4409-9102-5c3a52975852")

    assert call_order == ["api", "direct_page"]
    assert result.title == "Direct Lever Page"


@pytest.mark.parametrize(
    "posting_url,expected",
    [
        ("https://jobs.lever.co/1inch/fdd44b03-06e7-4409-9102-5c3a52975852", ("1inch", "fdd44b03-06e7-4409-9102-5c3a52975852")),
        ("https://jobs.lever.co/1inch/fdd44b03-06e7-4409-9102-5c3a52975852/apply", ("1inch", "fdd44b03-06e7-4409-9102-5c3a52975852")),
        ("https://jobs.lever.co/1inch", None),
    ],
)
def test_extract_lever_posting_ref(posting_url, expected) -> None:
    from services.resumes.listing import _extract_lever_posting_ref

    assert _extract_lever_posting_ref(posting_url) == expected


# ── Ashby direct-API fallback ─────────────────────────────────────────────────

ASHBY_JOB_POSTING_JSON = {
    "data": {
        "jobPosting": {
            "id": "0e810d06-c325-4edf-833b-16300e09057c",
            "title": "Account Executive",
            "departmentName": "Sales",
            "locationName": "SF Bay Area",
            "employmentType": "FullTime",
            "descriptionHtml": "<p>Own the full sales cycle.</p><ul><li>Requirement: 3+ years closing experience.</li></ul>",
        }
    }
}


def test_scrape_job_posting_uses_ashby_api_before_direct_page_fetch(monkeypatch) -> None:
    calls: list[str] = []

    def fake_post(url, *args, **kwargs):
        calls.append(url)
        return _mock_json_response(payload=ASHBY_JOB_POSTING_JSON)

    def fake_get(url, *args, **kwargs):
        raise AssertionError("direct Ashby page should not be fetched when the API succeeds")

    monkeypatch.setattr("services.resumes.listing.requests.post", fake_post)
    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    posting_url = "https://jobs.ashbyhq.com/coframe/0e810d06-c325-4edf-833b-16300e09057c"
    result = scrape_job_posting(posting_url)

    assert calls == ["https://jobs.ashbyhq.com/api/non-user-graphql?op=ApiJobPosting"]
    assert result.title == "Account Executive"
    assert result.company == "Coframe"
    assert result.location == "SF Bay Area"
    assert "Own the full sales cycle." in result.description
    assert result.source_url == posting_url


def test_scrape_job_posting_ashby_falls_back_when_api_returns_no_posting(monkeypatch) -> None:
    call_order: list[str] = []

    def fake_post(url, *args, **kwargs):
        call_order.append("api")
        return _mock_json_response(payload={"data": {"jobPosting": None}})

    def fake_get(url, *args, **kwargs):
        call_order.append("direct_page")

        class _Response:
            def raise_for_status(self) -> None:
                return None

            text = "<html><head><title>Direct Ashby Page</title></head><body></body></html>"

        return _Response()

    monkeypatch.setattr("services.resumes.listing.requests.post", fake_post)
    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    result = scrape_job_posting("https://jobs.ashbyhq.com/coframe/0e810d06-c325-4edf-833b-16300e09057c")

    assert call_order == ["api", "direct_page"]
    assert result.title == "Direct Ashby Page"


@pytest.mark.parametrize(
    "posting_url,expected",
    [
        ("https://jobs.ashbyhq.com/coframe/0e810d06-c325-4edf-833b-16300e09057c", ("coframe", "0e810d06-c325-4edf-833b-16300e09057c")),
        ("https://jobs.ashbyhq.com/coframe", None),
    ],
)
def test_extract_ashby_job_ref(posting_url, expected) -> None:
    from services.resumes.listing import _extract_ashby_job_ref

    assert _extract_ashby_job_ref(posting_url) == expected


# ── Workday direct-API fallback ───────────────────────────────────────────────

WORKDAY_JOB_JSON = {
    "jobPostingInfo": {
        "title": "HP PC Sales Representative",
        "location": "Naples, FL",
        "jobDescription": "<p>Sell premium PCs.</p><ul><li>Requirement: retail sales experience.</li></ul>",
    },
    "hiringOrganization": {"name": "2020 Companies, Inc."},
}


def test_scrape_job_posting_uses_workday_api_before_direct_page_fetch(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get(url, *args, **kwargs):
        calls.append(url)
        if "/wday/cxs/" in url:
            return _mock_json_response(payload=WORKDAY_JOB_JSON)
        raise AssertionError("direct Workday page should not be fetched when the API succeeds")

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    posting_url = "https://2020companies.wd1.myworkdayjobs.com/external_careers/job/Naples-FL/HP-PC-Sales-Representative_REQ_107459"
    result = scrape_job_posting(posting_url)

    assert calls == [
        "https://2020companies.wd1.myworkdayjobs.com/wday/cxs/2020companies/external_careers"
        "/job/Naples-FL/HP-PC-Sales-Representative_REQ_107459"
    ]
    assert result.title == "HP PC Sales Representative"
    assert result.company == "2020 Companies, Inc."
    assert result.location == "Naples, FL"
    assert "Sell premium PCs." in result.description
    assert result.source_url == posting_url


def test_scrape_job_posting_workday_falls_back_when_api_returns_error(monkeypatch) -> None:
    call_order: list[str] = []

    def fake_get(url, *args, **kwargs):
        if "/wday/cxs/" in url:
            call_order.append("api")
            return _mock_json_response(status_code=404)
        call_order.append("direct_page")

        class _Response:
            def raise_for_status(self) -> None:
                return None

            text = "<html><head><title>Direct Workday Page</title></head><body></body></html>"

        return _Response()

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    posting_url = "https://2020companies.wd1.myworkdayjobs.com/external_careers/job/Naples-FL/HP-PC-Sales-Representative_REQ_107459"
    result = scrape_job_posting(posting_url)

    assert call_order == ["api", "direct_page"]
    assert result.title == "Direct Workday Page"


@pytest.mark.parametrize(
    "posting_url,expected",
    [
        (
            "https://2020companies.wd1.myworkdayjobs.com/external_careers/job/Naples-FL/HP-PC-Sales-Representative_REQ_107459",
            ("2020companies", "wd1", "external_careers", "/job/Naples-FL/HP-PC-Sales-Representative_REQ_107459"),
        ),
        (
            "https://transamerica.wd5.myworkdayjobs.com/spain/job/Madrid-Cristalia-Spain/Service-Design_R20062081-2",
            ("transamerica", "wd5", "spain", "/job/Madrid-Cristalia-Spain/Service-Design_R20062081-2"),
        ),
        ("https://2020companies.wd1.myworkdayjobs.com/external_careers", None),
        ("https://example.com/job/whatever", None),
    ],
)
def test_extract_workday_job_ref(posting_url, expected) -> None:
    from services.resumes.listing import _extract_workday_job_ref

    assert _extract_workday_job_ref(posting_url) == expected


# ── iCIMS direct-API fallback ─────────────────────────────────────────────────

ICIMS_LD_JSON_HTML = """
<html><body>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "JobPosting",
  "title": "Structural Engineer",
  "hiringOrganization": {"name": "142 Design Group"},
  "jobLocation": {"address": {"addressLocality": "White Plains", "addressRegion": "NY", "addressCountry": "US"}},
  "description": "<p>Design structural systems.</p><p>Requirement: PE license preferred.</p>"
}
</script>
</body></html>
"""


def test_scrape_job_posting_uses_icims_ld_json_before_direct_page_fetch(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get(url, *args, **kwargs):
        calls.append(url)

        class _Response:
            def raise_for_status(self) -> None:
                return None

            text = ICIMS_LD_JSON_HTML

        return _Response()

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    posting_url = "https://careers-142designgroup.icims.com/jobs/11380/structural-engineer/job"
    result = scrape_job_posting(posting_url)

    assert calls == [posting_url + "?in_iframe=1"]
    assert result.title == "Structural Engineer"
    assert result.company == "142 Design Group"
    assert result.location == "White Plains, NY, US"
    assert "Design structural systems." in result.description
    assert result.source_url == posting_url


def test_scrape_job_posting_icims_falls_back_when_no_ld_json_present(monkeypatch) -> None:
    call_order: list[str] = []

    def fake_get(url, *args, **kwargs):
        if "in_iframe=1" in url:
            call_order.append("iframe_shell")

            class _ShellResponse:
                def raise_for_status(self) -> None:
                    return None

                text = "<html><body>No structured data here</body></html>"

            return _ShellResponse()

        call_order.append("direct_page")

        class _Response:
            def raise_for_status(self) -> None:
                return None

            text = "<html><head><title>Direct iCIMS Page</title></head><body></body></html>"

        return _Response()

    monkeypatch.setattr("services.resumes.listing.requests.get", fake_get)

    result = scrape_job_posting("https://careers-142designgroup.icims.com/jobs/11380/structural-engineer/job")

    assert call_order == ["iframe_shell", "direct_page"]
    assert result.title == "Direct iCIMS Page"


# ── LLM typography lint repairs (freeform path) ───────────────────────────────


def test_lint_latex_document_converts_markdown_bold_to_textbf() -> None:
    document = (
        "\\documentclass{article}\\begin{document}"
        "Built **Python** services with **Docker**."
        "\\end{document}"
    )
    result = lint_latex_document_for_compile(document)
    assert "\\textbf{Python}" in result.latex_document
    assert "\\textbf{Docker}" in result.latex_document
    assert "**" not in result.latex_document
    assert "converted markdown **bold** to \\textbf" in result.repairs_applied


def test_lint_latex_document_restores_bare_textbf_backslash() -> None:
    document = (
        "\\documentclass{article}\\begin{document}"
        "Built textbf{Python} services. Kept \\textbf{Docker} intact."
        "\\end{document}"
    )
    result = lint_latex_document_for_compile(document)
    assert "\\textbf{Python}" in result.latex_document
    # The already-correct command must not gain a second backslash.
    assert "\\\\textbf{Docker}" not in result.latex_document
    assert "restored missing backslashes on text commands" in result.repairs_applied


def test_lint_latex_document_normalizes_smart_quotes() -> None:
    document = (
        "\\documentclass{article}\\begin{document}"
        "Improved the team’s “mission critical” tooling."
        "\\end{document}"
    )
    result = lint_latex_document_for_compile(document)
    assert "’" not in result.latex_document
    assert "“" not in result.latex_document and "”" not in result.latex_document
    assert "``mission critical''" in result.latex_document
    assert "team's" in result.latex_document
    assert "normalized smart quotes to LaTeX quote ligatures" in result.repairs_applied


def test_lint_latex_document_does_not_report_backslash_repair_for_trailing_whitespace() -> None:
    # Trailing whitespace alone must not trigger the dangling-backslash repair
    # label (it previously fired on every structured-renderer document).
    document = (
        "\\documentclass{article}   \n"
        "\\begin{document}\t\n"
        "Built \\textbf{Python} services. \n"
        "\\end{document}"
    )
    result = lint_latex_document_for_compile(document)
    assert result.latex_document == document
    assert result.repairs_applied == []


def test_lint_latex_document_leaves_clean_document_unchanged() -> None:
    document = (
        "\\documentclass{article}\\begin{document}"
        "Built \\textbf{Python} services with \\textit{care}."
        "\\end{document}"
    )
    result = lint_latex_document_for_compile(document)
    assert result.latex_document == document
    assert result.repairs_applied == []


# ── Compile page-count detection ──────────────────────────────────────────────


def test_compile_latex_to_pdf_reports_page_count(monkeypatch, tmp_path: Path) -> None:
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
        stdout = "Output written on Python-Developer.pdf (2 pages, 48213 bytes)."
        return subprocess.CompletedProcess(command, 0, stdout, "")

    result = compile_latex_to_pdf(
        latex_document="\\documentclass{article}\\begin{document}Hi\\end{document}",
        job_title="Python Developer",
        template_path=template_path,
        log_path=tmp_path / "template.log",
        command_runner=_runner,
    )

    assert result.status == "ok"
    assert result.page_count == 2


# ── Iterative LLM compile repair ──────────────────────────────────────────────


class _FakeCompileResult:
    def __init__(self, status: str, log_excerpt: str | None = None, page_count: int | None = None):
        self.status = status
        self.log_excerpt = log_excerpt
        self.page_count = page_count


def test_repair_latex_until_compiles_converges_over_two_rounds(monkeypatch) -> None:
    from services.resumes.listing import repair_latex_until_compiles

    settings = GeminiSettings(api_key="key", model=DEFAULT_GEMINI_MODEL)
    repair_calls: list[str] = []

    def fake_repair(document, error_excerpt, passed_settings, client_factory=None):
        repair_calls.append(error_excerpt)
        if len(repair_calls) == 1:
            return "\\documentclass{article}fixed-once\\end{document}", "gemini"
        return "\\documentclass{article}fixed-twice\\end{document}", "openrouter"

    monkeypatch.setattr("services.resumes.listing.repair_latex_with_llm", fake_repair)

    compile_calls: list[str] = []

    def fake_compiler(document: str):
        compile_calls.append(document)
        if "fixed-twice" in document:
            return _FakeCompileResult("ok")
        return _FakeCompileResult("error", log_excerpt="second error: undefined control sequence")

    initial = _FakeCompileResult("error", log_excerpt="first error: unbalanced brace")
    final_doc, final_result, provider, rounds = repair_latex_until_compiles(
        "\\documentclass{article}broken\\end{document}", initial, settings, fake_compiler
    )

    assert rounds == 2
    assert provider == "openrouter"
    assert final_result.status == "ok"
    assert "fixed-twice" in final_doc
    # Round 2 must have been fed the fresh error from round 1's compile, not the original.
    assert repair_calls[1] == "second error: undefined control sequence"


def test_repair_latex_until_compiles_returns_original_when_all_rounds_fail(monkeypatch) -> None:
    from services.resumes.listing import repair_latex_until_compiles

    settings = GeminiSettings(api_key="key", model=DEFAULT_GEMINI_MODEL)

    monkeypatch.setattr(
        "services.resumes.listing.repair_latex_with_llm",
        lambda document, error, passed_settings, client_factory=None: (
            "\\documentclass{article}still-broken\\end{document}",
            "gemini",
        ),
    )

    def fake_compiler(document: str):
        return _FakeCompileResult("error", log_excerpt="persistent error")

    original_doc = "\\documentclass{article}broken\\end{document}"
    initial = _FakeCompileResult("error", log_excerpt="initial error")
    final_doc, final_result, provider, rounds = repair_latex_until_compiles(
        original_doc, initial, settings, fake_compiler, max_rounds=2
    )

    assert rounds == 2
    assert provider is None
    assert final_doc == original_doc
    assert final_result is initial


def test_repair_latex_until_compiles_stops_when_no_provider_produces_fix(monkeypatch) -> None:
    from services.resumes.listing import repair_latex_until_compiles

    settings = GeminiSettings(api_key="key", model=DEFAULT_GEMINI_MODEL)
    monkeypatch.setattr(
        "services.resumes.listing.repair_latex_with_llm",
        lambda document, error, passed_settings, client_factory=None: (None, None),
    )

    compile_calls: list[str] = []

    def fake_compiler(document: str):
        compile_calls.append(document)
        return _FakeCompileResult("error", log_excerpt="err")

    initial = _FakeCompileResult("error", log_excerpt="initial error")
    final_doc, final_result, provider, rounds = repair_latex_until_compiles(
        "doc", initial, settings, fake_compiler
    )

    assert rounds == 0
    assert provider is None
    assert compile_calls == []
    assert final_result is initial


# ── LLM page-overflow condensing ──────────────────────────────────────────────


def test_condense_latex_if_overflowing_adopts_shorter_compiling_version(monkeypatch) -> None:
    from services.resumes.listing import condense_latex_if_overflowing

    settings = GeminiSettings(api_key="key", model=DEFAULT_GEMINI_MODEL)
    monkeypatch.setattr(
        "services.resumes.listing._generate_fixed_latex_with_providers",
        lambda prompt, original, passed_settings, client_factory=None: (
            "\\documentclass{article}condensed\\end{document}",
            "gemini",
        ),
    )

    def fake_compiler(document: str):
        return _FakeCompileResult("ok", page_count=1)

    overflowing = _FakeCompileResult("ok", page_count=2)
    final_doc, final_result, provider = condense_latex_if_overflowing(
        "\\documentclass{article}long\\end{document}", overflowing, 1, settings, fake_compiler
    )

    assert provider == "gemini"
    assert "condensed" in final_doc
    assert final_result.page_count == 1


def test_condense_latex_if_overflowing_keeps_original_when_condense_does_not_shrink(monkeypatch) -> None:
    from services.resumes.listing import condense_latex_if_overflowing

    settings = GeminiSettings(api_key="key", model=DEFAULT_GEMINI_MODEL)
    monkeypatch.setattr(
        "services.resumes.listing._generate_fixed_latex_with_providers",
        lambda prompt, original, passed_settings, client_factory=None: (
            "\\documentclass{article}condensed\\end{document}",
            "gemini",
        ),
    )

    def fake_compiler(document: str):
        return _FakeCompileResult("ok", page_count=2)

    original_doc = "\\documentclass{article}long\\end{document}"
    overflowing = _FakeCompileResult("ok", page_count=2)
    final_doc, final_result, provider = condense_latex_if_overflowing(
        original_doc, overflowing, 1, settings, fake_compiler
    )

    assert provider is None
    assert final_doc == original_doc
    assert final_result is overflowing


def test_condense_latex_if_overflowing_skips_when_within_page_budget(monkeypatch) -> None:
    from services.resumes.listing import condense_latex_if_overflowing

    settings = GeminiSettings(api_key="key", model=DEFAULT_GEMINI_MODEL)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM should not be invoked when the page budget is met")

    monkeypatch.setattr("services.resumes.listing._generate_fixed_latex_with_providers", fail_if_called)

    within_budget = _FakeCompileResult("ok", page_count=1)
    final_doc, final_result, provider = condense_latex_if_overflowing(
        "doc", within_budget, 1, settings, lambda document: None
    )

    assert provider is None
    assert final_result is within_budget


def test_condense_latex_if_overflowing_keeps_original_when_condensed_fails_to_compile(monkeypatch) -> None:
    from services.resumes.listing import condense_latex_if_overflowing

    settings = GeminiSettings(api_key="key", model=DEFAULT_GEMINI_MODEL)
    monkeypatch.setattr(
        "services.resumes.listing._generate_fixed_latex_with_providers",
        lambda prompt, original, passed_settings, client_factory=None: (
            "\\documentclass{article}broken-condensed\\end{document}",
            "gemini",
        ),
    )

    def fake_compiler(document: str):
        return _FakeCompileResult("error", log_excerpt="condense broke it")

    original_doc = "\\documentclass{article}long\\end{document}"
    overflowing = _FakeCompileResult("ok", page_count=3)
    final_doc, final_result, provider = condense_latex_if_overflowing(
        original_doc, overflowing, 1, settings, fake_compiler
    )

    assert provider is None
    assert final_doc == original_doc
    assert final_result is overflowing


def test_build_latex_condense_prompt_mentions_pages_and_forbids_deletion() -> None:
    from services.resumes.listing import build_latex_condense_prompt

    prompt = build_latex_condense_prompt("\\documentclass{article}body\\end{document}", 2, 1)

    assert "2 pages" in prompt
    assert "at most 1 page(s)" in prompt
    assert "never delete lines" in prompt
    assert "\\documentclass{article}body\\end{document}" in prompt


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

    # Suppress env/dotenv so provider selection is deterministic for this test.
    monkeypatch.setattr(_lm, "_groq_api_key", lambda: None)

    post_calls: list[str] = []

    def _mock_post(url: str, **kwargs: object) -> object:
        post_calls.append(url)
        return _make_openai_compat_response(
            json.dumps({"summary": "ok", "targeted_bullets": ["a"], "revised_profile": "b", "rewritten_tex": "\\documentclass{article}\\begin{document}Hi\\end{document}"})
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
            json.dumps({"summary": "ok", "targeted_bullets": ["a"], "revised_profile": "b", "rewritten_tex": "\\documentclass{article}\\begin{document}Hi\\end{document}"})
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


def test_sanitize_latex_document_does_not_escape_trailing_percent_whitespace_suppression() -> None:
    """Trailing % (whitespace-suppression idiom) inside braces must NOT become \\%."""
    source = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\newcommand{\\project}[4]{%\n"
        "  \\vspace{2pt}%\n"
        "  #1\n"
        "}\n"
        "\\centerline{%\n"
        "  content\n"
        "}\n"
        "\\end{document}\n"
    )
    fixed = sanitize_latex_document_for_compile(source)
    assert "\\newcommand{\\project}[4]{%\n" in fixed, "trailing % in \\newcommand must stay as comment"
    assert "  \\vspace{2pt}%\n" in fixed, "trailing % after \\vspace must stay as comment"
    assert "\\centerline{%\n" in fixed, "trailing % after \\centerline{ must stay as comment"


def test_sanitize_latex_document_still_escapes_mid_content_percent() -> None:
    """% that appears mid-argument (not trailing) must still be escaped to \\%."""
    source = "\\resumeItem{Improved throughput by 15% for all services}"
    fixed = sanitize_latex_document_for_compile(source)
    assert "15\\%" in fixed


def test_sanitize_latex_document_fixes_resume_subheading_with_3_args() -> None:
    """\\resumeSubheading with only 3 args must be padded to 4 empty groups."""
    source = (
        "\\section{Experience}\n"
        "\\resumeSubHeadingListStart\n"
        "\\resumeSubheading{Software Engineer}{2020--2022}{Acme Corp}\n"
        "\\resumeSubHeadingListEnd\n"
    )
    fixed = sanitize_latex_document_for_compile(source)
    assert "\\resumeSubheading{Software Engineer}{2020--2022}{Acme Corp}{}" in fixed


def test_sanitize_latex_document_keeps_resume_subheading_with_4_args() -> None:
    """\\resumeSubheading with the correct 4 args must not be modified."""
    source = (
        "\\resumeSubheading{Software Engineer}{2020--2022}{Acme Corp}{Toronto, ON}\n"
    )
    fixed = sanitize_latex_document_for_compile(source)
    assert "\\resumeSubheading{Software Engineer}{2020--2022}{Acme Corp}{Toronto, ON}" in fixed


def test_sanitize_latex_document_removes_duplicate_preamble() -> None:
    """A duplicate \\documentclass (LLM regeneration) must drop the first preamble."""
    source = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "garbage from first attempt\n"
        "\\documentclass{article}\n"
        "\\usepackage{hyperref}\n"
        "\\begin{document}\n"
        "real content\n"
        "\\end{document}\n"
    )
    fixed = sanitize_latex_document_for_compile(source)
    assert "garbage from first attempt" not in fixed
    assert "real content" in fixed
    assert fixed.count("\\documentclass") == 1


def test_close_unclosed_explicit_itemize_on_truncation() -> None:
    """Truncated \\begin{itemize} without \\end{itemize} must be auto-closed."""
    source = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\begin{itemize}\n"
        "  \\item First\n"
        "  \\item Second\n"
        "\\end{document}\n"
    )
    fixed = sanitize_latex_document_for_compile(source)
    assert fixed.count("\\end{itemize}") == 1
    end_pos = fixed.index("\\end{document}")
    close_pos = fixed.index("\\end{itemize}")
    assert close_pos < end_pos, "\\end{itemize} must appear before \\end{document}"


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


# ── LLM-based compile-error repair (last resort after deterministic fixes) ────


def test_build_latex_repair_prompt_includes_error_and_broken_latex() -> None:
    from services.resumes.listing import build_latex_repair_prompt

    prompt = build_latex_repair_prompt(
        "\\documentclass{article}\\begin{document}\\resumeSubheading{a}{b}\\end{document}",
        "! Undefined control sequence.\nl.12 \\resumeSubheading",
    )

    assert "Undefined control sequence" in prompt
    assert "\\resumeSubheading" in prompt
    assert "<broken_latex>" in prompt
    assert "<compile_error>" in prompt


def test_repair_latex_with_llm_returns_fixed_document_from_gemini(monkeypatch) -> None:
    from services.resumes.listing import repair_latex_with_llm

    settings = GeminiSettings(api_key="gemini-key", model="gemini-2.5-flash")
    broken = "\\documentclass{article}\\begin{document}broken\\end{document}"
    fixed = "\\documentclass{article}\\begin{document}fixed\\end{document}"

    class _FakeResponse:
        text = f"<latex>{fixed}</latex>"

    class _FakeModels:
        def generate_content(self, **kwargs):
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    fixed_latex, provider = repair_latex_with_llm(
        broken,
        "! LaTeX Error: something broke.",
        settings,
        client_factory=lambda api_key: _FakeClient(),
    )

    assert fixed_latex == fixed
    assert provider == "gemini"


def test_repair_latex_with_llm_falls_back_to_openrouter_when_gemini_unavailable(monkeypatch) -> None:
    from services.resumes.listing import repair_latex_with_llm
    import services.resumes.listing as _lm

    settings = GeminiSettings(api_key=None, model="gemini-2.5-flash", openrouter_api_key="or-key")
    monkeypatch.setattr(_lm, "_groq_api_key", lambda: None)

    broken = "\\documentclass{article}\\begin{document}broken\\end{document}"
    fixed = "\\documentclass{article}\\begin{document}fixed\\end{document}"

    monkeypatch.setattr(
        _lm.requests,
        "post",
        lambda *a, **kw: _make_openai_compat_response(f"<latex>{fixed}</latex>"),
    )

    fixed_latex, provider = repair_latex_with_llm(broken, "! LaTeX Error: something broke.", settings)

    assert fixed_latex == fixed
    assert provider == "openrouter"


def test_repair_latex_with_llm_returns_none_when_no_providers_configured(monkeypatch) -> None:
    from services.resumes.listing import repair_latex_with_llm
    import services.resumes.listing as _lm

    # Explicitly neutralize env/.env fallback keys so this exercises the
    # "nothing configured" path regardless of what's in the real .env file.
    monkeypatch.setattr(_lm, "_openrouter_api_key", lambda: None)
    monkeypatch.setattr(_lm, "_groq_api_key", lambda: None)

    settings = GeminiSettings(api_key=None, model="gemini-2.5-flash")
    broken = "\\documentclass{article}\\begin{document}broken\\end{document}"

    fixed_latex, provider = repair_latex_with_llm(broken, "! LaTeX Error: something broke.", settings)

    assert fixed_latex is None
    assert provider is None


def test_repair_latex_with_llm_rejects_response_identical_to_input(monkeypatch) -> None:
    """If the model echoes the broken document back unchanged, that isn't a
    fix — treat it the same as no fix so the caller doesn't loop forever."""
    from services.resumes.listing import repair_latex_with_llm
    import services.resumes.listing as _lm

    monkeypatch.setattr(_lm, "_openrouter_api_key", lambda: None)
    monkeypatch.setattr(_lm, "_groq_api_key", lambda: None)

    settings = GeminiSettings(api_key="gemini-key", model="gemini-2.5-flash")
    broken = "\\documentclass{article}\\begin{document}broken\\end{document}"

    class _FakeResponse:
        text = f"<latex>{broken}</latex>"

    class _FakeModels:
        def generate_content(self, **kwargs):
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    fixed_latex, provider = repair_latex_with_llm(
        broken,
        "! LaTeX Error: something broke.",
        settings,
        client_factory=lambda api_key: _FakeClient(),
    )

    assert fixed_latex is None
    assert provider is None


def test_repair_latex_with_llm_rejects_response_without_valid_latex(monkeypatch) -> None:
    from services.resumes.listing import repair_latex_with_llm
    import services.resumes.listing as _lm

    monkeypatch.setattr(_lm, "_openrouter_api_key", lambda: None)
    monkeypatch.setattr(_lm, "_groq_api_key", lambda: None)

    settings = GeminiSettings(api_key="gemini-key", model="gemini-2.5-flash")
    broken = "\\documentclass{article}\\begin{document}broken\\end{document}"

    class _FakeResponse:
        text = "Sorry, I can't help with that."

    class _FakeModels:
        def generate_content(self, **kwargs):
            return _FakeResponse()

    class _FakeClient:
        models = _FakeModels()

    fixed_latex, provider = repair_latex_with_llm(
        broken,
        "! LaTeX Error: something broke.",
        settings,
        client_factory=lambda api_key: _FakeClient(),
    )

    assert fixed_latex is None
    assert provider is None


def test_repair_latex_with_llm_falls_through_on_provider_exception(monkeypatch) -> None:
    from services.resumes.listing import repair_latex_with_llm
    import services.resumes.listing as _lm

    settings = GeminiSettings(api_key="gemini-key", model="gemini-2.5-flash", openrouter_api_key="or-key")
    monkeypatch.setattr(_lm, "_groq_api_key", lambda: None)

    broken = "\\documentclass{article}\\begin{document}broken\\end{document}"
    fixed = "\\documentclass{article}\\begin{document}fixed\\end{document}"

    monkeypatch.setattr(
        _lm.requests,
        "post",
        lambda *a, **kw: _make_openai_compat_response(f"<latex>{fixed}</latex>"),
    )

    fixed_latex, provider = repair_latex_with_llm(
        broken,
        "! LaTeX Error: something broke.",
        settings,
        client_factory=lambda api_key: (_ for _ in ()).throw(RuntimeError("Gemini SDK unavailable")),
    )

    assert fixed_latex == fixed
    assert provider == "openrouter"
