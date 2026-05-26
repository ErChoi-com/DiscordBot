
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from services import job_service

def test_build_keyword_variants_unquoted():
    # Unquoted: biomedical engineering intern
    result = job_service.build_keyword_variants("biomedical engineering intern")
    assert "biomedical intern" in result
    assert "engineering intern" in result
    assert "biomedical engineering intern" in result
    assert len(result) >= 3

def test_build_keyword_variants_quoted():
    # Quoted: "biomedical engineer" intern
    result = job_service.build_keyword_variants('"biomedical engineer" intern')
    assert "biomedical engineer intern" in result
    assert '"biomedical engineer" intern' in result or 'biomedical engineer intern' in result
    assert len(result) >= 2

def test_build_keyword_variants_mixed():
    # Mixed: remote "data scientist" entry
    result = job_service.build_keyword_variants('remote "data scientist" entry')
    assert "remote entry" in result
    assert "data scientist entry" in result
    assert "remote data scientist entry" in result or 'remote "data scientist" entry' in result
    assert len(result) >= 3

def test_build_keyword_variants_multiple_quoted():
    # Multiple quoted: "data scientist" "machine learning" remote
    result = job_service.build_keyword_variants('"data scientist" "machine learning" remote')
    assert "data scientist remote" in result
    assert "machine learning remote" in result
    assert "data scientist machine learning remote" in result or '"data scientist" "machine learning" remote' in result
    assert len(result) >= 3

def test_build_keyword_variants_unmatched_quote():
    # Unmatched quote: "biomedical engineer intern
    result = job_service.build_keyword_variants('"biomedical engineer intern')
    assert any("biomedical engineer intern" in v for v in result)
    assert len(result) >= 1
