"""HTTPS verification in the ATS scrapers uses one CA store, loaded once.

requests gives urllib3 the CA-bundle path for every connection and urllib3
builds a new SSLContext and parses the bundle each time -- per connection, not
per Session. A py-spy dump of the live bot found 166 of 541 threads inside that
parse. Pooling only helps hosts we reconnect to, and most ATS platforms are one
host per company, so the adapter supplies a preloaded context and withholds
the path when verification is the default.
"""
from __future__ import annotations

import ssl
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import ats_service as a  # noqa: E402


def _adapter() -> a._PreloadedTLSAdapter:
    return a._PreloadedTLSAdapter()


def test_the_context_really_verifies_with_the_default_bundle():
    ctx = a._PreloadedTLSAdapter.context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname
    assert ctx.cert_store_stats()["x509_ca"] > 50, "the default CA bundle must be loaded into it"


def test_the_context_is_built_once_and_shared():
    seen: list[int] = []

    def _grab():
        seen.append(id(a._PreloadedTLSAdapter.context()))

    threads = [threading.Thread(target=_grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(set(seen)) == 1


def test_the_pool_manager_carries_the_preloaded_context():
    adapter = _adapter()
    assert adapter.poolmanager.connection_pool_kw["ssl_context"] is a._PreloadedTLSAdapter.context()


def test_default_verification_withholds_the_per_connection_bundle_path():
    """cert_reqs stays CERT_REQUIRED -- verification is on -- but urllib3 gets
    no path to reload; the preloaded store does the checking."""
    conn = SimpleNamespace(cert_reqs=None, ca_certs="stale", ca_cert_dir="stale")
    _adapter().cert_verify(conn, "https://boards.example", True, None)
    assert conn.cert_reqs == "CERT_REQUIRED"
    assert conn.ca_certs is None
    assert conn.ca_cert_dir is None


def test_an_explicit_bundle_still_reaches_urllib3(tmp_path):
    """verify="/path" (what REQUESTS_CA_BUNDLE becomes) is the operator
    overriding the default store, and must keep working exactly as before."""
    bundle = tmp_path / "corp.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----\n")
    conn = SimpleNamespace(cert_reqs=None, ca_certs=None, ca_cert_dir=None)
    _adapter().cert_verify(conn, "https://boards.example", str(bundle), None)
    assert conn.cert_reqs == "CERT_REQUIRED"
    assert conn.ca_certs == str(bundle)


def test_verification_off_is_still_off():
    conn = SimpleNamespace(cert_reqs=None, ca_certs=None, ca_cert_dir=None)
    _adapter().cert_verify(conn, "https://boards.example", False, None)
    assert conn.cert_reqs == "CERT_NONE"


def test_plain_http_is_untouched():
    conn = SimpleNamespace(cert_reqs=None, ca_certs=None, ca_cert_dir=None)
    _adapter().cert_verify(conn, "http://boards.example", True, None)
    assert conn.cert_reqs == "CERT_NONE"


def test_the_thread_session_mounts_the_adapter_for_https_only(monkeypatch):
    monkeypatch.setattr(a, "_HTTP_LOCAL", threading.local())
    session = a._http()
    assert isinstance(session.get_adapter("https://boards.example/x"), a._PreloadedTLSAdapter)
    assert not isinstance(session.get_adapter("http://boards.example/x"), a._PreloadedTLSAdapter)
    assert isinstance(session.get_adapter("http://boards.example/x"), requests.adapters.HTTPAdapter)
