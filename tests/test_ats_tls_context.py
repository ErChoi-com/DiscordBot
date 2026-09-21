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


# ── the environment is read once per Session, not once per request ──────────

def test_the_thread_session_does_not_reconsult_the_environment_per_request(monkeypatch):
    """trust_env makes requests walk the Windows proxy registry and look up
    ~/.netrc on every call; the profile showed both beside the TLS work."""
    monkeypatch.setattr(a, "_HTTP_LOCAL", threading.local())
    assert a._http().trust_env is False


def test_a_proxy_environment_is_still_honoured_once(monkeypatch):
    monkeypatch.setattr(a, "_HTTP_LOCAL", threading.local())
    monkeypatch.setattr(a.requests.utils, "getproxies", lambda: {"https": "http://proxy.corp:3128"})
    assert a._http().proxies["https"] == "http://proxy.corp:3128"


def test_a_ca_bundle_environment_is_still_honoured_once(monkeypatch, tmp_path):
    monkeypatch.setattr(a, "_HTTP_LOCAL", threading.local())
    bundle = tmp_path / "corp.pem"
    bundle.write_text("x")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(bundle))
    assert a._http().verify == str(bundle)


def test_no_environment_means_default_verification(monkeypatch):
    monkeypatch.setattr(a, "_HTTP_LOCAL", threading.local())
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    monkeypatch.setattr(a.requests.utils, "getproxies", lambda: {})
    session = a._http()
    assert session.verify is True
    assert session.proxies == {}


# ── each thread keeps a small pool, so idle host connections close ───────────

def test_the_thread_session_keeps_a_small_pool(monkeypatch):
    """Measured at the start of a fan-out: 1,161 established connections from
    this process, most of them idle keep-alives to company hosts a thread
    would never ask again, and the gateway heartbeat's ack ~42s late while
    the loop was clear."""
    monkeypatch.setattr(a, "_HTTP_LOCAL", threading.local())
    session = a._http()
    for scheme in ("https://x/", "http://x/"):
        adapter = session.get_adapter(scheme)
        assert adapter._pool_connections == a.HTTP_POOL_HOSTS_PER_THREAD
        assert adapter._pool_maxsize == a.HTTP_POOL_PER_HOST
    assert a.HTTP_POOL_HOSTS_PER_THREAD <= 2 and a.HTTP_POOL_PER_HOST <= 2


def test_moving_to_a_new_host_evicts_the_oldest_hosts_pool(monkeypatch):
    """The point of the small pool, exercised against urllib3 itself: with
    two host pools, asking a third host closes the first host's pool."""
    monkeypatch.setattr(a, "_HTTP_LOCAL", threading.local())
    pm = a._http().get_adapter("https://x/").poolmanager
    pools = [pm.connection_from_host(f"company-{i}.example", 443, scheme="https") for i in range(3)]
    assert len(pm.pools) <= a.HTTP_POOL_HOSTS_PER_THREAD
    # Evicted means forgotten: asking the first host again builds a new pool
    # (and, with nothing else referencing the old one, its sockets close).
    assert pm.connection_from_host("company-0.example", 443, scheme="https") is not pools[0]
    assert pm.connection_from_host("company-2.example", 443, scheme="https") is pools[2]
