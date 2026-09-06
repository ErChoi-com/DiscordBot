"""Per-member shares of the expensive LLM-backed commands.

These drive the real policy resolution, the real store round-trip and the real
scaling helpers -- the only thing stubbed is Discord itself, which is replaced
by plain objects carrying the ids and roles the resolver reads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from services import quota  # noqa: E402
from state.store import RuntimeStore  # noqa: E402


# ---------------------------------------------------------------------------
# Parsing the argument
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["0", "0.5", "1", "1.0", " 0.25 ", 0.75])
def test_valid_shares_are_accepted(raw):
    assert quota.clamp_share(raw) is not None


@pytest.mark.parametrize("raw", ["-0.1", "1.1", "50%", "half", "", None, "nan", "inf"])
def test_anything_outside_zero_to_one_is_rejected(raw):
    """Rejected rather than clamped: "50%" clamped to 1.0 would silently grant
    a full allowance the owner never meant to give."""
    assert quota.clamp_share(raw) is None


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------

def test_the_owner_is_never_limited():
    """The owner pays for the bot; a quota that could lock them out of their
    own server is a footgun with no upside."""
    policy = quota.QuotaPolicy(default_share=0.0, user_shares={5: 0.0})
    assert quota.resolve_share(policy, user_id=5, is_owner=True) == quota.FULL_SHARE
    assert not quota.allowance_for(policy, user_id=5, is_owner=True).blocked


def test_a_manual_override_beats_a_role_and_the_default():
    policy = quota.QuotaPolicy(
        default_share=0.2, role_shares={10: 0.9}, user_shares={5: 0.4}
    )
    assert quota.resolve_share(policy, user_id=5, role_ids=(10,)) == 0.4


def test_the_most_generous_role_wins():
    """Roles are additive everywhere else in Discord, so a member handed a more
    privileged role expects it to help rather than to be averaged away."""
    policy = quota.QuotaPolicy(default_share=0.1, role_shares={10: 0.3, 11: 0.8})
    assert quota.resolve_share(policy, user_id=5, role_ids=(10, 11)) == 0.8


def test_an_unmatched_member_gets_the_default():
    policy = quota.QuotaPolicy(default_share=0.6, role_shares={10: 0.9})
    assert quota.resolve_share(policy, user_id=5, role_ids=(99,)) == 0.6


def test_the_default_default_changes_nothing():
    """Adding quotas to a server that never configured them must not silently
    degrade it."""
    assert quota.allowance_for(quota.QuotaPolicy(), user_id=5).full


# ---------------------------------------------------------------------------
# What a share actually does
# ---------------------------------------------------------------------------

def test_a_share_scales_the_expensive_knobs():
    half = quota.Allowance(0.5)
    assert half.scale(60) == 30
    assert half.scale(28) == 14


def test_scaling_never_reaches_zero():
    """A shortlist scaled to nothing would return an empty ranking, which reads
    as a bug. A blocked member is refused up front instead -- a different and
    clearly explained outcome."""
    tiny = quota.Allowance(0.06)
    assert tiny.scale(60, minimum=5) == 5
    assert tiny.scale(28, minimum=2) == 2
    assert not tiny.blocked


def test_a_full_share_returns_the_value_untouched():
    assert quota.Allowance(1.0).scale(60) == 60


def test_optional_passes_drop_in_order_of_cost():
    """A reduced share loses marginal passes first and the core generation
    last, so a cheaper resume is still a resume."""
    assert quota.Allowance(1.0).permits("skills_rewrite")
    assert not quota.Allowance(0.7).permits("skills_rewrite")
    assert quota.Allowance(0.7).permits("grounding_audit")
    assert not quota.Allowance(0.5).permits("grounding_audit")
    assert quota.Allowance(0.5).permits("domain_fit_audit")
    assert not quota.Allowance(0.3).permits("domain_fit_audit")


def test_an_unregistered_feature_is_permitted():
    """A new pass added elsewhere must not silently vanish for every non-owner
    because nobody remembered to register a threshold here."""
    assert quota.Allowance(0.1).permits("some_future_pass")


def test_a_zero_share_blocks_the_command():
    assert quota.Allowance(0.0).blocked
    assert quota.Allowance(0.0).describe() == "no access"


def test_describe_names_what_a_share_gives_up():
    text = quota.Allowance(0.5).describe()
    assert "50%" in text
    assert "skills_rewrite" in text and "grounding_audit" in text
    assert "domain_fit_audit" not in text        # still affordable at 0.5


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_a_policy_survives_a_store_round_trip(tmp_path: Path):
    path = tmp_path / "state.json"
    store = RuntimeStore(path)
    store.set_quota_policy(
        123, quota.QuotaPolicy(default_share=0.5, role_shares={10: 0.8},
                               user_shares={5: 0.2})
    )

    reloaded = RuntimeStore(path)
    reloaded.load()
    policy = reloaded.get_quota_policy(123)

    assert policy.default_share == 0.5
    assert policy.role_shares == {10: 0.8}
    assert policy.user_shares == {5: 0.2}


def test_an_unconfigured_guild_gets_defaults(tmp_path: Path):
    store = RuntimeStore(tmp_path / "state.json")
    assert store.get_quota_policy(999).default_share == quota.DEFAULT_SHARE
    assert store.get_quota_policy(None).default_share == quota.DEFAULT_SHARE


def test_a_corrupt_entry_costs_that_entry_not_the_bot():
    """Read on every command and at startup, so tolerance matters more than
    strictness."""
    policy = quota.QuotaPolicy.from_dict({
        "default_share": "not a number",
        "role_shares": {"10": 0.8, "bad": 0.5, "11": 7.0},
        "user_shares": "not a dict",
    })

    assert policy.default_share == quota.DEFAULT_SHARE   # fell back
    assert policy.role_shares == {10: 0.8}               # bad id and 7.0 dropped
    assert policy.user_shares == {}


def test_from_dict_tolerates_junk():
    assert quota.QuotaPolicy.from_dict(None).default_share == quota.DEFAULT_SHARE
    assert quota.QuotaPolicy.from_dict("nonsense").role_shares == {}


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

def test_quota_payloads_parse_into_action_target_value():
    from commands.handlers import parse_quota_payload

    assert parse_quota_payload("") == ("show", "", "")
    assert parse_quota_payload("default 0.5") == ("default", "", "0.5")
    assert parse_quota_payload("user <@123> 0.3") == ("user", "<@123>", "0.3")
    assert parse_quota_payload("role <@&99> 0.7") == ("role", "<@&99>", "0.7")
    assert parse_quota_payload("clear user <@123>") == ("clear", "user", "<@123>")
    assert parse_quota_payload("wat") == ("unknown", "", "")


def test_quota_command_is_reachable_through_dispatch(tmp_path: Path):
    """Regression: handle_quota was fully implemented (and documented in the
    `.cmd` cheatsheet as owner-only) but was never added to
    CommandRouter.handlers, so typing `.quota` in Discord silently did
    nothing -- dispatch() fell through every other handler and returned
    False without ever calling it. Caught by auditing what actually responds
    to the command, not just what parse_quota_payload does in isolation.
    """
    import asyncio

    from commands.handlers import CommandRouter
    from services.health import WatcherHealthTracker
    from watchers.manager import WatcherManager

    class _Config:
        resume_profiles_dir = tmp_path / "resumes"
        resume_cache_dir = tmp_path / ".resume_cache"
        base_dir = tmp_path
        main_user_profile_key = "owner-profile"

    _Config.resume_profiles_dir.mkdir(parents=True, exist_ok=True)
    _Config.resume_cache_dir.mkdir(parents=True, exist_ok=True)

    store = RuntimeStore(tmp_path / ".bot_state.json")
    health = WatcherHealthTracker()
    manager = WatcherManager(client=object(), config=_Config(), store=store, health=health)
    router = CommandRouter(
        client=object(), config=_Config(), store=store, watcher_manager=manager, health=health
    )

    class _Channel:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, content=None, **kwargs):
            self.sent.append(content or "")
            return type("SentMessage", (), {"id": 1})()

    channel = _Channel()
    message = type("FakeMessage", (), {
        "content": ".quota",
        "channel": channel,
        "author": type("Author", (), {"id": 7, "name": "owner"})(),
        "guild": type("Guild", (), {"id": 555, "owner_id": 7})(),
        "id": 1,
        "to_reference": lambda self, fail_if_not_exists=False: None,
    })()

    async def dispatch() -> bool:
        for handler in router.handlers:
            if await handler(message):
                return True
        return False

    handled = asyncio.run(dispatch())

    assert handled is True, ".quota is not reachable through CommandRouter.dispatch()"
    assert channel.sent, "handle_quota ran but sent no response"


def test_a_router_without_a_store_still_runs_its_commands():
    """The quota lookup must fail open. A limiter that can take a command down
    is worse than no limiter -- caught by the resume wiring tests, which build
    a router with no persistence at all."""
    from commands.handlers import CommandRouter

    router = object.__new__(CommandRouter)
    message = type(
        "M", (), {"guild": None, "author": type("A", (), {"id": 5, "roles": []})()}
    )()

    allowance = CommandRouter._member_allowance(router, message)
    assert allowance.full
    assert not allowance.blocked
