"""Per-user shares of the expensive work in the LLM-backed commands.

Every command that calls a model spends real money and real network budget:
`.bestjobs` judges a shortlist and fetches descriptions, `.resumebuild` and
`.cover` run a generation pass plus optional audit passes. On a private bot
that is the owner's own quota being spent, so a server with other members in
it needs a way to say how much of it each of them may use.

The knob is one number per user, 0.0 to 1.0, meaning "share of the full
allowance":

    1.0   everything the owner gets
    0.5   half the shortlist, half the descriptions, the cheaper audits only
    0.0   the command is refused

Two primitives, because expensive work comes in two shapes
----------------------------------------------------------
`Allowance.scale()` for the knobs that are numbers -- how many postings reach
the judge, how many descriptions are fetched. Those shrink smoothly.

`Allowance.permits()` for the passes that are all-or-nothing -- an audit either
runs as an extra model call or it does not. Those switch off in order of how
much they cost relative to what they add, so a reduced share loses the
marginal passes first and the core generation last. A resume built at 0.3 is a
worse resume, not a broken one.

Resolution order
----------------
1. The guild owner, always 1.0. The owner pays for the bot; a quota that could
   lock them out of their own server would be a footgun with no upside.
2. A manual per-user share set by the owner.
3. The most generous share among the member's roles. Most generous rather than
   least: roles are additive in every other Discord permission system, and a
   member given a more privileged role expects it to help.
4. The guild default.

This module is deliberately free of discord imports so the policy is testable
without a gateway connection; callers pass ids and role ids in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: What the owner always gets, and the ceiling on anyone else.
FULL_SHARE: float = 1.0

#: What a member gets when nothing else applies. 1.0 keeps behaviour unchanged
#: until the owner actually sets a limit -- installing this module should not
#: silently degrade a server that was working fine. Set a guild default with
#: the quota command to start limiting.
DEFAULT_SHARE: float = 1.0

#: Below this a share is treated as "no access" rather than "a very small
#: allowance", because a scaled-down run still costs a model call and a user
#: given 0.01 would otherwise spend nearly as much as one given 0.2.
BLOCKED_BELOW: float = 0.05

#: Thresholds for the optional passes, cheapest-to-drop first. These are the
#: extra model calls in the resume pipeline; the core generation has no
#: threshold because a resume without it is not a resume.
FEATURE_THRESHOLDS: dict[str, float] = {
    # Ranks entries against the listing's domain. The most valuable of the
    # three, so it survives longest.
    "domain_fit_audit": 0.4,
    # Checks tailored bullets against baseinfo. Valuable, one extra call.
    "grounding_audit": 0.6,
    # Rewrites the skills line for the listing. Nice to have, drops first.
    "skills_rewrite": 0.8,
}


def clamp_share(value: object) -> float | None:
    """Parse a share argument. Returns None when it is not a usable number.

    None rather than a raised exception or a silent default: the caller is a
    chat command, and "0.5" typed as "50%" or "half" should produce a usage
    message rather than a quota nobody intended.
    """
    try:
        share = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if share != share or share in (float("inf"), float("-inf")):  # NaN / inf
        return None
    if not 0.0 <= share <= FULL_SHARE:
        return None
    return share


@dataclass(slots=True)
class QuotaPolicy:
    """One guild's quota configuration."""

    default_share: float = DEFAULT_SHARE
    #: role id -> share
    role_shares: dict[int, float] = field(default_factory=dict)
    #: user id -> share, set manually by the owner; wins over any role
    user_shares: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "default_share": self.default_share,
            "role_shares": {str(k): v for k, v in self.role_shares.items()},
            "user_shares": {str(k): v for k, v in self.user_shares.items()},
        }

    @classmethod
    def from_dict(cls, payload: object) -> "QuotaPolicy":
        """Rebuild from stored JSON, ignoring anything malformed.

        Tolerant because this is read on every command: a hand-edited state
        file with one bad entry should cost that entry, not the bot.
        """
        if not isinstance(payload, dict):
            return cls()
        default = clamp_share(payload.get("default_share"))

        def _shares(raw: object) -> dict[int, float]:
            out: dict[int, float] = {}
            if not isinstance(raw, dict):
                return out
            for key, value in raw.items():
                share = clamp_share(value)
                if share is None:
                    continue
                try:
                    out[int(key)] = share
                except (TypeError, ValueError):
                    continue
            return out

        return cls(
            default_share=DEFAULT_SHARE if default is None else default,
            role_shares=_shares(payload.get("role_shares")),
            user_shares=_shares(payload.get("user_shares")),
        )


def resolve_share(
    policy: QuotaPolicy,
    user_id: int | None = None,
    role_ids: tuple[int, ...] = (),
    is_owner: bool = False,
) -> float:
    """The share this member is entitled to. See the module docstring."""
    if is_owner:
        return FULL_SHARE
    if user_id is not None and user_id in policy.user_shares:
        return policy.user_shares[user_id]
    role_matches = [policy.role_shares[rid] for rid in role_ids if rid in policy.role_shares]
    if role_matches:
        return max(role_matches)
    return policy.default_share


@dataclass(slots=True)
class Allowance:
    """How much of the expensive work one invocation may do."""

    share: float = FULL_SHARE

    @property
    def blocked(self) -> bool:
        return self.share < BLOCKED_BELOW

    @property
    def full(self) -> bool:
        return self.share >= FULL_SHARE

    def scale(self, value: int, minimum: int = 1) -> int:
        """`value` reduced to this share, never below `minimum`.

        The floor matters: a shortlist scaled to zero would make the command
        return nothing at all, which reads as a bug rather than as a limit. A
        blocked user is refused up front instead -- that is a different and
        clearly-explained outcome.
        """
        if self.full:
            return value
        return max(minimum, int(round(value * self.share)))

    def permits(self, feature: str) -> bool:
        """Whether an optional extra model call is within this share.

        An unknown feature name is permitted: a new pass added elsewhere in the
        codebase should not silently vanish for every non-owner because nobody
        remembered to register a threshold here.
        """
        return self.share >= FEATURE_THRESHOLDS.get(feature, 0.0)

    def describe(self) -> str:
        if self.blocked:
            return "no access"
        if self.full:
            return "full allowance"
        dropped = [name for name in FEATURE_THRESHOLDS if not self.permits(name)]
        text = f"{self.share:.0%} allowance"
        if dropped:
            text += " (skips: " + ", ".join(sorted(dropped)) + ")"
        return text


def allowance_for(
    policy: QuotaPolicy,
    user_id: int | None = None,
    role_ids: tuple[int, ...] = (),
    is_owner: bool = False,
) -> Allowance:
    return Allowance(resolve_share(policy, user_id, role_ids, is_owner))
