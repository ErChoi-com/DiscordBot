from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from services import capacity

if TYPE_CHECKING:
    import discord


def _fmt_ago(ts: float) -> str:
    if ts == 0.0:
        return "never"
    delta = time.time() - ts
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        h = int(delta // 3600)
        m = int((delta % 3600) // 60)
        return f"{h}h {m}m ago"
    d = int(delta // 86400)
    h = int((delta % 86400) // 3600)
    return f"{d}d {h}h ago"


def _fmt_uptime(seconds: float) -> str:
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if d > 0:
        return f"{d}d {h}h {m}m"
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


@dataclass
class ScrapeEvent:
    timestamp: float
    raw_count: int
    sent_count: int
    duration_s: float
    error: str | None = None


@dataclass
class JobWatcherHealth:
    channel_id: int
    started_at: float = 0.0
    last_scrape_at: float = 0.0
    last_scrape_duration_s: float = 0.0
    last_raw_count: int = 0
    last_filtered_count: int = 0
    last_sent_count: int = 0
    last_error: str | None = None
    last_error_at: float = 0.0
    last_success_at: float = 0.0
    consecutive_errors: int = 0
    total_scrapes: int = 0
    total_jobs_sent: int = 0
    total_errors: int = 0
    recent_events: list[ScrapeEvent] = field(default_factory=list)


@dataclass
class ATSPlatformHealth:
    platform: str
    last_scrape_at: float = 0.0
    last_job_count: int = 0
    last_new_count: int = 0
    total_jobs: int = 0
    total_new_jobs: int = 0
    total_errors: int = 0
    last_error: str | None = None
    last_was_error: bool = False
    # A scrape that raises is an error. A scrape that returns nothing is not --
    # it is either a platform with no openings or one that refused every
    # request, and those are indistinguishable from the return value alone
    # (workable answers 429 in 0.1s and the scraper reports []). Counting the
    # runs since anything was returned is what separates them: a platform with
    # real coverage does not stay empty run after run.
    consecutive_silent: int = 0
    last_nonempty_at: float = 0.0
    # How much of the fleet a cycle actually reached. The fan-out submits every
    # company and cancels whatever has not finished when the budget runs out,
    # and until this was recorded the only trace was a print -- so "how many
    # companies did we ask?" could not be answered, and a wrong answer to it
    # (2%, when the real figure is 43-100%) went uncorrected for hours.
    last_submitted: int = 0
    last_completed: int = 0
    total_submitted: int = 0
    total_completed: int = 0


@dataclass
class ATSScrapeHealth:
    last_scrape_at: float = 0.0
    last_total_jobs: int = 0
    scrapes_today: int = 0
    daily_cap: int = 4
    total_jobs_logged: int = 0
    per_platform: dict[str, ATSPlatformHealth] = field(default_factory=dict)
    task_alive: bool = False

    # Every platform that exists, not only those that have reported. Without
    # it `.health` is a log of what happened to run rather than a roll-call:
    # before the first cycle the Platforms field was absent entirely, after one
    # report it listed one of sixteen, and a platform disabled in config never
    # appeared at all -- so bamboohr, the largest fleet at 21,291 boards, was
    # invisible rather than shown as switched off.
    roster: tuple[str, ...] = ()
    disabled: frozenset[str] = frozenset()

    # Platforms whose refusal breaker opened during the last cycle, mapped to
    # the streak that opened it. A cut-off platform returns [] exactly like a
    # quiet one, so without this the single most actionable ATS state -- "the
    # host is refusing us and we stopped asking" -- reached the log and nothing
    # else.
    refusing: dict[str, int] = field(default_factory=dict)


@dataclass
class BrowserServiceHealth:
    dispatch_saturated_count: int = 0
    dispatch_timeout_count: int = 0
    dispatch_yield_count: int = 0
    session_probe_success_count: int = 0
    session_probe_failure_count: int = 0
    # Cumulative counts alone let one lucky early probe permanently mask a
    # later, ongoing outage (success_count > 0 forever). Track the outcome of
    # the most recent probe so the health indicator reflects current state.
    last_probe_success: bool | None = None
    resync_attempt_count: int = 0
    resync_success_count: int = 0
    active_profile_path: str = ""
    last_event_at: float = 0.0


@dataclass
class WatchdogRestart:
    timestamp: float
    target: str


@dataclass
class WatchdogHealth:
    total_restarts: int = 0
    last_restart_at: float = 0.0
    last_restart_target: str = ""
    recent_restarts: list[WatchdogRestart] = field(default_factory=list)


class WatcherHealthTracker:
    MAX_RECENT_EVENTS = 8

    MAX_RECENT_RESTARTS = 8

    def __init__(self) -> None:
        self._job: dict[int, JobWatcherHealth] = {}
        self._ats: ATSScrapeHealth = ATSScrapeHealth()
        self._browser: BrowserServiceHealth = BrowserServiceHealth()
        self._watchdog: WatchdogHealth = WatchdogHealth()
        self._started_at: float = time.time()

    # ── Job watcher ──────────────────────────────────────────────────────────

    def record_job_watcher_started(self, channel_id: int) -> None:
        health = self._job.setdefault(channel_id, JobWatcherHealth(channel_id=channel_id))
        health.started_at = time.time()
        health.consecutive_errors = 0

    def begin_scrape(self) -> float:
        return time.time()

    def record_scrape_success(
        self,
        channel_id: int,
        start_ts: float,
        raw_count: int,
        filtered_count: int,
        sent_count: int,
    ) -> None:
        health = self._job.setdefault(channel_id, JobWatcherHealth(channel_id=channel_id))
        now = time.time()
        duration = now - start_ts
        health.last_scrape_at = now
        health.last_success_at = now
        health.last_scrape_duration_s = duration
        health.last_raw_count = raw_count
        health.last_filtered_count = filtered_count
        health.last_sent_count = sent_count
        health.consecutive_errors = 0
        health.total_scrapes += 1
        health.total_jobs_sent += sent_count
        self._push_event(health, ScrapeEvent(now, raw_count, sent_count, duration))

    def record_scrape_error(self, channel_id: int, start_ts: float, error: str) -> None:
        health = self._job.setdefault(channel_id, JobWatcherHealth(channel_id=channel_id))
        now = time.time()
        duration = now - start_ts
        health.last_scrape_at = now
        health.last_error = error[:200]
        health.last_error_at = now
        health.consecutive_errors += 1
        health.total_errors += 1
        health.total_scrapes += 1
        self._push_event(health, ScrapeEvent(now, 0, 0, duration, error=error[:100]))

    def _push_event(self, health: JobWatcherHealth, event: ScrapeEvent) -> None:
        health.recent_events.append(event)
        if len(health.recent_events) > self.MAX_RECENT_EVENTS:
            health.recent_events.pop(0)

    def get_job_health(self, channel_id: int) -> JobWatcherHealth | None:
        return self._job.get(channel_id)

    def all_job_health(self) -> dict[int, JobWatcherHealth]:
        return dict(self._job)

    # ── ATS ──────────────────────────────────────────────────────────────────

    def set_ats_task_alive(self, alive: bool) -> None:
        self._ats.task_alive = alive

    def record_ats_platform_result(
        self,
        platform: str,
        job_count: int,
        new_count: int = 0,
        error: str | None = None,
        submitted: int = 0,
        completed: int = 0,
    ) -> None:
        ph = self._ats.per_platform.setdefault(platform, ATSPlatformHealth(platform=platform))
        ph.last_scrape_at = time.time()
        # Recorded even on an error: a cycle that raised after reaching 9,000 of
        # 10,000 companies is a different problem from one that reached 12, and
        # the counts are the only thing that tells them apart.
        if submitted > 0:
            ph.last_submitted = submitted
            ph.last_completed = completed
            ph.total_submitted += submitted
            ph.total_completed += completed
        if error:
            ph.total_errors += 1
            ph.last_error = error[:120]
            ph.last_was_error = True
        else:
            ph.last_job_count = job_count
            ph.last_new_count = new_count
            ph.total_jobs += job_count
            ph.total_new_jobs += new_count
            ph.last_was_error = False
            # Deliberately not an error: a refused platform has not crashed,
            # and marking it one would make every genuinely quiet platform look
            # broken. It is tracked as its own state instead.
            if job_count > 0:
                ph.consecutive_silent = 0
                ph.last_nonempty_at = ph.last_scrape_at
            else:
                ph.consecutive_silent += 1

    def set_ats_refusing(self, refusing) -> None:
        """Record which platforms stopped being asked, and after how many
        refusals. Empty clears it, so a recovered platform stops being flagged.
        """
        self._ats.refusing = {str(k): int(v) for k, v in dict(refusing or {}).items()}

    def set_ats_roster(self, platforms, disabled=()) -> None:
        """Declare every ATS platform that exists, and which are switched off.

        Registered by the scrape loop rather than imported here, so this module
        keeps knowing nothing about `ats_service` -- the roster is data the
        caller already has, and importing the scraper to render an embed would
        tie the health surface to the scraping stack.
        """
        # Not deduped here: `_ats_platform_lines` has to dedupe anyway, since
        # it unions the roster with whatever has reported. Doing it twice would
        # leave one of the two impossible to exercise.
        self._ats.roster = tuple(str(p) for p in platforms)
        self._ats.disabled = frozenset(str(p) for p in disabled)

    def ats_fleet_coverage(self) -> list[dict[str, object]]:
        """What fraction of each platform's fleet the last cycle actually reached.

        The question this answers could not previously be asked from inside the
        bot: the fan-out's submitted/completed counts existed only in a print.
        Reasoning about coverage from the printed *cancelled* count instead is
        how "43-100% reached" got misread as "2% reached" -- the log reports
        what was cancelled, and the interesting number is the complement.

        Platforms with no recorded cycle are omitted rather than reported as 0%,
        which would be indistinguishable from a platform that reached nothing.
        """
        out: list[dict[str, object]] = []
        for name, ph in self._ats.per_platform.items():
            if ph.last_submitted <= 0:
                continue
            out.append({
                "platform": name,
                "submitted": ph.last_submitted,
                "completed": ph.last_completed,
                "reached_pct": round(100.0 * ph.last_completed / ph.last_submitted, 1),
                "lifetime_reached_pct": (
                    round(100.0 * ph.total_completed / ph.total_submitted, 1)
                    if ph.total_submitted else None
                ),
            })
        return sorted(out, key=lambda row: row["reached_pct"])

    def silent_ats_platforms(self, threshold: int = 3) -> list[tuple[str, int]]:
        """Platforms that have returned nothing for `threshold` runs running.

        The signal this exists to surface is the one the archive cannot give:
        a platform validated weekly against thousands of live companies that
        still contributes no jobs. One empty run is ordinary; several in a row
        against real coverage is a scraper or an endpoint that has stopped
        answering, and nothing else in the pipeline says so.

        Sorted longest-silent first so the worst offender reads first.
        """
        silent = [
            (name, ph.consecutive_silent)
            for name, ph in self._ats.per_platform.items()
            if ph.consecutive_silent >= threshold
        ]
        return sorted(silent, key=lambda pair: pair[1], reverse=True)

    def record_ats_scrape_complete(
        self, total_jobs: int, scrapes_today: int, daily_cap: int
    ) -> None:
        self._ats.last_scrape_at = time.time()
        self._ats.last_total_jobs = total_jobs
        self._ats.scrapes_today = scrapes_today
        self._ats.daily_cap = daily_cap
        self._ats.total_jobs_logged += total_jobs

    def get_ats_health(self) -> ATSScrapeHealth:
        return self._ats

    # ── Browser/Playwright ───────────────────────────────────────────────────

    def record_browser_event(self, event: str, value: object = None) -> None:
        """Telemetry sink registered with browser_service.set_health_hook().
        Runs on the browser fetch path's calling thread -- must stay fast and
        must never raise."""
        b = self._browser
        b.last_event_at = time.time()
        if event == "dispatch_saturated":
            b.dispatch_saturated_count += 1
        elif event == "dispatch_timeout":
            b.dispatch_timeout_count += 1
        elif event == "dispatch_yield":
            b.dispatch_yield_count += 1
        elif event == "session_probe":
            b.last_probe_success = bool(value)
            if value:
                b.session_probe_success_count += 1
            else:
                b.session_probe_failure_count += 1
        elif event == "resync_attempt":
            b.resync_attempt_count += 1
            if value:
                b.resync_success_count += 1
        elif event == "active_profile":
            b.active_profile_path = str(value or "")

    def get_browser_health(self) -> BrowserServiceHealth:
        return self._browser

    # ── Watchdog ─────────────────────────────────────────────────────────────

    def record_watchdog_restart(self, target: str) -> None:
        """Called when the watchdog detects a dead-but-enabled watcher/loop and restarts it."""
        w = self._watchdog
        w.total_restarts += 1
        w.last_restart_at = time.time()
        w.last_restart_target = target
        w.recent_restarts.append(WatchdogRestart(w.last_restart_at, target))
        if len(w.recent_restarts) > self.MAX_RECENT_RESTARTS:
            w.recent_restarts.pop(0)

    def get_watchdog_health(self) -> WatchdogHealth:
        return self._watchdog

    # ── Bot-level ────────────────────────────────────────────────────────────

    def bot_uptime_s(self) -> float:
        return time.time() - self._started_at


# ── Discord embed builders ────────────────────────────────────────────────────


def _format_queue_field(scheduler_stats: dict[str, Any]) -> str:
    queued = scheduler_stats.get("queued", 0)
    queued_interactive = scheduler_stats.get("queued_interactive", 0)
    queued_background = scheduler_stats.get("queued_background", 0)
    active = scheduler_stats.get("active", 0)
    workers = scheduler_stats.get("workers", 0)
    completed = scheduler_stats.get("completed", 0)
    promoted = scheduler_stats.get("promoted", 0)
    icon = "🟢" if queued == 0 else "🟡" if queued < workers else "🔴"
    lines = [
        f"{icon} Queued: **{queued}** ({queued_interactive} interactive / {queued_background} background)",
        f"Active: **{active}**/{workers} workers · Completed: **{completed}** · Aged-up: **{promoted}**",
        # Detected hardware and the resulting pool scale: the fastest way to
        # confirm a new host (especially a container, where cpu_count lies)
        # sized its pools the way you expected.
        f"Hardware: {capacity.describe()}",
    ]
    return "\n".join(lines)


# Discord refuses an embed field whose value exceeds this, with a 400 that
# takes the whole `.health` reply down rather than the one field. Every field
# built from a per-platform list is therefore fitted rather than assumed to fit.
_FIELD_VALUE_LIMIT = 1024

# Runs of nothing before a platform is called out. Matches the default on
# `silent_ats_platforms`; one empty run is ordinary, three in a row is not.
SILENT_RUN_THRESHOLD = 3


def _join_within_limit(lines: list[str], limit: int = _FIELD_VALUE_LIMIT) -> str:
    """As many whole lines as fit, and an honest count of what did not.

    A fixed `lines[:12]` slice is wrong in both directions: it drops four of
    sixteen platforms without saying so, and it still overflows once the lines
    themselves grow. Fitting by length keeps every line that can be shown and
    names the remainder, so nothing disappears silently.
    """
    if not lines:
        return ""
    kept: list[str] = []
    used = 0
    for i, line in enumerate(lines):
        remaining = len(lines) - i
        # Reserve room for the "+N more" line, but only while one is still
        # needed -- the last line must not be evicted to make room for a
        # summary of nothing.
        tail = f"\n… +{remaining} more" if remaining > 1 else ""
        cost = len(line) + (1 if kept else 0)
        if used + cost + len(tail) > limit:
            break
        kept.append(line)
        used += cost
    dropped = len(lines) - len(kept)
    if not kept:
        # A single line longer than the whole budget: say so rather than
        # returning "" and letting the field vanish.
        return f"… {len(lines)} platform(s), too long to display"
    if dropped:
        kept.append(f"… +{dropped} more")
    return "\n".join(kept)


def _ats_platform_lines(ats: ATSScrapeHealth) -> list[str]:
    """One line per ATS platform that exists -- not per platform that reported.

    `per_platform` is written only when a scrape reports a result, so it is a
    log, not an inventory. Rendering straight from it meant the Platforms field
    was missing entirely until the first cycle finished, listed one of sixteen
    after one platform reported, and could never show a platform disabled in
    config at all. bamboohr is disabled by default and carries the largest
    fleet of the sixteen, so the one platform most worth knowing about was the
    one guaranteed to be invisible.

    Every roster entry appears, with an explicit state for the ones that have
    not reported, because "nothing here" and "not asked yet" are the two things
    this whole subsystem exists to tell apart.
    """
    names = list(dict.fromkeys(list(ats.roster) + sorted(ats.per_platform)))
    lines: list[str] = []
    for platform in names:
        if platform in ats.disabled:
            lines.append(f"⚪ **{platform}**: disabled in config")
            continue
        refused = ats.refusing.get(platform)
        ph = ats.per_platform.get(platform)
        if ph is None:
            note = (f": refused {refused}× and stopped" if refused
                    else ": no cycle yet")
            lines.append(f"⛔ **{platform}**{note}" if refused
                         else f"⚪ **{platform}**{note}")
            continue
        icon = "❌" if ph.last_was_error else ("✅" if ph.last_job_count else "🔇")
        errs = f" · {ph.total_errors} err" if ph.total_errors > 0 else ""
        silent = (f" · silent {ph.consecutive_silent}×"
                  if ph.consecutive_silent >= SILENT_RUN_THRESHOLD else "")
        if refused:
            # Distinct from silence: this platform was not quiet, it refused
            # every request and the cycle stopped asking. Reporting it as
            # merely silent is what let workable look ordinary for 45 runs.
            icon = "⛔"
            silent += f" · refused {refused}× and stopped"
        lines.append(
            f"{icon} **{platform}**: {ph.last_job_count:,} scraped / {ph.last_new_count:,} new · "
            f"{ph.total_new_jobs:,} new total{errs}{silent} · {_fmt_ago(ph.last_scrape_at)}"
        )
    return lines


def _format_ats_coverage_field(tracker: WatcherHealthTracker) -> str | None:
    """How much of each platform's fleet the last cycle reached, worst first.

    `ats_fleet_coverage` and `silent_ats_platforms` both answer a question the
    bot could not previously be asked from Discord: whether a platform is quiet
    because there was nothing to find, or because the cycle never got to most
    of its fleet. Returns None when neither has anything to say, so the caller
    can leave the field out rather than show an empty one.
    """
    coverage = tracker.ats_fleet_coverage()
    silent = dict(tracker.silent_ats_platforms(threshold=SILENT_RUN_THRESHOLD))
    if not coverage and not silent:
        return None

    lines: list[str] = []
    for row in coverage:
        name = str(row["platform"])
        pct = row["reached_pct"]
        icon = "🔴" if pct < 50 else "🟡" if pct < 90 else "🟢"
        note = f" · silent {silent[name]}×" if name in silent else ""
        lines.append(
            f"{icon} **{name}**: {pct}% reached "
            f"({int(row['completed']):,}/{int(row['submitted']):,}){note}"
        )
    # A platform can be silent without any recorded cycle -- it is refused
    # before submitting anything. Omitting it here would hide the loudest
    # symptom there is.
    seen = {str(row["platform"]) for row in coverage}
    for name, runs in silent.items():
        if name not in seen:
            lines.append(f"🔇 **{name}**: silent {runs}× · no cycle recorded")
    return _join_within_limit(lines)


def build_channel_health_embed(
    tracker: WatcherHealthTracker,
    channel_id: int,
    channel_name: str,
    job_task_alive: bool,
    reddit_task_alive: bool,
    active_job_watchers: int,
    active_reddit_watchers: int,
    scheduler_stats: dict[str, Any] | None = None,
) -> list[discord.Embed]:
    import discord as _discord

    job_health = tracker.get_job_health(channel_id)
    ats = tracker.get_ats_health()

    # Color reflects worst health signal on this channel
    critical = job_health and job_health.consecutive_errors >= 3
    degraded = job_health and job_health.consecutive_errors >= 1
    color = (
        _discord.Color.red() if critical
        else _discord.Color.yellow() if degraded
        else _discord.Color.green()
    )

    main = _discord.Embed(title=f"🩺 Health — #{channel_name}", color=color)

    # Bot row
    uptime = _fmt_uptime(tracker.bot_uptime_s())
    main.add_field(
        name="Bot",
        value=(
            f"Uptime: **{uptime}**\n"
            f"Job watchers: **{active_job_watchers}** · Reddit watchers: **{active_reddit_watchers}**"
        ),
        inline=False,
    )

    # Job watcher
    if job_health and job_health.started_at > 0:
        status_icon = "🟢" if job_task_alive else "🔴"
        status_label = "Running" if job_task_alive else "Stopped"
        lines = [f"{status_icon} **{status_label}** · Started {_fmt_ago(job_health.started_at)}"]

        if job_health.last_scrape_at > 0:
            lines.append(
                f"Last scrape: {_fmt_ago(job_health.last_scrape_at)} "
                f"({job_health.last_scrape_duration_s:.1f}s)"
            )
            lines.append(
                f"Last result: **{job_health.last_raw_count}** raw → "
                f"**{job_health.last_filtered_count}** filtered → "
                f"**{job_health.last_sent_count}** sent"
            )
        else:
            lines.append("Last scrape: first cycle pending")

        lines.append(
            f"Session: **{job_health.total_scrapes}** scrapes · "
            f"**{job_health.total_jobs_sent}** sent · "
            f"**{job_health.total_errors}** errors"
        )

        if job_health.consecutive_errors > 0:
            lines.append(f"⚠️ Consecutive errors: **{job_health.consecutive_errors}**")
            if job_health.last_error:
                lines.append(f"Last error: `{job_health.last_error[:80]}`")

        main.add_field(name="Job Watcher", value="\n".join(lines), inline=False)

        # Recent scrapes
        if job_health.recent_events:
            event_lines = []
            for ev in reversed(job_health.recent_events[-6:]):
                icon = "❌" if ev.error else "✅"
                ago = _fmt_ago(ev.timestamp)
                if ev.error:
                    event_lines.append(f"{icon} {ago} — `{ev.error[:60]}`")
                else:
                    event_lines.append(
                        f"{icon} {ago} — **{ev.sent_count}** sent "
                        f"({ev.raw_count} raw, {ev.duration_s:.1f}s)"
                    )
            main.add_field(
                name=f"Recent Scrapes (last {len(event_lines)})",
                value="\n".join(event_lines),
                inline=False,
            )
    elif job_task_alive:
        main.add_field(
            name="Job Watcher", value="🟢 Running · first scrape pending", inline=False
        )
    else:
        main.add_field(
            name="Job Watcher", value="⭕ No active job watcher on this channel", inline=False
        )

    # Reddit
    reddit_status = "🟢 Running" if reddit_task_alive else "⭕ Not active"
    main.add_field(name="Reddit Watcher", value=reddit_status, inline=True)

    if scheduler_stats is not None:
        main.add_field(name="Work Queue", value=_format_queue_field(scheduler_stats), inline=False)

    # ATS embed
    ats_color = _discord.Color.blurple() if ats.task_alive else _discord.Color.dark_gray()
    ats_embed = _discord.Embed(title="ATS Scrape Loop", color=ats_color)

    ats_icon = "🟢" if ats.task_alive else "🔴"
    ats_lines = [
        f"{ats_icon} **{'Running' if ats.task_alive else 'Not running'}**",
        f"Last run: {_fmt_ago(ats.last_scrape_at)}",
        f"Today: **{ats.scrapes_today}/{ats.daily_cap}** scrapes",
        f"Session total: **{ats.total_jobs_logged:,}** jobs logged",
    ]
    ats_embed.add_field(name="Status", value="\n".join(ats_lines), inline=False)

    plat_lines = _ats_platform_lines(ats)
    if plat_lines:
        ats_embed.add_field(
            name=f"Platforms ({len(plat_lines)})",
            value=_join_within_limit(plat_lines),
            inline=False,
        )

    coverage = _format_ats_coverage_field(tracker)
    if coverage:
        ats_embed.add_field(name="Fleet Coverage", value=coverage, inline=False)

    return [main, ats_embed]


def build_all_health_embed(
    tracker: WatcherHealthTracker,
    channel_job_tasks: dict[int, object],
    channel_names: dict[int, str],
    active_job_watchers: int,
    active_reddit_watchers: int,
    scheduler_stats: dict[str, Any] | None = None,
) -> discord.Embed:
    import discord as _discord

    uptime = _fmt_uptime(tracker.bot_uptime_s())
    embed = _discord.Embed(
        title="🩺 All Watcher Health",
        description=(
            f"Uptime: **{uptime}** · "
            f"Job: **{active_job_watchers}** · "
            f"Reddit: **{active_reddit_watchers}**"
        ),
        color=_discord.Color.blurple(),
    )

    all_health = tracker.all_job_health()
    if not all_health:
        embed.add_field(name="Job Watchers", value="No watchers tracked yet.", inline=False)

    lines = []
    for channel_id, health in sorted(all_health.items()):
        task = channel_job_tasks.get(channel_id)
        alive = bool(task and not getattr(task, "done", lambda: True)())

        if health.consecutive_errors >= 3:
            icon = "🔴"
        elif health.consecutive_errors >= 1 or not alive:
            icon = "🟡"
        else:
            icon = "🟢"

        name = channel_names.get(channel_id) or str(channel_id)
        sent = health.total_jobs_sent
        err = health.total_errors
        ago = _fmt_ago(health.last_scrape_at)
        err_str = f" · {err} err" if err > 0 else ""
        lines.append(f"{icon} **#{name}** — {sent} sent · last {ago}{err_str}")

    if lines:
        embed.add_field(
            name=f"Job Watchers ({len(lines)})",
            value="\n".join(lines),
            inline=False,
        )

    ats = tracker.get_ats_health()
    ats_icon = "🟢" if ats.task_alive else "🔴"
    embed.add_field(
        name="ATS Loop",
        value=(
            f"{ats_icon} {'Running' if ats.task_alive else 'Stopped'} · "
            f"last {_fmt_ago(ats.last_scrape_at)} · "
            f"{ats.scrapes_today}/{ats.daily_cap} today · "
            f"{ats.total_jobs_logged:,} total"
        ),
        inline=False,
    )

    plat_lines = _ats_platform_lines(ats)
    if plat_lines:
        embed.add_field(
            name=f"ATS Platforms ({len(plat_lines)})",
            value=_join_within_limit(plat_lines),
            inline=False,
        )

    coverage = _format_ats_coverage_field(tracker)
    if coverage:
        embed.add_field(name="ATS Fleet Coverage", value=coverage, inline=False)

    browser = tracker.get_browser_health()
    # last_probe_success reflects current state; the old check (any success
    # ever) let one early success permanently mask a later, ongoing outage.
    if browser.last_probe_success is None:
        browser_icon = "⚪"  # no probe yet
    elif browser.last_probe_success:
        browser_icon = "🟢"
    else:
        browser_icon = "🔴"
    # Split on either separator rather than using Path().name.
    #
    # rsplit("\\") was wrong on Linux (POSIX paths have no backslash, so it
    # returned the whole path). Path().name fixed that but is equally
    # platform-dependent in the other direction: on Linux, PosixPath treats
    # "C:\\Users\\...\\chrome_profile_runtime" as a single component and returns
    # the entire string. The bot runs on both, and this value is only ever
    # displayed, so parse it independently of the host's flavour.
    _raw_profile = str(browser.active_profile_path or "").rstrip("/\\")
    profile_name = re.split(r"[\\/]", _raw_profile)[-1] if _raw_profile else ""
    profile_name = profile_name or "n/a"
    embed.add_field(
        name="Browser (Playwright)",
        value=(
            f"{browser_icon} profile: `{profile_name}` · "
            f"saturated: {browser.dispatch_saturated_count} · "
            f"timeouts: {browser.dispatch_timeout_count} · "
            f"yielded to priority: {browser.dispatch_yield_count} · "
            f"probes: {browser.session_probe_success_count}✓/{browser.session_probe_failure_count}✗ · "
            f"resyncs: {browser.resync_success_count}/{browser.resync_attempt_count}"
        ),
        inline=False,
    )

    watchdog = tracker.get_watchdog_health()
    watchdog_icon = "🟢" if watchdog.total_restarts == 0 else "🟡"
    watchdog_lines = [
        f"{watchdog_icon} Auto-restarts: **{watchdog.total_restarts}** · last {_fmt_ago(watchdog.last_restart_at)}"
    ]
    if watchdog.recent_restarts:
        for entry in reversed(watchdog.recent_restarts[-5:]):
            watchdog_lines.append(f"  • {_fmt_ago(entry.timestamp)} — {entry.target}")
    embed.add_field(name="Watchdog", value="\n".join(watchdog_lines), inline=False)

    if scheduler_stats is not None:
        embed.add_field(name="Work Queue", value=_format_queue_field(scheduler_stats), inline=False)

    return embed
