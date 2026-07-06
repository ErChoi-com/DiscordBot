from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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


@dataclass
class ATSScrapeHealth:
    last_scrape_at: float = 0.0
    last_total_jobs: int = 0
    scrapes_today: int = 0
    daily_cap: int = 4
    total_jobs_logged: int = 0
    per_platform: dict[str, ATSPlatformHealth] = field(default_factory=dict)
    task_alive: bool = False


class WatcherHealthTracker:
    MAX_RECENT_EVENTS = 8

    def __init__(self) -> None:
        self._job: dict[int, JobWatcherHealth] = {}
        self._ats: ATSScrapeHealth = ATSScrapeHealth()
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
        self, platform: str, job_count: int, new_count: int = 0, error: str | None = None
    ) -> None:
        ph = self._ats.per_platform.setdefault(platform, ATSPlatformHealth(platform=platform))
        ph.last_scrape_at = time.time()
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

    # ── Bot-level ────────────────────────────────────────────────────────────

    def bot_uptime_s(self) -> float:
        return time.time() - self._started_at


# ── Discord embed builders ────────────────────────────────────────────────────


def build_channel_health_embed(
    tracker: WatcherHealthTracker,
    channel_id: int,
    channel_name: str,
    job_task_alive: bool,
    reddit_task_alive: bool,
    active_job_watchers: int,
    active_reddit_watchers: int,
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

    if ats.per_platform:
        plat_lines = []
        for platform, ph in sorted(ats.per_platform.items()):
            icon = "❌" if ph.last_was_error else "✅"
            errs = f" · {ph.total_errors} err" if ph.total_errors > 0 else ""
            plat_lines.append(
                f"{icon} **{platform}**: {ph.last_job_count:,} scraped / {ph.last_new_count:,} new · "
                f"{ph.total_new_jobs:,} new total{errs} · {_fmt_ago(ph.last_scrape_at)}"
            )
        ats_embed.add_field(
            name="Platforms", value="\n".join(plat_lines[:12]), inline=False
        )

    return [main, ats_embed]


def build_all_health_embed(
    tracker: WatcherHealthTracker,
    channel_job_tasks: dict[int, object],
    channel_names: dict[int, str],
    active_job_watchers: int,
    active_reddit_watchers: int,
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
        return embed

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

    embed.add_field(
        name=f"Job Watchers ({len(lines)})",
        value="\n".join(lines) or "None",
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

    return embed
