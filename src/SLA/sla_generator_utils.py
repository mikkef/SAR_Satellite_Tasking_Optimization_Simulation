"""
sla_generator_utils.py
======================
Generates the CONTRACTUAL side of the demand: when each order arrives, what SLA it carries and whether it is a RUSH task.

Why this matters: in real operations the planner never sees tomorrow's orders today. The fleet is already tasked when new
orders (with their own SLA clocks) keep arriving, so the plan must be continuously updated. This module produces such a stream.

Design of the stream
* BACKLOG   : a share of tasks is already known at t=0 (orders accepted yesterday and not yet fulfilled).
* DAILY WAVE: routine orders arrive mostly during business hours (UTC).
* BURSTS    : disaster orders arrive in clusters (a flood / earthquake triggers many orders within an hour or two).
* SLA       : measured from ORDER time to fully DELIVERED product; category dependent. RUSH orders get a short SLA and a higher reward.
"""
from __future__ import annotations
import math, random
from .event_creator import Event, make_events

HOUR = 3600.0

# Routine SLA choices (hours from order to delivery) per category
SLA_HOURS = {"maritime": [8, 12, 24], "infrastructure": [24, 36, 48], "disaster": [8, 12, 24], "environment": [36, 48, 72]}
RUSH_SLA_HOURS = [3, 4, 6]                                         # urgent orders
RUSH_SHARE = {"maritime": 0.12, "disaster": 0.30, "infrastructure": 0.05, "environment": 0.0}
RUSH_REWARD_MULTIPLIER = 1.8                                       # customers pay a premium for urgency
BURST_CENTERS_H = [5.5, 11.0, 16.5]                                # hours (UTC) when disaster bursts happen
BURST_SIGMA_H = 0.5
ARRIVAL_END_H = 20.0                                               # no orders after this (their SLA could not be tested in-horizon)


def sample_priority(category: str, rng: random.Random) -> str:
    """Rush or standard, with a category dependent probability."""
    return "rush" if rng.random() < RUSH_SHARE[category] else "standard"


def sample_sla_hours(category: str, priority: str, rng: random.Random) -> float:
    """SLA (hours, order -> delivered) for a task."""
    return float(rng.choice(RUSH_SLA_HOURS if priority == "rush" else SLA_HOURS[category]))


def _daily_wave_weight(hour: float) -> float:
    """Relative order intensity vs. UTC hour: ~1 at night, ~1.6 around midday."""
    return 1.0 + 0.6 * max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0))


def sample_arrival_time(category: str, rng: random.Random, backlog_share: float = 0.35) -> float:
    """Arrival time (seconds). Backlog -> 0; disaster -> mostly Gaussian bursts; others -> the daily wave."""
    if rng.random() < backlog_share:
        return 0.0                                                  # already in the queue when the day starts
    if category == "disaster" and rng.random() < 0.7:
        c = rng.choice(BURST_CENTERS_H)
        return min(max(rng.gauss(c, BURST_SIGMA_H), 0.1), ARRIVAL_END_H) * HOUR
    while True:                                                     # rejection sampling of the daily wave
        h = rng.uniform(0.1, ARRIVAL_END_H)
        if rng.random() * 1.6 < _daily_wave_weight(h):
            return h * HOUR


def assign_slas(events: list[Event], seed: int = 11, backlog_share: float = 0.35) -> list[Event]:
    """Attach arrival time, priority and SLA to every task (in place) and return the list sorted by arrival."""
    rng = random.Random(seed)
    for e in events:
        e.priority = sample_priority(e.category, rng)
        e.sla_h = sample_sla_hours(e.category, e.priority, rng)
        e.arrival_t = sample_arrival_time(e.category, rng, backlog_share)
        if e.priority == "rush":
            e.reward *= RUSH_REWARD_MULTIPLIER
    return events


def generate_task_stream(n: int = 3000, seed: int = 7, month: int = 11, backlog_share: float = 0.35) -> list[Event]:
    """Convenience: create the tasks and attach their order stream."""
    return assign_slas(make_events(n, seed, month), seed + 4, backlog_share)


def arrivals_between(events: list[Event], t0: float, t1: float) -> list[Event]:
    """Orders that reach the planner in (t0, t1]. A first call with t0 = -1 returns the day-start backlog."""
    return [e for e in events if t0 < e.arrival_t <= t1]


def stream_summary(events: list[Event]) -> dict:
    """Counts by priority / category and the arrival profile (used for the README and sanity checks)."""
    out = {"n": len(events), "backlog": sum(e.arrival_t == 0 for e in events), "rush": sum(e.priority == "rush" for e in events)}
    for cat in SLA_HOURS:
        sub = [e for e in events if e.category == cat]
        if sub: out[cat] = dict(n=len(sub), rush=sum(e.priority == "rush" for e in sub), mean_sla_h=sum(e.sla_h for e in sub) / len(sub))
    return out
