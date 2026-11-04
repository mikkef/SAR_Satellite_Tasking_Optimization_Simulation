"""
ground_station_utils.py
=======================
Operations on the ground segment: geometric passes, the downlink CONTACT PLAN (who talks to which antenna when), and updates to
that plan when a station goes down.

Key modelling choices
* An antenna serves one satellite at a time and a satellite talks to one antenna at a time.
* The network is shared with other customers, so only a fraction (`availability`) of geometric passes can actually be booked.
  Availability is a deterministic function of (seed, satellite, station, minute) so re-planning later reproduces the same
  "this pass was never offered to us" decisions.
* Longer / higher-capacity passes are booked first (greedy).
"""
from __future__ import annotations
import random, zlib
from dataclasses import dataclass
import numpy as np


@dataclass(eq=False)
class Contact:
    """A booked downlink window between one satellite and one antenna."""
    sat: object
    gs: object
    antenna: int
    t0: float
    t1: float
    rate: float                           # GB/s usable on this link

    @property
    def capacity_gb(self) -> float: return self.rate * (self.t1 - self.t0)


def compute_passes(sats, stations, t_end: float, dt: float = 10.0):
    """All geometric passes (sat, station, t_start, t_end) where the satellite is above the station's elevation mask."""
    ts = np.arange(0.0, t_end + dt, dt)
    gx = np.stack([g.xyz for g in stations]); up = gx / np.linalg.norm(gx, axis=1, keepdims=True)
    sin_mask = np.sin(np.radians([g.min_elev for g in stations]))
    out = []
    for sat in sats:
        r = sat.ecef(ts)[0]
        los = r[:, None, :] - gx[None]                                        # station -> satellite vectors
        sin_el = (los * up[None]).sum(2) / np.linalg.norm(los, axis=2)        # sine of elevation angle
        vis = sin_el > sin_mask[None]
        for gi, g in enumerate(stations):
            d = np.diff(np.concatenate([[0], vis[:, gi].astype(np.int8), [0]]))
            for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                out.append((sat, g, ts[a], ts[min(b, len(ts) - 1)]))
    return out


def _offered(seed, sat, g, t0, availability) -> bool:
    """Deterministic 'was this pass offered to us by the shared network?' decision."""
    rng = random.Random(zlib.crc32(f"{seed}|{sat.name}|{g.name}|{int(t0 // 60)}".encode()))
    return rng.random() < availability


def plan_contacts(passes, setup_s=20.0, guard_s=10.0, min_len=60.0, availability=0.6, seed=1, existing=None, outages=None):
    """Book contacts greedily (largest capacity first).

    existing : {sat.name: [Contact]}   contacts that are already booked and must be kept (they occupy antennas)
    outages  : [(station_name, t0, t1)] passes at that station overlapping the window cannot be used
    Returns {sat.name: [Contact sorted by time]}.
    """
    outages = outages or []
    plan = {k: list(v) for k, v in (existing or {}).items()}
    sat_busy, ant_busy, booked = {}, {}, set()
    for lst in plan.values():
        for c in lst:
            sat_busy.setdefault(c.sat.name, []).append((c.t0, c.t1)); ant_busy.setdefault((c.gs.name, c.antenna), []).append((c.t0, c.t1))
            booked.add((c.sat.name, c.gs.name, round(c.t0 - setup_s)))
    free = lambda lst, a, b: all(b <= s or a >= e for s, e in lst)
    cand = []
    for sat, g, t0, t1 in passes:
        a, b = t0 + setup_s, t1 - guard_s                        # usable window after link setup / before loss of signal
        if b - a < min_len or (sat.name, g.name, round(t0)) in booked: continue
        if any(n == g.name and a < e and b > s for n, s, e in outages): continue
        if not _offered(seed, sat, g, t0, availability): continue
        cand.append((min(sat.dl_gb_s, g.rate_gb_s) * (b - a), sat, g, a, b))
    cand.sort(key=lambda x: -x[0])
    for cap, sat, g, a, b in cand:
        if not free(sat_busy.setdefault(sat.name, []), a, b): continue
        for k in range(g.antennas):
            lst = ant_busy.setdefault((g.name, k), [])
            if free(lst, a, b):
                lst.append((a, b)); sat_busy[sat.name].append((a, b))
                plan.setdefault(sat.name, []).append(Contact(sat, g, k, a, b, min(sat.dl_gb_s, g.rate_gb_s)))
                break
    for v in plan.values(): v.sort(key=lambda c: c.t0)
    return plan


def apply_station_outage(contacts, passes, station_name, t0, t1, now, **kw):
    """A station goes down in [t0, t1]: cancel its contacts in that window, then try to re-book freed satellites on other
    stations/antennas (only passes that have not started yet). Contacts already flown stay untouched."""
    kept = {k: [c for c in v if not (c.gs.name == station_name and c.t0 < t1 and c.t1 > max(t0, now))] for k, v in contacts.items()}
    future = [p for p in passes if p[3] > now]
    return plan_contacts(future, existing=kept, outages=[(station_name, t0, t1)], **kw)
