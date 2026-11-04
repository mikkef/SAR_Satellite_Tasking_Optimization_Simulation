"""Vectorised access computation and candidate observation generation (satellite x task x product x time)."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from ..satellites.satellite_creator import Mode, MOD_COST


@dataclass(eq=False)
class Candidate:
    sat: object
    ev: object
    t: float
    mode: Mode
    angle: float          # signed roll, deg (+ = right of track)
    p: float              # observation chance
    @property
    def t_end(self): return self.t + self.mode.dwell


def mode_quality(events, mode: Mode) -> np.ndarray:
    """Resolution x family match per task, shape (E,)."""
    out = np.zeros(len(events))
    for i, e in enumerate(events):
        if mode.mod in e.mod_required:
            out[i] = max(min(1.0, r / mode.res) ** 2 for r in e.res_required)
    return out


def observation_chance(ev, mode: Mode, abs_angle: float) -> float:
    """Scalar reference implementation (the vectorised path must agree)."""
    q = max(min(1.0, r / mode.res) ** 2 for r in ev.res_required) if mode.mod in ev.mod_required else 0.0
    g = max(math.exp(-0.5 * ((abs_angle - a) / ev.angle_tol) ** 2) for a in ev.angle_required)
    return q * g


def _topk(keys, vals, k):
    """Indices of the k largest vals within each key group."""
    order = np.lexsort((-vals, keys))
    ks = keys[order]
    flag = np.concatenate([[True], ks[1:] != ks[:-1]])
    idx = np.arange(len(ks))
    rank = idx - np.maximum.accumulate(np.where(flag, idx, 0))
    return order[rank < k]


def build_candidates(sats, events, t0, t1, dt=10.0, min_p=0.4, bin_steps=6, per_pass=2, pass_s=1800.0, lam=0.5, month=11):
    E = len(events)
    rew = np.array([e.reward * e.season_factor(month) for e in events]); rev = np.array([e.times_revisited for e in events])
    ev_xyz = np.stack([e.xyz for e in events]); up = ev_xyz / np.linalg.norm(ev_xyz, axis=1, keepdims=True)
    A = np.full((E, 2), -999.0); tol = np.array([e.angle_tol for e in events])
    for i, e in enumerate(events):
        A[i, :len(e.angle_required)] = e.angle_required[:2]
    ts = np.arange(t0, t1 + 1e-9, dt)
    out = []
    for sat in sats:
        r, v = sat.ecef(ts)
        rn = r / np.linalg.norm(r, axis=1, keepdims=True); vn = v / np.linalg.norm(v, axis=1, keepdims=True)
        cr = np.cross(rn, vn)
        Cm = np.stack([mode_quality(events, m) for m in sat.products], 1)       # (E, M)
        ev_ok = np.where(Cm.max(1) > 0)[0]
        cmax = math.cos(math.radians(sat.max_central_deg() + 0.6))
        TI, EI = [], []
        for c0 in range(0, len(ts), 2000):
            ti, ej = np.nonzero(rn[c0:c0 + 2000] @ up[ev_ok].T > cmax)
            TI.append(ti + c0); EI.append(ev_ok[ej])
        ti, ei = np.concatenate(TI), np.concatenate(EI)
        if len(ti) == 0: continue
        los = ev_xyz[ei] - r[ti]; d = np.linalg.norm(los, axis=1)
        off = np.degrees(np.arccos(np.clip(-(rn[ti] * los).sum(1) / d, -1, 1)))
        roll = np.sign((los * cr[ti]).sum(1)) * off
        ok = ((-(los * up[ei]).sum(1) / d) > 0.1) & (off >= sat.min_angle) & (off <= sat.max_angle)
        ti, ei, roll, off = ti[ok], ei[ok], roll[ok], off[ok]
        gau = np.maximum(np.exp(-0.5 * ((off - A[ei, 0]) / tol[ei]) ** 2), np.exp(-0.5 * ((off - A[ei, 1]) / tol[ei]) ** 2))
        for mi, mode in enumerate(sat.products):
            p = Cm[ei, mi] * gau
            m = (p >= min_p) & (rew[ei] * p - lam * rev[ei] * MOD_COST[mode.mod] > 0)      # only candidates that can pay for their cost
            if not m.any(): continue
            t_i, e_i, r_i, p_i = ti[m], ei[m], roll[m], p[m]
            tsec = ts[t_i]
            keep = _topk((e_i.astype(np.int64) * 100000 + (t_i // bin_steps)), p_i, 1)           # best sample per time bin
            t_i, e_i, r_i, p_i, tsec = t_i[keep], e_i[keep], r_i[keep], p_i[keep], tsec[keep]
            keep = _topk(e_i.astype(np.int64) * 1000 + (tsec // pass_s).astype(np.int64), p_i, per_pass)   # best bins per pass
            for t_, e_, r_, p_ in zip(tsec[keep], e_i[keep], r_i[keep], p_i[keep]):
                out.append(Candidate(sat, events[e_], float(t_), mode, float(r_), float(p_)))
    return out
