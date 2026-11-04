"""
resources.py
============
Per-satellite resource model: battery (solar + eclipse + loads), onboard storage and the downlink queue.

Everything is evaluated on a fixed time grid (dt seconds) with closed-form VECTORISED recursions:

* battery   x_k = min(Cap, x_{k-1} + inc_k)              -> one-sided reflection  x = y - max(0, cummax(y) - Cap)
* storage   q_k = max(0, q_{k-1} + arrivals_k - drain_k) -> Lindley recursion      q = C - min(0, cummin(C))
* delivery  FIFO queue: a product is delivered at the first step where cumulative departures >= its cumulative arrivals.

A product can be split across several contacts (and several ground stations); storage is released continuously as bytes leave.

ANCHORING: the model can start from ANY carried-over state (`set_anchor(SatState)`): current battery charge, the products still
waiting in the downlink queue, degraded capacities, a solar-array factor ... This is what makes rolling-horizon re-planning possible.
`slow_simulate` is an independent step-by-step implementation used to verify the fast path.
"""
from __future__ import annotations
import math
from collections import deque
import numpy as np
from ..earth_sun_orbits.geo import eclipse_mask


class SatResources:
    def __init__(self, sat, n_steps: int, dt: float, guard_s: float = 15.0):
        self.sat, self.n, self.dt = sat, n_steps, dt
        ts = np.arange(n_steps) * dt
        self.eclipse = eclipse_mask(sat.eci(ts)[0])                       # in Earth's shadow?
        self.gen_wh = np.where(self.eclipse, 0.0, sat.solar_w) * dt / 3600.0   # solar energy per step (Wh)
        self.guard_k = int(math.ceil(guard_s / dt))
        self.unavail = np.zeros(n_steps, bool)                            # outage / maintenance windows (no imaging)
        self.set_contacts([])
        self.k0 = 0

    # ------------------------------------------------------------------ configuration (contacts, outages, anchor)
    def set_contacts(self, contacts):
        """(Re)build everything that depends on the downlink contact plan."""
        n, dt, sat = self.n, self.dt, self.sat
        self.contacts = list(contacts)
        self.in_contact = np.zeros(n, bool)
        self.step_contact = np.full(n, -1, dtype=int)                     # index of the contact active at each step
        self.cap = np.zeros(n)                                            # GB that can leave per step
        self.blocked_contact = np.zeros(n, bool)                          # pointing at the antenna: no imaging
        for ci, c in enumerate(self.contacts):
            a, b = int(math.ceil(c.t0 / dt)), min(int(math.floor(c.t1 / dt)), n)
            self.in_contact[a:b] = True; self.step_contact[a:b] = ci; self.cap[a:b] = c.rate * dt
            self.blocked_contact[max(0, a - self.guard_k):b + self.guard_k] = True
        # Wh per step before any imaging: solar - bus - (downlink radio while a contact is active)
        self.base0 = self.gen_wh - (sat.bus_w + sat.dl_w * self.in_contact) * dt / 3600.0
        self.cumcap = np.cumsum(self.cap)
        self._rebuild_blocked()

    def set_unavailable(self, windows):
        """Imaging is impossible inside these (t0, t1) windows (satellite outage, orbit-maintenance burn ...)."""
        self.unavail[:] = False
        for t0, t1 in windows:
            self.unavail[max(0, int(math.floor(t0 / self.dt))):min(self.n, int(math.ceil(t1 / self.dt)))] = True
        self._rebuild_blocked()

    def _rebuild_blocked(self):
        self.blocked_cum = np.concatenate([[0], np.cumsum(self.blocked_contact | self.unavail)])

    def set_anchor(self, state):
        """Start every evaluation from `state` (time, battery, queue, capacities, solar factor)."""
        sat = self.sat
        self.state = state
        k0 = self.k0 = int(round(state.t / self.dt))
        self.base_s = self.base0[k0:] + (state.solar_factor - 1.0) * self.gen_wh[k0:]
        self.cap_s = self.cap[k0:]
        self.batt_cap = state.battery_cap_wh
        self.soc0 = min(state.soc_wh, self.batt_cap)
        self.floor = min(sat.soc_min_wh, self.soc0)                       # an already-violated floor must not get worse
        self.queue_gb = np.array([q[1] for q in state.queue], dtype=float)
        self.q_total = float(self.queue_gb.sum())
        self.storage_cap = max(state.storage_cap_gb, self.q_total)
        prev = self.cumcap[k0 - 1] if k0 > 0 else 0.0
        self.dl_capacity_gb = float(self.cumcap[-1] - prev)               # downlink capacity still to come
        self.energy_budget_wh = max(0.0, self.soc0 - self.floor + float(self.base_s.sum()))   # optimistic (ignores clipping)

    # ------------------------------------------------------------------ helpers
    def is_blocked(self, k0, k1):
        """Does [k0, k1) touch a contact (+guard) or an unavailability window? Vectorised over arrays of steps."""
        return (self.blocked_cum[np.minimum(k1, self.n)] - self.blocked_cum[np.maximum(k0, 0)]) > 0

    def standalone_delivery_step(self, ka, d):
        """Earliest delivery step of a product alone in an empty queue (optimistic) - used for pruning impossible SLAs."""
        return np.searchsorted(self.cumcap, self.cumcap[ka - 1] + d - 1e-9)

    # ------------------------------------------------------------------ fast evaluation (relative to the anchor)
    def curves(self, k, ka, e, d):
        k, ka = np.asarray(k) - self.k0, np.asarray(ka) - self.k0
        inc = self.base_s.copy(); inc[k] -= e                                   # imaging energy at the start step
        y = self.soc0 + np.cumsum(inc)
        soc = y - np.maximum(0.0, np.maximum.accumulate(y) - self.batt_cap)    # charging clips at battery capacity
        arr = np.zeros(len(self.base_s)); arr[0] += self.q_total; arr[ka] += d  # queue content + new products
        C = np.cumsum(arr - self.cap_s)
        q = C - np.minimum(0.0, np.minimum.accumulate(C))
        return soc, q, arr

    def evaluate(self, k, ka, e, d):
        """-> (feasible, ABSOLUTE delivery step of each new product). Infeasible if battery or storage limit is violated."""
        soc, q, arr = self.curves(k, ka, e, d)
        if soc.min() < self.floor - 1e-9 or q.max() > self.storage_cap + 1e-9:
            return False, None
        A = np.cumsum(arr)
        D = np.maximum.accumulate(A - q)                                        # cumulative departures
        return True, np.searchsorted(D, A[np.asarray(ka) - self.k0] - 1e-7) + self.k0

    # ------------------------------------------------------------------ independent slow checks
    def slow_simulate(self, k, ka, e, d):
        """Step-by-step reference implementation (no shared code with `evaluate`)."""
        k0, n = self.k0, len(self.base_s)
        img, arrivals = {}, {}
        for j in range(len(k)):
            img[int(k[j]) - k0] = img.get(int(k[j]) - k0, 0.0) + e[j]
            arrivals.setdefault(int(ka[j]) - k0, []).append((j, d[j]))
        x, q = self.soc0, self.q_total
        fifo = deque([[-1, g] for g in self.queue_gb])
        done, ok = {}, True
        for s in range(n):
            x = min(self.batt_cap, x + self.base_s[s] - img.get(s, 0.0))
            if x < self.floor - 1e-9: ok = False
            for j, gb in arrivals.get(s, []): fifo.append([j, gb]); q += gb
            drain = self.cap_s[s]
            while drain > 1e-12 and fifo:
                take = min(drain, fifo[0][1]); fifo[0][1] -= take; drain -= take; q -= take
                if fifo[0][1] <= 1e-9:
                    j = fifo.popleft()[0]
                    if j >= 0: done[j] = s + k0
            if q > self.storage_cap + 1e-9: ok = False
        return ok, np.array([done.get(j, self.n) for j in range(len(k))], dtype=int)

    def transfer_log(self, k, ka, d):
        """Which bytes leave on which contact (FIFO): {contact_index: {product_position: GB}}; position -1 = pre-existing queue."""
        k0 = self.k0
        arrivals = {}
        for j in range(len(k)): arrivals.setdefault(int(ka[j]) - k0, []).append((j, d[j]))
        fifo = deque([[-1, g] for g in self.queue_gb]); log = {}
        for s in range(len(self.base_s)):
            for j, gb in arrivals.get(s, []): fifo.append([j, gb])
            ci = self.step_contact[s + k0]
            if ci < 0: continue
            drain = self.cap_s[s]
            while drain > 1e-12 and fifo:
                take = min(drain, fifo[0][1]); fifo[0][1] -= take; drain -= take
                log.setdefault(ci, {}); log[ci][fifo[0][0]] = log[ci].get(fifo[0][0], 0.0) + take
                if fifo[0][1] <= 1e-9: fifo.popleft()
        return log
