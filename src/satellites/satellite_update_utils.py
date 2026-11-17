"""
satellite_update_utils.py
=========================
Everything that CHANGES about a satellite while operations run: battery charge, data on board (the downlink queue), fuel,
agility state, degraded capacities, outages and telemetry corrections.

`SatState` is a snapshot of the satellite at time `t`. The planner always plans FROM a snapshot, never from t=0, which is what
lets the fleet be re-tasked continuously. Two kinds of updates exist:

1. PREDICTED evolution  -> `advance_state` executes the observations that start in [state.t, t_to) and rolls the state forward
   (battery, queue / storage, deliveries, fuel burn).
2. EXTERNAL changes     -> `apply_outage`, `apply_maintenance`, `apply_capacity_change`, `apply_telemetry` (things that happen
   to the satellite and that the planner learns about while operating).
"""
from __future__ import annotations
import copy, math
from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SatState:
    """The live snapshot: current resources, pointing, pending data, and known interruptions."""
    t: float                                   # snapshot time (s)
    soc_wh: float                              # battery energy
    queue: list                                # FIFO of onboard data: [task_id | None, remaining_gb, deadline_s, value]
    roll_deg: float                            # current signed roll angle
    last_end_t: float                          # when the last observation ended (agility / slew continuity)
    fuel_kg: float                             # remaining propellant
    storage_cap_gb: float                      # CURRENT usable storage (can degrade)
    battery_cap_wh: float                      # CURRENT usable battery capacity (can degrade)
    solar_factor: float = 1.0                  # 1.0 = nominal solar array output
    unavailable: list = field(default_factory=list)   # [(t0, t1)] windows with no imaging (outage, maintenance)
    maintenance: list = field(default_factory=list)   # [(t0, t1, burn_kg)] pending orbit-maintenance burns

    @property
    def stored_gb(self) -> float: return float(sum(q[1] for q in self.queue))


def initial_state(sat) -> SatState:
    """State at t=0 straight from the satellite specification (the backlog already on board is one anonymous queue entry)."""
    # Start with the design capacities, initial battery and fuel, and data already onboard.
    return SatState(t=0.0, soc_wh=sat.soc_init_wh, queue=[[None, sat.backlog_gb, math.inf, 0.0]], roll_deg=sat.current_angle,
                    last_end_t=0.0, fuel_kg=sat.propellant_kg, storage_cap_gb=sat.storage_gb, battery_cap_wh=sat.battery_wh)


# ------------------------------------------------------------------------------------- external changes
def apply_outage(state: SatState, t0: float, t1: float) -> None:
    """Safe mode / anomaly: no imaging between t0 and t1 (downlink contacts are unaffected in this model)."""
    state.unavailable.append((t0, t1))


def apply_maintenance(state: SatState, t0: float, t1: float, burn_kg: float) -> None:
    """Planned orbit-maintenance burn: blocks imaging in [t0, t1] and consumes propellant once it is over."""
    state.unavailable.append((t0, t1)); state.maintenance.append((t0, t1, burn_kg))


def apply_capacity_change(state: SatState, storage_cap_gb=None, battery_cap_wh=None, solar_factor=None) -> None:
    """Degradation / repartitioning: new usable storage, battery capacity or solar-array output factor."""
    if storage_cap_gb is not None: state.storage_cap_gb = storage_cap_gb
    if battery_cap_wh is not None: state.battery_cap_wh = battery_cap_wh; state.soc_wh = min(state.soc_wh, battery_cap_wh)
    if solar_factor is not None: state.solar_factor = solar_factor


def apply_telemetry(state: SatState, soc_wh=None, stored_gb=None) -> None:
    """Replace the MODELLED battery level / stored data by what the satellite actually reported."""
    if soc_wh is not None: state.soc_wh = min(soc_wh, state.battery_cap_wh)
    if stored_gb is not None and state.stored_gb > 0:
        # Scale every queued item equally so their combined size matches the satellite report.
        f = stored_gb / state.stored_gb
        for q in state.queue: q[1] *= f                                     # scale every queued product proportionally


def fuel_status(sat, state: SatState) -> str:
    """'ok' | 'low' (below reserve) | 'empty'. Operators would schedule a maintenance burn / end-of-life manoeuvre."""
    if state.fuel_kg <= 0: return "empty"
    return "low" if state.fuel_kg < sat.fuel_reserve_kg else "ok"


# ------------------------------------------------------------------------------------- predicted evolution
@dataclass
class ExecObs:
    """One observation to execute: identifiers, timing and its resource footprint."""
    gid: int; task: str; t: float; t_end: float; k: int; ka: int
    roll: float; e: float; d: float; deadline: float; value: float
    ok: bool = True                  # False = acquisition fails (energy and time are spent, no data is produced)


def advance_state(R, sat, state: SatState, obs: list, t_to: float):
    """Roll the satellite forward from state.t to t_to while EXECUTING `obs` (those starting in the interval).

    Reality checks: an observation is skipped if the battery would go below its floor or the data does not fit in storage
    (a correct plan never triggers this; a stale plan after a disruption can).
    Returns (new_state, info) with info = dict(soc, q, deliveries, transfers, executed, skipped).
    Approximation: data of an observation still in progress at t_to joins the queue immediately.
    """
    # Update a copy so the caller's original snapshot remains available.
    dt = R.dt
    k0, k1 = int(round(state.t / dt)), int(round(t_to / dt))
    st = copy.deepcopy(state)
    f = st.solar_factor
    base = R.base0[k0:k1] + (f - 1.0) * R.gen_wh[k0:k1]
    cap = R.cap[k0:k1]
    floor = min(sat.soc_min_wh, st.soc_wh)
    x, q = min(st.soc_wh, st.battery_cap_wh), st.stored_gb
    fifo = deque(st.queue)
    starts = {o.k - k0: o for o in sorted(obs, key=lambda o: o.t)}
    arrive = {}
    soc_h, q_h = np.empty(k1 - k0), np.empty(k1 - k0)
    deliveries, transfers, executed, skipped = [], {}, [], []
    for i in range(k1 - k0):
        # Each step starts an image, adds data from completed images, and then drains the downlink queue.
        o = starts.get(i); e_use = 0.0
        if o is not None:
            if min(st.battery_cap_wh, x + base[i] - o.e) < floor - 1e-9:
                skipped.append((o, "battery")); o = None
            else:
                e_use = o.e
                if o.ok:
                    arrive.setdefault(o.ka - k0, []).append(o)
                executed.append(o)
                st.roll_deg, st.last_end_t = o.roll, o.t_end
        x = min(st.battery_cap_wh, x + base[i] - e_use)                       # same ordering as SatResources.curves
        for oo in arrive.pop(i, []):                                          # data of a finished acquisition
            if q + oo.d > max(st.storage_cap_gb, q) + 1e-9:
                skipped.append((oo, "storage")); continue
            fifo.append([oo.task, oo.d, oo.deadline, oo.value]); q += oo.d
        drain = cap[i]
        ci = R.step_contact[k0 + i]
        while drain > 1e-12 and fifo:
            take = min(drain, fifo[0][1]); fifo[0][1] -= take; drain -= take; q -= take
            if ci >= 0:
                tr = transfers.setdefault(R.contacts[ci], {}); tr[fifo[0][0]] = tr.get(fifo[0][0], 0.0) + take
            if fifo[0][1] <= 1e-9:
                tk, _, dl, val = fifo.popleft()
                deliveries.append((tk, k0 + i, dl, val))
        soc_h[i], q_h[i] = x, q
    for left in arrive.values():                                              # acquisitions still running at t_to
        for oo in left: fifo.append([oo.task, oo.d, oo.deadline, oo.value]); q += oo.d
    st.t, st.soc_wh, st.queue = t_to, x, [list(r) for r in fifo]
    # fuel: continuous drag make-up + completed maintenance burns
    # Drag uses fuel throughout the interval; maintenance fuel is charged once when its burn ends.
    st.fuel_kg = max(0.0, st.fuel_kg - sat.drag_kg_per_day * (t_to - state.t) / 86400.0)
    for t0, t1, burn in list(st.maintenance):
        if state.t < t1 <= t_to: st.fuel_kg = max(0.0, st.fuel_kg - burn); st.maintenance.remove((t0, t1, burn))
    return st, dict(soc=soc_h, q=q_h, deliveries=deliveries, transfers=transfers, executed=executed, skipped=skipped)
