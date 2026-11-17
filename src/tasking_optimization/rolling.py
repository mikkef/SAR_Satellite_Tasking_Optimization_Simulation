"""
rolling.py
==========
Rolling-horizon operations simulator: the fleet is ALREADY TASKED, new orders (with SLAs) keep arriving, and things go wrong.

Loop (every `tick` seconds)
  1. disruptions that became known are applied (satellite outage / maintenance / capacity loss / telemetry, ground-station outage)
  2. the policy updates the plan from the CURRENT satellite states (frozen observations are protected)
  3. observations starting before the next tick are EXECUTED against reality (`satellite_update_utils.advance_state`):
     random acquisition failures, battery / storage checks, downlink queue, deliveries
All policies are scored from what actually happened (deliveries before the deadline), never from the plan.

Policies
  static          plan once from the day-start backlog and never update (what happens if you ignore new orders)
  insert_only     every tick, insert new orders into free capacity; nothing is ever moved
  insert_preempt  like insert_only, but RUSH orders may displace lower-value observations
  rolling         hourly full re-optimisation (greedy fill + LNS) from the current state
  hybrid          15-min insertion with rush preemption + hourly full re-optimisation   <- recommended
  oracle          perfect foresight of future orders (reference only; still reacts to disruptions; no acquisition failures)
"""
from __future__ import annotations
import random, time
from dataclasses import dataclass, field
import numpy as np
from ..satellites.satellite_creator import make_fleet
from ..ground_stations.ground_station_creator import make_stations
from ..ground_stations.ground_station_utils import compute_passes, plan_contacts, apply_station_outage
from ..SLA.sla_generator_utils import generate_task_stream
from ..SLA.candidates import build_candidates
from ..satellites.resources import SatResources
from ..satellites.satellite_update_utils import (initial_state, advance_state, ExecObs, apply_outage, apply_maintenance,
                                     apply_capacity_change, apply_telemetry, fuel_status)
from .problem import CandidatePool, Problem
from .solvers import insert_candidates, preempt_candidates, solve_lns

H = 3600.0
POLICIES = ["static", "insert_only", "insert_preempt", "rolling", "hybrid", "oracle"]


# ----------------------------------------------------------------------------- configuration + disruptions
@dataclass
class SimConfig:
    dt: float = 10.0                       # time grid (s)
    t_task: float = 24 * H                 # tasking window (observations start inside it)
    t_end: float = 30 * H                  # simulation end (late downlinks still count)
    lam: float = 0.5                       # weight of cost in the objective
    plan_horizon_s: float = 8 * H          # how far ahead a re-plan looks
    freeze_s: float = 900.0                # observations starting within this time are frozen (already uplinked)
    exec_fail_prob: float = 0.03           # chance an acquisition fails (same draws for every policy)
    seed: int = 5
    tick_s: dict = field(default_factory=lambda: {"static": 900.0, "insert_only": 900.0, "insert_preempt": 900.0,
                                                  "rolling": 3600.0, "hybrid": 900.0, "oracle": 900.0})
    lns_iters: dict = field(default_factory=lambda: {"rolling": 150, "hybrid": 100, "oracle": 400})
    hybrid_full_every: int = 4             # hybrid: full re-optimisation every N ticks (= hourly)


@dataclass
class Disruption:
    """Something that changes while operating. `announce_t` = when the planner (and reality) change."""
    kind: str                              # sat_outage | sat_maintenance | sat_capacity | telemetry | gs_outage
    target: str                            # satellite or station name
    t_start: float
    t_end: float
    announce_t: float
    params: dict = field(default_factory=dict)
    label: str = ""


def default_disruptions() -> list[Disruption]:
    return [
        Disruption("sat_maintenance", "VX-A2", 12 * H, 12.5 * H, 0.0, {"burn_kg": 0.05}, "VX-A2 orbit-maintenance burn (known in advance)"),
        Disruption("sat_outage", "VX-C2", 5 * H, 11 * H, 5 * H, {}, "VX-C2 safe mode 05-11 h"),
        Disruption("gs_outage", "Svalbard", 8 * H, 14 * H, 8 * H, {}, "Svalbard station down 08-14 h"),
        Disruption("sat_capacity", "VX-B6", 10 * H, 10 * H, 10 * H, {"storage_cap_gb": 80.0}, "VX-B6 storage partition lost: 150 -> 80 GB"),
        Disruption("sat_capacity", "VX-A4", 14 * H, 14 * H, 14 * H, {"battery_cap_wh": 85.0, "solar_factor": 0.9}, "VX-A4 battery fade (110 -> 85 Wh) and solar array -10 %"),
        Disruption("telemetry", "VX-A4", 16 * H, 16 * H, 16 * H, {"soc_delta_wh": -10.0}, "VX-A4 telemetry: battery reads 10 Wh lower than modelled"),
    ]


# ----------------------------------------------------------------------------- world (static, shared by all policies)
@dataclass
class World:
    cfg: SimConfig; fleet: list; stations: list; events: list; pool: CandidatePool
    passes: list; contacts0: dict; disruptions: list
    @property
    def n_steps(self): return int(self.cfg.t_end / self.cfg.dt) + 1


def build_world(n_tasks=1500, seed=7, cfg: SimConfig | None = None, disruptions=None) -> World:
    """Create the shared scenario: orders, satellites, ground stations, and possible observations."""
    cfg = cfg or SimConfig()
    # Contacts are planned before tasking; all policies start with this same network capacity.
    fleet, stations, events = make_fleet(), make_stations(), generate_task_stream(n_tasks, seed)
    passes = compute_passes(fleet, stations, cfg.t_end)
    contacts0 = plan_contacts(passes)
    pool = CandidatePool(fleet, events, build_candidates(fleet, events, 0, cfg.t_task, lam=cfg.lam), cfg.dt)
    return World(cfg, fleet, stations, events, pool, passes, contacts0, default_disruptions() if disruptions is None else disruptions)


# ----------------------------------------------------------------------------- outcome
@dataclass
class Outcome:
    policy: str
    kpi: dict
    executed: list                 # [(gid, ok, delivered_t | None, on_time)]
    deliveries: list               # [(task_id, delivered_t, on_time, value)]
    timeline: dict                 # per tick: t, backlog (known, unserved), planned
    hist_soc: list; hist_q: list   # per satellite, length n_steps (actual trajectory)
    transfers: list                # per satellite: {Contact: {task: GB}}
    contacts: dict                 # final contact plan
    fuel: dict
    wall_s: float


def run_policy(world: World, policy: str, cfg: SimConfig | None = None, verbose=False) -> Outcome:
    """Operate one planning strategy through the day and return what actually happened."""
    cfg = cfg or world.cfg; t_wall = time.time()
    fleet, pool, events = world.fleet, world.pool, world.events
    S, E, n_steps, dt = len(fleet), len(events), world.n_steps, cfg.dt
    contacts = {k: list(v) for k, v in world.contacts0.items()}
    R = [SatResources(sat, n_steps, dt) for sat in fleet]
    for r, sat in zip(R, fleet): r.set_contacts(contacts.get(sat.name, []))
    states = [initial_state(sat) for sat in fleet]
    sidx = {sat.name: i for i, sat in enumerate(fleet)}
    disr = sorted(world.disruptions, key=lambda d: d.announce_t); applied = [False] * len(disr)
    done = np.zeros(E, bool); retry = np.zeros(E, bool); plan_gids = [[] for _ in range(S)]
    hist_soc = [np.full(n_steps, np.nan) for _ in range(S)]; hist_q = [np.full(n_steps, np.nan) for _ in range(S)]
    transfers = [dict() for _ in range(S)]
    executed, deliveries_raw, timeline = [], [], dict(t=[], backlog=[], planned=[])
    stats = dict(skipped=0, failed=0, churn_opt=0, churn_disruption=0, replans=0)
    tick, last_plan_t, tick_i = cfg.tick_s[policy], -1.0, 0

    def fails(g):                                           # identical draws for every policy (oracle is failure-free by definition)
        # Reusing the same random draw for a candidate keeps failure luck comparable
        # when two policies choose to observe the same order.
        return policy != "oracle" and random.Random(cfg.seed * 1_000_003 + int(g)).random() < cfg.exec_fail_prob

    def apply_disruption(d, t):
        # Update the live satellite or station state when an announced problem takes effect.
        nonlocal contacts
        if d.kind == "gs_outage":
            contacts = apply_station_outage(contacts, world.passes, d.target, d.t_start, d.t_end, now=t)
            for r, sat in zip(R, fleet): r.set_contacts(contacts.get(sat.name, []))
            return
        s = sidx[d.target]; st = states[s]
        if d.kind == "sat_outage": apply_outage(st, d.t_start, d.t_end)
        elif d.kind == "sat_maintenance": apply_maintenance(st, d.t_start, d.t_end, d.params["burn_kg"])
        elif d.kind == "sat_capacity": apply_capacity_change(st, **d.params)
        elif d.kind == "telemetry": apply_telemetry(st, soc_wh=st.soc_wh + d.params["soc_delta_wh"])
        R[s].set_unavailable(st.unavailable)

    t = 0.0
    while t < cfg.t_task - 1e-9:
        t_next = min(t + tick, cfg.t_task)
        # ---- 1. disruptions
        # Apply newly announced outages or equipment changes before making the next plan.
        changed = False
        for i, d in enumerate(disr):
            if not applied[i] and d.announce_t <= t + 1e-9: apply_disruption(d, t); applied[i] = True; changed = True
        # ---- 2. planning
        # Each strategy chooses when and how to plan. The problem view only exposes orders
        # known so far (except the oracle), and protects observations too close to change.
        need_plan = (policy != "static" and policy != "oracle") or (policy == "static" and t == 0) or (policy == "oracle" and (t == 0 or changed))
        if need_plan:
            known = np.ones(E, bool) if policy == "oracle" else pool.ev_arrival <= t
            horizon = cfg.t_task if policy in ("static", "oracle") else cfg.plan_horizon_s
            P = Problem(pool, R, states, t, known, done, cfg.lam, horizon)
            plan, dropped = P.make_plan(plan_gids, t + cfg.freeze_s)
            # Compare old and new plans to count observations removed during replanning.
            old = {g for l in plan_gids for g in l}
            fresh = ((pool.ev_arrival > last_plan_t) & (pool.ev_arrival <= t)) | retry          # orders since last planning + retries
            new_idx = np.where(fresh[P.ev] & ~done[P.ev])[0]; rush_idx = new_idx[P.rush[new_idx]]
            allc = np.arange(P.n); lns = lambda pl, it: solve_lns(P, init=pl, iters=it, seed=tick_i)[0]
            if policy == "static": insert_candidates(P, plan, allc)
            elif policy == "insert_only": insert_candidates(P, plan, new_idx)
            elif policy == "insert_preempt": insert_candidates(P, plan, new_idx); preempt_candidates(P, plan, rush_idx)
            elif policy == "rolling": insert_candidates(P, plan, allc); plan = lns(plan, cfg.lns_iters["rolling"])
            elif policy == "hybrid":
                insert_candidates(P, plan, new_idx); preempt_candidates(P, plan, rush_idx)
                if tick_i % cfg.hybrid_full_every == 0: insert_candidates(P, plan, allc); plan = lns(plan, cfg.lns_iters["hybrid"])
            elif policy == "oracle": insert_candidates(P, plan, allc); plan = lns(plan, cfg.lns_iters["oracle"])
            plan_gids = [[int(P.gid[j]) for j in l] for l in plan.lists]
            new_set = {g for l in plan_gids for g in l}
            stats["churn_disruption"] += len(dropped); stats["churn_opt"] += max(0, len(old - new_set) - len(dropped)); stats["replans"] += 1
            last_plan_t = t; retry[:] = False
        timeline["t"].append(t); timeline["backlog"].append(int(((pool.ev_arrival <= t) & ~done).sum()))
        timeline["planned"].append(sum(len(l) for l in plan_gids))
        # ---- 3. execution against reality
        # Run observations that begin in this time interval. A planned observation can still
        # be blocked by a real outage or fail; successful data must also be downlinked later.
        for s in range(S):
            gl = plan_gids[s]; obs = []
            for g in [g for g in gl if pool.t[g] < t_next]:
                if R[s].is_blocked(np.array([pool.k[g]]), np.array([pool.ka[g]]))[0]:    # stale plan: contact / outage now blocks it
                    stats["skipped"] += 1; retry[pool.ev[g]] = True; continue
                ev = pool.ev[g]
                obs.append(ExecObs(int(g), events[ev].id, float(pool.t[g]), float(pool.t_end[g]), int(pool.k[g]), int(pool.ka[g]), float(pool.angle[g]),
                                   float(pool.e[g]), float(pool.d[g]), float(pool.ev_deadline[ev]), float(pool.val[g]), ok=not fails(g)))
            k0 = int(round(states[s].t / dt)); k1 = int(round(t_next / dt))
            states[s], info = advance_state(R[s], fleet[s], states[s], obs, t_next)
            plan_gids[s] = [g for g in gl if pool.t[g] >= t_next]
            for o in info["executed"]:
                executed.append((o.gid, o.ok)); stats["failed"] += (not o.ok)
                if o.ok: done[pool.ev[o.gid]] = True
                else: retry[pool.ev[o.gid]] = True
            for o, why in info["skipped"]: stats["skipped"] += 1; retry[pool.ev[o.gid]] = True
            deliveries_raw += info["deliveries"]
            hist_soc[s][k0:k1], hist_q[s][k0:k1] = info["soc"], info["q"]
            for c, dct in info["transfers"].items():
                tr = transfers[s].setdefault(c, {})
                for tk, gb in dct.items(): tr[tk] = tr.get(tk, 0.0) + gb
        t = t_next; tick_i += 1
    # ---- flush: let the downlink queues drain until the end of the simulation
    # The tasking day can end while data is still on a satellite, so continue the simulation
    # to allow those already-collected products to reach the ground.
    t_fin = (n_steps - 1) * dt
    for s in range(S):
        k0 = int(round(states[s].t / dt)); k1 = int(round(t_fin / dt))
        states[s], info = advance_state(R[s], fleet[s], states[s], [], t_fin)
        deliveries_raw += info["deliveries"]; hist_soc[s][k0:k1], hist_q[s][k0:k1] = info["soc"], info["q"]
        for c, dct in info["transfers"].items():
            tr = transfers[s].setdefault(c, {})
            for tk, gb in dct.items(): tr[tk] = tr.get(tk, 0.0) + gb
    return _score(world, cfg, policy, executed, deliveries_raw, timeline, hist_soc, hist_q, transfers, contacts, states, stats, time.time() - t_wall)


# ----------------------------------------------------------------------------- scoring from realised outcomes
def _score(world, cfg, policy, executed, deliveries_raw, timeline, hist_soc, hist_q, transfers, contacts, states, stats, wall):
    """Calculate policy results from completed deliveries, not just scheduled observations."""
    pool, events, dt = world.pool, world.events, cfg.dt
    deliv = {tk: (step * dt, dl, val) for tk, step, dl, val in deliveries_raw if tk is not None}
    recs, served, lat, value_events, dels = [], {}, [], [], []
    for g, ok in executed:
        ev = pool.ev[g]; tk = events[ev].id; td = deliv[tk][0] if (ok and tk in deliv) else None
        on_time = td is not None and td <= events[ev].deadline_t
        recs.append((int(g), ok, td, on_time))
        if on_time: served[ev] = g; lat.append(td - events[ev].arrival_t); value_events.append((td, float(pool.val[g]))); dels.append((tk, td, True, float(pool.val[g])))
    reward = sum(float(pool.val[g]) for g in served.values()); cost = sum(float(pool.cost[g]) for g, _ in executed)
    rush = pool.ev_rush; sv = np.zeros(len(events), bool); sv[list(served)] = True
    by_cat = {c: [int(sum(e.category == c for e in events)), int(sum(sv[i] for i, e in enumerate(events) if e.category == c))] for c in {e.category for e in events}}
    lat = np.array(lat) / H if lat else np.array([np.nan])
    kpi = dict(objective=reward - cfg.lam * cost, reward=reward, cost=cost, tasks=len(events), served=int(sv.sum()),
               sla_rate=float(sv.mean()), rush_total=int(rush.sum()), rush_served=int(sv[rush].sum()), rush_rate=float(sv[rush].mean()) if rush.any() else float("nan"),
               std_rate=float(sv[~rush].mean()), median_latency_h=float(np.nanmedian(lat)), p90_latency_h=float(np.nanpercentile(lat, 90)),
               observations=len(executed), failed=stats["failed"], skipped=stats["skipped"], churn_opt=stats["churn_opt"],
               churn_disruption=stats["churn_disruption"], replans=stats["replans"], wall_s=wall, by_cat=by_cat,
               gb=float(sum(pool.d[g] for g, ok in executed if ok)))
    fuel = {sat.name: (st.fuel_kg, fuel_status(sat, st)) for sat, st in zip(world.fleet, states)}
    return Outcome(policy, kpi, recs, dels, timeline, hist_soc, hist_q, transfers, contacts, fuel, wall)


def run_policies(world, policies=POLICIES, cfg=None, verbose=True):
    """Run the requested strategies on the same scenario and collect their outcomes."""
    out = {}
    for p in policies:
        out[p] = run_policy(world, p, cfg)
        if verbose:
            k = out[p].kpi; print(f"  {p:15s} objective={k['objective']:8.1f}  SLA {100 * k['sla_rate']:5.1f}%  rush {100 * k['rush_rate']:5.1f}%  obs={k['observations']:4d}  ({k['wall_s']:.0f}s)", flush=True)
    return out


def run_study(n_tasks=1500, seeds=(11, 12, 13, 14), policies=POLICIES, cfg=None, verbose=True):
    """Repeat the whole comparison on several independent random order streams (different tasks, arrivals, SLAs).
    Returns {policy: {metric: (mean, std)}} so that conclusions do not rest on one lucky stream."""
    # Keep each stream separate, then summarize how much each metric varies between runs.
    runs = {p: [] for p in policies}
    for sd in seeds:
        W = build_world(n_tasks, seed=sd, cfg=cfg)
        if verbose: print(f" stream seed {sd}: pool {W.pool.n:,} candidates", flush=True)
        for p in policies: runs[p].append(run_policy(W, p).kpi)
    keys = ["objective", "sla_rate", "rush_rate", "std_rate", "median_latency_h", "churn_opt", "churn_disruption", "observations", "wall_s"]
    return {p: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in keys} for p, rs in runs.items()}
