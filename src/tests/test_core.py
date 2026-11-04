"""Run with:  python -m pytest -q      (the globe JS test needs `node`; it is skipped if node is missing)"""
import copy, math, random, shutil, subprocess, tempfile, os
import numpy as np
import pytest
from sartasking.geo import RE
from sartasking.satellite_creator import make_fleet
from sartasking.ground_station_creator import make_stations
from sartasking.ground_station_utils import compute_passes, plan_contacts, apply_station_outage
from sartasking.sla_generator_utils import generate_task_stream, stream_summary, RUSH_SLA_HOURS, ARRIVAL_END_H
from sartasking.candidates import observation_chance
from sartasking.resources import SatResources
from sartasking.satellite_update_utils import (initial_state, advance_state, ExecObs, apply_outage, apply_maintenance,
                                               apply_capacity_change, apply_telemetry, fuel_status)
from sartasking.problem import Problem, Plan
from sartasking.solvers import (solve_greedy_density, solve_lns, repair_insert, upper_bound, insert_candidates, preempt_candidates)
from sartasking.rolling import build_world, run_policy, SimConfig

H = 3600.0


@pytest.fixture(scope="module")
def world():
    return build_world(400, seed=5)


def make_problem(world, t=0.0, known=None, horizon=24 * H):
    R = [SatResources(s, world.n_steps, world.cfg.dt) for s in world.fleet]
    for r, s in zip(R, world.fleet): r.set_contacts(world.contacts0.get(s.name, []))
    states = [initial_state(s) for s in world.fleet]
    E = len(world.events)
    known = np.ones(E, bool) if known is None else known
    return Problem(world.pool, R, states, t, known, np.zeros(E, bool), world.cfg.lam, horizon), R, states


# ------------------------------------------------------------------------------------- physics
def test_orbit_is_circular_and_periodic():
    s = make_fleet()[0]
    r, _ = s.eci(np.array([0.0, s.period / 3, s.period]))
    assert np.allclose(np.linalg.norm(r, axis=1), RE + s.altitude_km)
    assert np.allclose(r[0], r[2], atol=1e-3) and 5600 < s.period < 5800


def test_dawn_dusk_plane_has_no_eclipse_and_others_do(world):
    ecl = {s.plane: SatResources(s, world.n_steps, 10.0).eclipse.mean() for s in world.fleet}
    assert ecl["C"] < 0.01 and 0.25 < ecl["A"] < 0.45


def test_lower_orbits_burn_more_propellant():
    f = make_fleet(); a, b = f[0], f[12]                         # 560 km vs 600 km
    assert a.drag_kg_per_day > b.drag_kg_per_day


# ------------------------------------------------------------------------------------- SLA generator
def test_sla_stream_properties(world):
    ev = world.events; s = stream_summary(ev)
    assert 0.25 < s["backlog"] / s["n"] < 0.45                    # a good share of orders is already known at t=0
    assert all(0 <= e.arrival_t <= ARRIVAL_END_H * H for e in ev)
    assert all(e.deadline_t == pytest.approx(e.arrival_t + e.sla_h * H) for e in ev)
    rush = [e for e in ev if e.priority == "rush"]
    assert rush and all(e.sla_h in RUSH_SLA_HOURS for e in rush)
    assert np.mean([e.sla_h for e in rush]) < np.mean([e.sla_h for e in ev if e.priority == "standard"])


def test_disaster_orders_arrive_in_bursts(world):
    arr = np.array([e.arrival_t / H for e in world.events if e.category == "disaster" and e.arrival_t > 0])
    hist = np.histogram(arr, bins=20, range=(0, 20))[0]
    assert hist.max() > 3 * np.median(hist)


def test_vectorised_observation_chance_matches_scalar(world):
    for c in random.Random(0).sample(world.pool.cands, 300):
        assert c.p == pytest.approx(observation_chance(c.ev, c.mode, abs(c.angle)), abs=1e-9)


# ------------------------------------------------------------------------------------- state updates
def _toy_obs(R, n=24):
    ks = np.arange(300, 8300, 8000 // n)[:n]; ka = ks + 2
    ok = ~R.is_blocked(ks, ka); return ks[ok], ka[ok]


def test_anchored_fast_model_matches_slow_simulator(world):
    sat = world.fleet[0]; R = SatResources(sat, world.n_steps, 10.0); R.set_contacts(world.contacts0[sat.name]); R.set_anchor(initial_state(sat))
    ks, ka = _toy_obs(R); e, d = np.full(len(ks), 3.0), np.full(len(ks), 12.0)
    ok, ds = R.evaluate(ks, ka, e, d); ok2, ds2 = R.slow_simulate(ks, ka, e, d)
    assert ok and ok2 and np.array_equal(ds, ds2)


def test_chunked_execution_equals_one_shot_evaluation(world):
    sat = world.fleet[0]; R = SatResources(sat, world.n_steps, 10.0); R.set_contacts(world.contacts0[sat.name]); R.set_anchor(initial_state(sat))
    ks, ka = _toy_obs(R); N = len(ks); e, d = np.full(N, 3.0), np.full(N, 12.0)
    ok, ds = R.evaluate(ks, ka, e, d); assert ok
    state, got, soc = initial_state(sat), {}, []
    for c in range(12):
        t0, t1 = c * 2 * H, (c + 1) * 2 * H
        obs = [ExecObs(j, f"T{j}", ks[j] * 10.0, ks[j] * 10.0 + 20, int(ks[j]), int(ka[j]), 0.0, 3.0, 12.0, 1e9, 1.0) for j in range(N) if t0 <= ks[j] * 10 < t1]
        state, info = advance_state(R, sat, state, obs, t1); soc.append(info["soc"])
        got.update({tk: step for tk, step, _, _ in info["deliveries"]})
    chunk = np.array([got.get(f"T{j}", R.n) for j in range(N)])
    assert np.abs(chunk - ds).max() <= 3
    soc_fast = R.curves(ks, ka, e, d)[0]; soc = np.concatenate(soc)
    assert np.allclose(soc, soc_fast[:len(soc)], atol=1e-6)


def test_reality_checks_skip_infeasible_observations(world):
    sat = world.fleet[0]; R = SatResources(sat, world.n_steps, 10.0); R.set_contacts([]); st = initial_state(sat); R.set_anchor(st)
    greedy_battery = ExecObs(0, "A", 100.0, 120.0, 10, 12, 0.0, 500.0, 5.0, 1e9, 1.0)            # needs far more energy than available
    too_big = ExecObs(1, "B", 500.0, 520.0, 50, 52, 0.0, 1.0, 1000.0, 1e9, 1.0)                  # does not fit in storage
    _, info = advance_state(R, sat, st, [greedy_battery, too_big], 1000.0)
    assert {why for _, why in info["skipped"]} == {"battery", "storage"}


def test_fuel_burns_down_and_maintenance_consumes_propellant(world):
    sat = world.fleet[1]; R = SatResources(sat, world.n_steps, 10.0); R.set_contacts([]); st = initial_state(sat); apply_maintenance(st, 3 * H, 3.5 * H, 0.2)
    st2, _ = advance_state(R, sat, st, [], 6 * H)
    assert st2.fuel_kg < sat.propellant_kg - 0.2 and fuel_status(sat, st2) in ("ok", "low")
    assert (3 * H, 3.5 * H) in st2.unavailable


def test_capacity_and_telemetry_updates():
    sat = make_fleet()[0]; st = initial_state(sat)
    apply_capacity_change(st, storage_cap_gb=60.0, battery_cap_wh=80.0, solar_factor=0.8)
    assert st.storage_cap_gb == 60.0 and st.battery_cap_wh == 80.0 and st.soc_wh <= 80.0 and st.solar_factor == 0.8
    apply_telemetry(st, soc_wh=50.0, stored_gb=st.stored_gb / 2); assert st.soc_wh == 50.0


# ------------------------------------------------------------------------------------- ground segment
def test_station_outage_cancels_and_rebooks_contacts(world):
    new = apply_station_outage(world.contacts0, world.passes, "Svalbard", 8 * H, 14 * H, now=8 * H)
    assert not any(c.gs.name == "Svalbard" and c.t0 < 14 * H and c.t1 > 8 * H for v in new.values() for c in v)
    old_n = sum(len(v) for v in world.contacts0.values()); assert sum(len(v) for v in new.values()) <= old_n + 20
    busy = {}                                                     # an antenna never serves two satellites at once
    for v in new.values():
        for c in v: busy.setdefault((c.gs.name, c.antenna), []).append((c.t0, c.t1))
    for lst in busy.values():
        lst.sort(); assert all(b[0] >= a[1] - 1e-9 for a, b in zip(lst, lst[1:]))


# ------------------------------------------------------------------------------------- problem view / plan
def test_problem_only_contains_known_orders(world):
    known = world.pool.ev_arrival <= 0
    P, _, _ = make_problem(world, 0.0, known)
    assert P.n > 0 and all(world.events[e].arrival_t == 0 for e in np.unique(P.ev))
    P2, _, _ = make_problem(world, 6 * H, world.pool.ev_arrival <= 6 * H, 4 * H)
    assert (P2.t >= 6 * H).all() and (P2.t <= 10 * H).all()


def test_outage_window_removes_candidates(world):
    P, R, states = make_problem(world, 0.0, world.pool.ev_arrival <= 0, 12 * H)
    s = int(np.bincount(P.s).argmax()); n0 = int((P.s == s).sum())
    apply_outage(states[s], 0.0, 6 * H); R[s].set_unavailable(states[s].unavailable)
    E = len(world.events); P2 = Problem(world.pool, R, states, 0.0, world.pool.ev_arrival <= 0, np.zeros(E, bool), world.cfg.lam, 12 * H)
    assert int((P2.s == s).sum()) < n0 and not ((P2.s == s) & (P2.t < 6 * H)).any()


def test_greedy_lns_plans_verified_and_below_lp_bound(world):
    P, _, _ = make_problem(world, 0.0, world.pool.ev_arrival <= 0, 24 * H)
    plan, _ = solve_greedy_density(P); assert P.verify(plan)
    plan2, _ = solve_lns(P, init=plan, iters=15); assert P.verify(plan2) and plan2.total >= plan.total - 1e-9
    assert upper_bound(P)[0] >= plan2.total - 1e-6


def test_frozen_observations_survive_lns_and_preemption(world):
    P, _, _ = make_problem(world, 0.0, world.pool.ev_arrival <= 0, 12 * H)
    plan, _ = solve_greedy_density(P); plan.fixed = {j for j in plan.chosen() if P.t[j] < 3 * H}; before = set(plan.fixed)
    plan2, _ = solve_lns(P, init=plan, iters=30)
    for i in np.where(P.rush)[0][:200]: repair_insert(P, plan2, int(i))
    assert before <= set(plan2.chosen()) and P.verify(plan2)


def test_make_plan_drops_observations_that_became_invalid(world):
    P, R, states = make_problem(world, 0.0, world.pool.ev_arrival <= 0, 12 * H)
    plan, _ = solve_greedy_density(P); gids = [[int(P.gid[j]) for j in l] for l in plan.lists]
    s = int(np.argmax([len(l) for l in plan.lists]))
    apply_outage(states[s], 0.0, 12 * H); R[s].set_unavailable(states[s].unavailable)
    E = len(world.events); P2 = Problem(world.pool, R, states, 0.0, world.pool.ev_arrival <= 0, np.zeros(E, bool), world.cfg.lam, 12 * H)
    plan2, dropped = P2.make_plan(gids, 0.0)
    assert len(dropped) >= len(gids[s]) and plan2.lists[s] == [] and P2.verify(plan2)


# ------------------------------------------------------------------------------------- rolling simulation
@pytest.fixture(scope="module")
def outcomes(world):
    return {p: run_policy(world, p) for p in ["static", "insert_only", "hybrid", "oracle"]}


def test_updating_the_plan_beats_ignoring_new_orders(outcomes):
    k = {p: o.kpi["objective"] for p, o in outcomes.items()}
    assert k["insert_only"] > 1.3 * k["static"] and k["hybrid"] >= k["insert_only"] * 0.97


def test_kpis_are_consistent_with_realised_deliveries(world, outcomes):
    for o in outcomes.values():
        k = o.kpi; assert k["served"] <= k["tasks"] and k["rush_served"] <= k["rush_total"]
        on_time = [r for r in o.executed if r[3]]; assert len(on_time) == k["served"]          # one on-time delivery per served order
        ev_of = world.pool.ev; assert len({ev_of[r[0]] for r in on_time}) == len(on_time)
        for g, ok, td, on in on_time: assert td <= world.events[ev_of[g]].deadline_t and ok


def test_executed_observations_obey_slew_and_resource_limits(world, outcomes):
    pool = world.pool
    for p, o in outcomes.items():
        per = {}
        for g, ok, td, on in o.executed: per.setdefault(pool.s[g], []).append(g)
        for s, gl in per.items():
            gl.sort(key=lambda g: pool.t[g]); sat = world.fleet[s]
            for a, b in zip(gl, gl[1:]):
                assert pool.t[b] - pool.t_end[a] >= sat.slew_need(pool.angle[a], pool.angle[b]) - 1e-6, (p, sat.name)
        for s, sat in enumerate(world.fleet):                     # actual trajectories respect limits (except where a disruption changed them)
            hit = {d.target for d in world.disruptions if d.kind in ("sat_capacity", "telemetry")}
            if sat.name in hit: continue
            assert np.nanmin(o.hist_soc[s]) >= min(sat.soc_min_wh, sat.soc_init_wh) - 1e-6
            assert np.nanmax(o.hist_q[s]) <= sat.storage_gb + 1e-6


def test_acquisition_failures_are_identical_across_policies(outcomes):
    flags = {p: {g: ok for g, ok, _, _ in o.executed} for p, o in outcomes.items() if p != "oracle"}
    common = set.intersection(*[set(f) for f in flags.values()])
    assert common and all(len({f[g] for f in flags.values()}) == 1 for g in common)


def test_oracle_has_no_failures_and_is_best(outcomes):
    assert outcomes["oracle"].kpi["failed"] == 0
    assert outcomes["oracle"].kpi["objective"] >= max(o.kpi["objective"] for p, o in outcomes.items() if p != "oracle") * 0.98


# ------------------------------------------------------------------------------------- globe JS (headless)
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_globe_controls_only_touch_moving_traces(world, outcomes):
    from sartasking import viz
    rings = viz.load_rings(os.path.join(os.path.dirname(__file__), "..", "data", "land_110m.json"))
    with tempfile.TemporaryDirectory() as d:
        D = viz.html_globe(world, outcomes, rings, os.path.join(d, "g.html"))
        js = os.path.join(d, "g.js"); open(js, "w").write(viz.globe_script(D))
        r = subprocess.run(["node", os.path.join(os.path.dirname(__file__), "globe_js_harness.js"), js], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr + r.stdout
