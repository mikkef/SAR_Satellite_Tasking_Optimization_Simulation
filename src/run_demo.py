"""
End-to-end demo of the rolling-horizon SAR tasking study.

  python run_demo.py              # full study (~8 min): main stream + 4 extra order streams + batch solver comparison
  python run_demo.py --quick      # ~1.5 min: main stream only
"""
import argparse, json, os, pickle, time
import numpy as np
from src.tasking_optimization.rolling import build_world, run_policies, run_study, SimConfig, POLICIES
from src.satellites.satellite_update_utils import initial_state
from src.satellites.resources import SatResources
from src.tasking_optimization.problem import Problem
from src.tasking_optimization.solvers import run_all, upper_bound
from src.SLA.sla_generator_utils import stream_summary
from src.vizualization import viz

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", type=int, default=1500); ap.add_argument("--out", default="results")
ap.add_argument("--quick", action="store_true", help="skip the multi-stream study and the batch solver comparison")
ap.add_argument("--no-html", action="store_true"); ap.add_argument("--cache", default=None, help="dev: reuse a pickled main run")
# Read the command-line choices once so the rest of this file can run the requested study.
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T = time.time()

# ------------------------------------------------------------------ 1. main order stream, all policies
# Build one shared set of orders and satellite resources, then run every policy against it.
# Sharing the same input makes the policy comparison fair: only the planning strategy changes.
if a.cache and os.path.exists(a.cache):
    W, out = pickle.load(open(a.cache, "rb")); print("loaded cache")
else:
    W = build_world(a.tasks, seed=7)
    print(f"{len(W.fleet)} satellites, {len(W.stations)} stations, {len(W.events)} orders, {sum(map(len, W.contacts0.values()))} contacts booked, "
          f"{W.pool.n:,} candidate observations; disruptions: {len(W.disruptions)}")
    print("order stream:", json.dumps({k: v for k, v in stream_summary(W.events).items() if k in ("n", "backlog", "rush")}))
    print("\nPolicy comparison on the main stream (scored from what actually happened):")
    out = run_policies(W)
    if a.cache: pickle.dump((W, out), open(a.cache, "wb"))

# ------------------------------------------------------------------ 2. several independent streams
# Optional repeat runs use different order streams. This checks whether the main comparison
# is typical, instead of depending on one especially easy or difficult set of orders.
study = None
if not a.quick:
    print("\nRepeating on independent order streams ...")
    study = run_study(a.tasks, seeds=(11, 12, 13, 14))
    study_main = {p: {k: (v, 0.0) for k, v in dict(objective=out[p].kpi["objective"]).items()} for p in out}
    print(f"\n{'policy':15s} {'objective (mean ± std)':>24s} {'SLA %':>7s} {'rush %':>7s} {'churn':>7s}")
    for p, s in study.items():
        print(f"{p:15s} {s['objective'][0]:14.0f} ± {s['objective'][1]:5.0f} {100 * s['sla_rate'][0]:7.1f} {100 * s['rush_rate'][0]:7.1f} {s['churn_opt'][0] + s['churn_disruption'][0]:7.0f}")

# ------------------------------------------------------------------ 3. batch solver comparison on the hindsight problem
# This separate experiment gives the solvers every order at the start. It is a reference
# for solver quality, not another rolling-operation policy.
batch = None
if not a.quick:
    print("\nBatch solver comparison (all orders known, plan from t=0) ...")
    R = [SatResources(s, W.n_steps, W.cfg.dt) for s in W.fleet]
    for r, s in zip(R, W.fleet): r.set_contacts(W.contacts0.get(s.name, []))
    P = Problem(W.pool, R, [initial_state(s) for s in W.fleet], 0.0, np.ones(len(W.events), bool), np.zeros(len(W.events), bool), W.cfg.lam, W.cfg.t_task)
    res = run_all(P, {"GRASP": {"restarts": 2}, "LNS (ruin & recreate)": {"iters": 250}})
    ub, _ = upper_bound(P)
    batch = {"bound": ub, "solvers": {k: dict(objective=r["objective"], obs=r["obs"], time=r["time"], verified=P.verify(r["plan"])) for k, r in res.items()}}
    for k, r in batch["solvers"].items(): print(f"  {k:26s} objective={r['objective']:8.1f} ({100 * r['objective'] / ub:5.1f}% of LP bound)  obs={r['obs']:4d}  {r['time']:5.1f}s  verified={r['verified']}")
    print(f"  LP relaxation upper bound: {ub:.1f}")

# ------------------------------------------------------------------ 4. results + figures
# Save the numbers and pictures after the simulations finish. These reporting steps read
# the outcomes; they do not feed back into or change any policy's decisions.
kpis = {p: {k: v for k, v in o.kpi.items() if k != "by_cat"} for p, o in out.items()}
json.dump({"main_stream": kpis, "multi_stream": study, "batch": batch}, open(f"{a.out}/results.json", "w"), indent=2, default=float)
rings = viz.load_rings("data/land_110m.json")
viz.fig_earth(W, out["hybrid"], rings, f"{a.out}/earth_overview.png")
viz.fig_policy_dashboard(W, out, f"{a.out}/policy_dashboard.png", study=study)
viz.fig_ops_dashboard(W, out["hybrid"], f"{a.out}/ops_dashboard.png")
if not a.no_html: viz.html_globe(W, out, rings, f"{a.out}/globe_3d.html")
print(f"\ndone in {time.time() - T:.0f}s -> {a.out}/")
