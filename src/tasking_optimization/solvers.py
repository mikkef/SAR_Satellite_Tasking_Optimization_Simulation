"""
solvers.py
==========
Optimisers and plan-repair moves for the tasking problem. They all work on a `Problem` (a view at the current time) and
respect FROZEN observations (`plan.fixed`).

Batch-style solvers : greedy (by score / by value-per-resource), GRASP, LNS (ruin & recreate), LP upper bound.
Operational moves   : `insert_candidates` (free capacity only) and `preempt_candidates` (may displace lower-value observations,
                      used for RUSH tasks).
"""
from __future__ import annotations
import bisect, math, random, time
import numpy as np
from .problem import Plan


# ----------------------------------------------------------------------------- greedy family
def density_key(P):
    """Value per unit of scarce resource: share of the satellite's remaining downlink capacity + share of its energy budget."""
    use = P.d / np.maximum(P.dl_cap[P.s], 1e-9) + P.e / np.maximum(P.e_budget[P.s], 1e-9)
    return P.score / np.maximum(use, 1e-12)


def insert_candidates(P, plan, idxs, key=None):
    """Try to add observations (best key first) WITHOUT removing anything. Each success must keep the plan feasible and improve the objective."""
    idxs = np.asarray(idxs, dtype=int)
    if len(idxs) == 0: return 0
    key = density_key(P) if key is None else key
    n_ok = 0
    for i in idxs[np.argsort(-key[idxs])]:
        n_ok += P.try_insert(plan, int(i)) is not None
    return n_ok


def solve_greedy_density(P, init=None, subset=None, **_):
    plan = init if init is not None else Plan(P)
    insert_candidates(P, plan, np.arange(P.n) if subset is None else subset)
    return plan, {}


def solve_greedy_score(P, **_):
    plan = Plan(P); insert_candidates(P, plan, np.arange(P.n), key=P.score)
    return plan, {}


def solve_grasp(P, restarts=4, sigma=0.25, seed=0, **_):
    """Randomised greedy multi-start (alternating the two rankings)."""
    rng = np.random.default_rng(seed); best, bv = None, -1e18
    for r in range(restarts):
        base = P.score if r % 2 == 0 else density_key(P)
        plan = Plan(P); insert_candidates(P, plan, np.arange(P.n), key=base * np.exp(sigma * rng.standard_normal(P.n)))
        if plan.total > bv: best, bv = plan, plan.total
    return best, {"restarts": restarts}


# ----------------------------------------------------------------------------- large neighbourhood search
def solve_lns(P, init=None, iters=250, seed=0, window_h=3.0, n_sats=3, **_):
    """Ruin & recreate: clear a few satellites' (non-frozen) observations inside a random time window, rebuild that neighbourhood
    greedily (value per scarce resource, randomised) while letting EVERY satellite re-bid for the freed tasks. Accept if not worse."""
    rnd = random.Random(seed); nr = np.random.default_rng(seed)
    plan = init if init is not None else solve_greedy_density(P)[0]
    plan = plan.copy(); dens = density_key(P)
    if P.n == 0: return plan, {"iters": 0, "improving_moves": 0}
    by_ev = {}
    for i, e in enumerate(P.ev): by_ev.setdefault(int(e), []).append(i)
    lo, hi = float(P.t.min()), float(P.t.max()); W = min(window_h * 3600.0, max(hi - lo, 60.0)); S = len(P.sats); improved = 0
    for _ in range(iters):
        t0 = rnd.uniform(lo, max(lo, hi - W)); t1 = t0 + W
        trial = plan.copy(); freed, pool = [], []
        for s in rnd.sample(range(S), min(n_sats, S)):
            gone = [j for j in trial.lists[s] if t0 <= P.t[j] <= t1 and j not in trial.fixed]
            keep = [j for j in trial.lists[s] if j not in set(gone)]
            freed += [P.ev[j] for j in gone]
            ok, v = P.eval_sat(s, keep); P.commit(trial, s, keep, v)
            m = P.by_sat[s]; pool.append(m[(P.t[m] >= t0) & (P.t[m] <= t1)])
        for e in freed: pool.append(np.array(by_ev[int(e)], dtype=int))
        pool = np.unique(np.concatenate(pool)) if pool else np.array([], int)
        if len(pool) == 0: continue
        key = dens[pool] * np.exp(0.25 * nr.standard_normal(len(pool)))
        for j in pool[np.argsort(-key)]: P.try_insert(trial, int(j))
        if trial.total > plan.total + 1e-9: improved += 1
        if trial.total >= plan.total - 1e-9: plan = trial
    return plan, {"iters": iters, "improving_moves": improved}


# ----------------------------------------------------------------------------- preemption (operational move for rush tasks)
def repair_insert(P, plan, i, max_extra=3):
    """Insert candidate i even if it conflicts: remove the (non-frozen) observations it collides with in time/slew and, if
    storage or battery would still be violated, the least valuable non-frozen observations of that satellite (at most `max_extra`).
    Accepted only if the net objective improves. The displaced tasks simply return to the pool."""
    if P.ev[i] in plan.event_of: return None
    s = P.s[i]; lst, tm = plan.lists[s], plan.times[s]
    lo, hi = bisect.bisect_left(tm, P.t[i] - P.win[s]), bisect.bisect_right(tm, P.t[i] + P.win[s])
    drop = set()
    for j in lst[lo:hi]:
        a, b = (j, i) if P.t[j] <= P.t[i] else (i, j)
        if P.t[b] - P.t_end_c[a] < P.need(s, P.angle[a], P.angle[b]):
            if j in plan.fixed: return None                       # cannot displace a frozen observation
            drop.add(j)
    new = sorted([j for j in lst if j not in drop] + [i], key=lambda j: P.t[j])
    if new[0] == i and not P.anchor_slew_ok(i): return None
    for _ in range(max_extra + 1):
        ok, v = P.eval_sat(s, new)
        if ok: break
        rm = [j for j in new if j != i and j not in plan.fixed]
        if not rm: return None
        j = min(rm, key=lambda j: P.score[j]); new.remove(j)
    else:
        return None
    if v - plan.val[s] <= 1e-9: return None
    gain = v - plan.val[s]; P.commit(plan, s, new, v); return gain


def preempt_candidates(P, plan, idxs):
    """For each task (by best candidate first): normal insertion, else preempting insertion. Returns #tasks placed."""
    placed = 0
    for i in np.asarray(idxs, dtype=int)[np.argsort(-P.score[np.asarray(idxs, dtype=int)])]:
        i = int(i)
        if P.ev[i] in plan.event_of: continue
        if P.try_insert(plan, i) is not None or repair_insert(P, plan, i) is not None: placed += 1
    return placed


# ----------------------------------------------------------------------------- relaxation upper bound
def upper_bound(P, horizon_s=86400.0, **_):
    """LP relaxation (HiGHS). Keeps: one observation per task, per-satellite data budget (storage + remaining downlink capacity),
    energy budget and a time budget. Drops slew geometry, battery/queue dynamics and SLA delays -> a valid UPPER BOUND, not a plan."""
    import scipy.sparse as sp
    from scipy.optimize import linprog
    n, S = P.n, len(P.sats)
    ev_ids, ev_row = np.unique(P.ev, return_inverse=True)
    rows, cols, vals, rhs = list(ev_row), list(range(n)), [1.0] * n, [1.0] * len(ev_ids)
    r0 = len(ev_ids); dwell = P.t_end_c - P.t
    for s, sat in enumerate(P.sats):
        st = P.states[s]; idx = P.by_sat[s]
        for w, cap in [(P.d, max(st.storage_cap_gb - st.stored_gb, 0) + P.dl_cap[s]), (P.e, P.e_budget[s]), (dwell + sat.settle_s, horizon_s)]:
            rows += [r0] * len(idx); cols += list(idx); vals += list(w[idx]); rhs.append(cap); r0 += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r0, n))
    res = linprog(-P.score, A_ub=A, b_ub=np.array(rhs), bounds=(0, 1), method="highs")
    return -res.fun, {"rows": r0}


SOLVERS = {"Greedy (score)": solve_greedy_score, "Greedy (value/resource)": solve_greedy_density, "GRASP": solve_grasp,
           "LNS (ruin & recreate)": solve_lns}


def run_all(P, options=None):
    """Run every batch solver on the same problem; LNS starts from the better of the two greedy plans."""
    options, out = options or {}, {}
    for name, fn in SOLVERS.items():
        t = time.time(); kw = dict(options.get(name, {}))
        if name.startswith("LNS"): kw["init"] = max(out["Greedy (score)"], out["Greedy (value/resource)"], key=lambda r: r["objective"])["plan"]
        plan, info = fn(P, **kw)
        out[name] = {"plan": plan, "time": time.time() - t, "info": info, "objective": plan.total, "obs": len(plan.chosen())}
    return out
