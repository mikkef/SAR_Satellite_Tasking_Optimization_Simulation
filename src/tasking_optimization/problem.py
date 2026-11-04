"""
problem.py
==========
The planning problem at one moment in time.

CandidatePool  : ALL possible observations of the day (satellite x task x product x time), computed once. It is static.
Problem        : a VIEW of the pool at time `t_now`: only tasks the planner knows about (arrived), not yet observed, inside the
                 planning horizon and still able to meet their SLA, evaluated from the carried-over satellite states.
Plan           : the selected observations per satellite (sorted by time); `fixed` observations are frozen and may not be changed.

Objective:  sum( reward * p_obs * season * [delivered before the task deadline] )  -  lambda * sum( cost )
Constraints: slew/overlap, battery, onboard storage / downlink queue (see resources.py), one observation per task,
             no imaging during contacts or outages.
"""
from __future__ import annotations
import bisect
import numpy as np
from ..satellites.satellite_creator import MOD_COST


class CandidatePool:
    """Static arrays describing every candidate observation (index = global id 'gid')."""
    def __init__(self, sats, events, cands, dt: float, month: int = 11):
        sidx = {s.name: i for i, s in enumerate(sats)}; eidx = {e.id: i for i, e in enumerate(events)}
        self.sats, self.events, self.dt, self.cands = sats, events, dt, cands
        self.s = np.array([sidx[c.sat.name] for c in cands]); self.ev = np.array([eidx[c.ev.id] for c in cands])
        self.t = np.array([c.t for c in cands]); dw = np.array([c.mode.dwell for c in cands]); self.t_end = self.t + dw
        self.k = np.rint(self.t / dt).astype(int); self.ka = self.k + np.ceil(dw / dt).astype(int)   # start step / data-ready step
        self.d = np.array([c.mode.data_gb for c in cands]); self.angle = np.array([c.angle for c in cands])
        self.e = np.array([c.sat.energy_wh(c.mode, c.angle) for c in cands])
        self.val = np.array([c.ev.reward * c.p * c.ev.season_factor(month) for c in cands])
        self.cost = np.array([c.ev.times_revisited * MOD_COST[c.mode.mod] for c in cands])
        self.ev_arrival = np.array([e.arrival_t for e in events]); self.ev_deadline = np.array([e.deadline_t for e in events])
        self.ev_rush = np.array([e.priority == "rush" for e in events])
        for i, c in enumerate(cands): c.gid = i
        self.n = len(cands)


class Plan:
    """Selected observations: per-satellite lists of LOCAL candidate indices, kept sorted by time, with cached per-satellite value."""
    def __init__(self, P):
        S = len(P.sats)
        self.lists = [[] for _ in range(S)]; self.times = [[] for _ in range(S)]
        self.val = [0.0] * S; self.event_of = {}; self.fixed = set()

    def copy(self):
        q = Plan.__new__(Plan)
        q.lists = [l[:] for l in self.lists]; q.times = [t[:] for t in self.times]; q.val = self.val[:]
        q.event_of = dict(self.event_of); q.fixed = set(self.fixed)
        return q

    @property
    def total(self): return float(sum(self.val))
    def chosen(self): return [i for l in self.lists for i in l]


class Problem:
    def __init__(self, pool: CandidatePool, res, states, t_now: float, known, done, lam=0.5, horizon_s=8 * 3600.0):
        self.pool, self.res, self.states, self.t_now, self.lam = pool, res, states, t_now, lam
        self.sats, self.dt = pool.sats, pool.dt
        self.n_steps = res[0].n
        for R, st in zip(res, states): R.set_anchor(st)                       # plan FROM the carried state, not from t=0
        # ---- which candidates are relevant right now?
        m = known[pool.ev] & ~done[pool.ev] & (pool.t >= np.maximum(t_now, pool.ev_arrival[pool.ev])) & (pool.t <= t_now + horizon_s)
        ix = np.where(m)[0]
        S = pool.s[ix]; keep = np.zeros(len(ix), bool)
        for s, (R, st, sat) in enumerate(zip(res, states, self.sats)):
            sel = np.where(S == s)[0]
            if not len(sel): continue
            g = ix[sel]
            ok = ~R.is_blocked(pool.k[g], pool.ka[g])                          # not while pointing at an antenna / during an outage
            ok &= (pool.t[g] - max(t_now, st.last_end_t)) >= np.abs(pool.angle[g] - st.roll_deg) * sat.slew_s_per_deg + sat.settle_s
            ds = R.standalone_delivery_step(pool.ka[g], pool.d[g])             # can the SLA be met even in the best case?
            ok &= (ds < self.n_steps) & (ds * self.dt <= pool.ev_deadline[pool.ev[g]])
            keep[sel[ok]] = True
        score = pool.val[ix] - lam * pool.cost[ix]
        ix = ix[keep & (score > 0)]
        self.gid = ix; self.n = len(ix); self.gid2loc = dict(zip(ix.tolist(), range(self.n)))
        self.s, self.t, self.t_end_c = pool.s[ix], pool.t[ix], pool.t_end[ix]
        self.k, self.ka, self.d, self.e, self.angle = pool.k[ix], pool.ka[ix], pool.d[ix], pool.e[ix], pool.angle[ix]
        self.val, self.cost, self.ev = pool.val[ix], pool.cost[ix], pool.ev[ix]
        self.deadline = pool.ev_deadline[self.ev]; self.rush = pool.ev_rush[self.ev]
        self.score = self.val - lam * self.cost
        self.cands = [pool.cands[g] for g in ix]
        self.slew = np.array([s.slew_s_per_deg for s in self.sats]); self.settle = np.array([s.settle_s for s in self.sats])
        self.win = [2 * s.max_angle * s.slew_s_per_deg + s.settle_s + 40 for s in self.sats]
        self.by_sat = [np.where(self.s == s)[0] for s in range(len(self.sats))]
        self.dl_cap = np.array([R.dl_capacity_gb for R in res]); self.e_budget = np.array([R.energy_budget_wh for R in res])

    # ------------------------------------------------------------------ evaluation of one satellite timeline
    def need(self, s, a1, a2): return abs(a2 - a1) * self.slew[s] + self.settle[s]

    def eval_sat(self, s, idxs):
        """-> (feasible, value of this satellite's future observations). Value counts only products delivered before their deadline."""
        if not idxs: return True, 0.0
        ia = np.array(idxs)
        ok, ds = self.res[s].evaluate(self.k[ia], self.ka[ia], self.e[ia], self.d[ia])
        if not ok: return False, 0.0
        good = (ds < self.n_steps) & (ds * self.dt <= self.deadline[ia])
        return True, float((self.val[ia] * good).sum() - self.lam * self.cost[ia].sum())

    def anchor_slew_ok(self, i):
        """Can the satellite get from its CURRENT attitude to candidate i (used when i would be the first future observation)?"""
        s, st = self.s[i], self.states[self.s[i]]
        return self.t[i] - max(self.t_now, st.last_end_t) >= self.need(s, st.roll_deg, self.angle[i])

    def slew_ok(self, plan, i):
        s = self.s[i]; lst, tm = plan.lists[s], plan.times[s]
        pos = bisect.bisect_left(tm, self.t[i])
        if pos > 0:
            p = lst[pos - 1]
            if self.t[i] - self.t_end_c[p] < self.need(s, self.angle[p], self.angle[i]): return False
        elif not self.anchor_slew_ok(i): return False
        if pos < len(lst):
            q = lst[pos]
            if self.t[q] - self.t_end_c[i] < self.need(s, self.angle[i], self.angle[q]): return False
        return True

    def commit(self, plan, s, new_list, new_val):
        for j in plan.lists[s]: plan.event_of.pop(self.ev[j], None)
        plan.lists[s] = new_list; plan.times[s] = [self.t[j] for j in new_list]; plan.val[s] = new_val
        for j in new_list: plan.event_of[self.ev[j]] = j

    def try_insert(self, plan, i, min_gain=1e-9):
        """Insert candidate i if its task is still free, slew allows it, every resource limit holds and the objective improves."""
        if self.ev[i] in plan.event_of or not self.slew_ok(plan, i): return None
        s = self.s[i]; pos = bisect.bisect_left(plan.times[s], self.t[i])
        new = plan.lists[s][:pos] + [i] + plan.lists[s][pos:]
        ok, v = self.eval_sat(s, new)
        if not ok or v - plan.val[s] <= min_gain: return None
        gain = v - plan.val[s]; self.commit(plan, s, new, v); return gain

    # ------------------------------------------------------------------ carry a plan over from the previous planning step
    def make_plan(self, prev_gids, fixed_until):
        """Rebuild the previous plan inside this problem. Observations that are no longer valid (outage, contact lost, task
        gone ...) are dropped; if the plan became infeasible, the least valuable observations are removed until it is feasible.
        Observations starting before `fixed_until` are FROZEN. Returns (plan, dropped_gids)."""
        plan, dropped = Plan(self), []
        for s, gl in enumerate(prev_gids):
            loc = sorted((self.gid2loc[g] for g in gl if g in self.gid2loc), key=lambda j: self.t[j])
            dropped += [g for g in gl if g not in self.gid2loc]
            ok, v = self.eval_sat(s, loc)
            while not ok and loc:
                j = min(loc, key=lambda j: self.score[j]); loc.remove(j); dropped.append(int(self.gid[j])); ok, v = self.eval_sat(s, loc)
            self.commit(plan, s, loc, v)
            plan.fixed |= {j for j in loc if self.t[j] < fixed_until}
        return plan, dropped

    # ------------------------------------------------------------------ independent verification (slow simulator)
    def verify(self, plan) -> bool:
        seen, total = set(), 0.0
        for s, lst in enumerate(plan.lists):
            st = self.states[s]; prev_a, prev_end = st.roll_deg, max(self.t_now, st.last_end_t)
            for j in lst:
                if self.ev[j] in seen: return False
                seen.add(self.ev[j])
                if self.t[j] - prev_end < self.need(s, prev_a, self.angle[j]) - 1e-9: return False
                prev_a, prev_end = self.angle[j], self.t_end_c[j]
            if not lst: continue
            ia = np.array(lst)
            ok, ds = self.res[s].slow_simulate(self.k[ia], self.ka[ia], self.e[ia], self.d[ia])
            if not ok: return False
            good = (ds < self.n_steps) & (ds * self.dt <= self.deadline[ia])
            total += float((self.val[ia] * good).sum() - self.lam * self.cost[ia].sum())
        return abs(total - plan.total) < 1e-6
