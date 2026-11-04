"""
viz.py
======
Figures for the rolling-horizon study:
  fig_earth             world map + two 3D globe views of what was actually delivered
  fig_policy_dashboard  policy comparison: value, SLA attainment, cumulative value, backlog, plan churn, latency
  fig_ops_dashboard     resources of individual satellites under disruptions: storage, battery, split downlinks, fuel, stations
  html_globe            interactive 3D globe (rotate freely WHILE it plays; switch policy; play / pause / scrub)
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ..earth_sun_orbits.geo import RE, latlon_to_xyz, hhmm

CAT_COLORS = {"maritime": "#1f77b4", "infrastructure": "#ff7f0e", "disaster": "#d62728", "environment": "#2ca02c"}
PLANE_COLORS = {"A": "#4c78a8", "B": "#f58518", "C": "#54a24b", "D": "#b279a2"}
POLICY_COLORS = {"static": "#9d9d9d", "insert_only": "#6baed6", "insert_preempt": "#3182bd", "rolling": "#fd8d3c", "hybrid": "#d62728", "oracle": "#2ca02c"}
H = 3600.0


def load_rings(path):
    gj = json.load(open(path)); rings = []
    for f in gj["features"]:
        g = f["geometry"]
        for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
            rings += [np.array(r) for r in poly]
    return rings


def split_wrap(lon, lat):
    """Split a path wherever it crosses the antimeridian so it is not drawn across the whole map."""
    lon, lat = np.array(lon, float), np.array(lat, float)
    j = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    return list(zip(np.split(lon, j), np.split(lat, j)))


def served_obs(world, outcome):
    """Observations that produced data delivered before the task deadline: list of pool Candidates."""
    return [world.pool.cands[g] for g, ok, td, on in outcome.executed if on]


def _mark_disruptions(ax, world, ymax=None, label=True):
    for d in world.disruptions:
        if d.kind in ("sat_outage", "gs_outage"): ax.axvspan(d.t_start / H, d.t_end / H, color="red", alpha=.07, lw=0)
        else: ax.axvline(d.announce_t / H, color="red", ls=":", lw=1, alpha=.6)


# ===================================================================================== world map + 3D globes
def fig_earth(world, outcome, rings, path, title=""):
    obs = served_obs(world, outcome); served = {c.ev.id for c in obs}
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], hspace=0.04, wspace=0.0, left=0.02, right=0.98, top=0.96, bottom=0.01)
    ax = fig.add_subplot(gs[0, :])
    for r in rings:
        for lo, la in split_wrap(r[:, 0], r[:, 1]): ax.plot(lo, la, color="0.6", lw=.5)
    tt = np.arange(0, 2.5 * 3600, 20)
    for pl in "ABCD":
        sat = next(s for s in world.fleet if s.plane == pl); la, lo = sat.latlon(tt)
        for a, b in split_wrap(lo, la): ax.plot(a, b, color=PLANE_COLORS[pl], lw=1.1, alpha=.7)
        ax.plot([], [], color=PLANE_COLORS[pl], lw=3, label=f"plane {pl} (e.g. {sat.name}, {sat.inclination_deg:.0f}°)")
    for e in world.events:
        if e.id not in served: ax.scatter(e.position[1], e.position[0], s=3, color="0.78", zorder=2)
    for c in obs:
        la, lo = c.sat.latlon(np.array([c.t]))
        if abs(lo[0] - c.ev.position[1]) < 180: ax.plot([lo[0], c.ev.position[1]], [la[0], c.ev.position[0]], color=CAT_COLORS[c.ev.category], lw=.4, alpha=.5)
        ax.scatter(c.ev.position[1], c.ev.position[0], s=10 + 4 * c.ev.reward, color=CAT_COLORS[c.ev.category], edgecolor="k", linewidth=.3, zorder=4)
    for g in world.stations:
        ax.scatter(g.lon, g.lat, marker="^", s=90, color="gold", edgecolor="k", zorder=6); ax.text(g.lon + 2, g.lat + 2, g.name, fontsize=7, zorder=6)
    for cat, col in CAT_COLORS.items(): ax.scatter([], [], color=col, label=f"{cat} (delivered in SLA)", edgecolor="k", linewidth=.3)
    ax.scatter([], [], marker="^", color="gold", edgecolor="k", label="ground station"); ax.scatter([], [], s=3, color="0.78", label="not delivered in SLA")
    ax.set_xlim(-180, 180); ax.set_ylim(-80, 85); ax.set_aspect("equal"); ax.grid(alpha=.2); ax.legend(ncol=5, fontsize=8, loc="lower center", frameon=True)
    ax.set_title(title or f"Policy '{outcome.policy}': {len(served)} of {len(world.events)} orders delivered within their SLA  (tracks: first 2.5 h of one satellite per plane)", fontsize=11)

    def globe(ax3, clat, clon):
        cam = latlon_to_xyz(np.array(clat), np.array(clon), 1.0)
        vis = lambda p: ((p @ cam) > 0) | (np.linalg.norm(p - (p @ cam)[:, None] * cam, axis=1) > RE)
        def put(p, **kw):
            p = np.array(p, float); p[~vis(p)] = np.nan; ax3.plot(p[:, 0], p[:, 1], p[:, 2], **kw)
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 90), np.linspace(0, np.pi, 45))
        ax3.plot_surface(RE * np.cos(u) * np.sin(v), RE * np.sin(u) * np.sin(v), RE * np.cos(v), color="#cfe3f2", linewidth=0, shade=False, zorder=1)
        for r in rings: put(latlon_to_xyz(r[:, 1], r[:, 0], RE * 1.003), color="#4a6a4a", lw=.5, zorder=5)
        for pl in "ABCD":
            sat = next(s for s in world.fleet if s.plane == pl); put(sat.ecef(np.arange(0, sat.period, 30))[0], color=PLANE_COLORS[pl], lw=1.0, alpha=.6, zorder=5)
        for c in obs:
            sp, ep = c.sat.ecef(np.array([c.t]))[0][0], c.ev.xyz
            put(np.array([sp, ep]), color=CAT_COLORS[c.ev.category], lw=.35, alpha=.6, zorder=6)
            if vis(ep[None])[0]: ax3.scatter(*(ep * 1.003), s=8 + 3 * c.ev.reward, color=CAT_COLORS[c.ev.category], edgecolor="k", linewidth=.3, depthshade=False, zorder=8)
        for g in world.stations:
            if vis(g.xyz[None])[0]: ax3.scatter(*(g.xyz * 1.01), marker="^", s=70, color="gold", edgecolor="k", depthshade=False, zorder=9)
        lim = RE * 1.12
        ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim); ax3.set_zlim(-lim, lim)
        ax3.set_box_aspect((1, 1, 1), zoom=1.55); ax3.set_proj_type("ortho"); ax3.view_init(elev=clat, azim=clon); ax3.set_axis_off()
        ax3.set_title(f"3D globe centered on lat {clat}°, lon {clon}°\n(thin lines = line of sight at imaging time; arcs = orbits)", fontsize=10, y=0.93)
    for k, (la, lo) in enumerate([(25, 30), (10, 115)]):
        globe(fig.add_subplot(gs[1, k], projection="3d", computed_zorder=False), la, lo)
    fig.savefig(path, dpi=105); plt.close(fig)


# ===================================================================================== policy comparison
def fig_policy_dashboard(world, outcomes, path, focus="hybrid", study=None):
    """`study` (optional) = {policy: {metric: (mean, std)}} aggregated over several random order streams."""
    names = [p for p in outcomes]; ev_by_id = {e.id: e for e in world.events}
    fig = plt.figure(figsize=(19, 15)); G = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.18, left=0.05, right=0.98, top=0.94, bottom=0.05)
    ref = outcomes["oracle"].kpi["objective"] if "oracle" in outcomes else max(o.kpi["objective"] for o in outcomes.values())

    ax = fig.add_subplot(G[0, 0]); vals = [outcomes[p].kpi["objective"] for p in names]
    bars = ax.bar(range(len(names)), vals, color=[POLICY_COLORS[p] for p in names])
    if study:
        ax.errorbar(range(len(names)), [study[p]["objective"][0] for p in names], yerr=[study[p]["objective"][1] for p in names], fmt="D", color="k", ms=5, capsize=4, label="mean ± std over several order streams")
        ax.legend(fontsize=8, loc="upper left")
    for b, p, v in zip(bars, names, vals): ax.text(b.get_x() + .4, v + ref * .01, f"{v:.0f}\n{100 * v / ref:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("realised objective")
    ax.set_title("Realised value = reward of products delivered before their deadline − λ·cost  (% of foresight reference)", fontsize=10)

    ax = fig.add_subplot(G[0, 1]); w = .27; X = np.arange(len(names))
    for k, (key, lab, col) in enumerate([("sla_rate", "all orders", "#4c78a8"), ("rush_rate", "RUSH orders", "#d62728"), ("std_rate", "standard orders", "#9ecae1")]):
        ax.bar(X + (k - 1) * w, [100 * outcomes[p].kpi[key] for p in names], w, label=lab, color=col)
    ax.set_xticks(X); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("% delivered within SLA"); ax.legend(fontsize=8)
    ax.set_title("SLA attainment (the day is oversubscribed, so < 100% is expected)", fontsize=10)

    ax = fig.add_subplot(G[1, 0])
    for p in names:
        ev = sorted((td / H, v) for _, td, on, v in outcomes[p].deliveries)
        if ev: ax.plot([0] + [e[0] for e in ev], np.concatenate([[0], np.cumsum([e[1] for e in ev])]), color=POLICY_COLORS[p], lw=1.8, label=p)
    _mark_disruptions(ax, world); ax.set_xlabel("hours since 00:00 UTC"); ax.set_ylabel("cumulative realised reward"); ax.legend(fontsize=8); ax.set_xlim(0, 30)
    ax.set_title("Value arrives as products are DELIVERED (red = outages / events: safe mode, station down, degradation, telemetry)", fontsize=10)

    ax = fig.add_subplot(G[1, 1])
    for p in names:
        tl = outcomes[p].timeline; ax.plot(np.array(tl["t"]) / H, tl["backlog"], color=POLICY_COLORS[p], lw=1.6, label=p)
    _mark_disruptions(ax, world); ax.set_xlabel("hours since 00:00 UTC"); ax.set_ylabel("known orders not yet observed"); ax.legend(fontsize=8)
    ax.set_title("Backlog of orders waiting for an observation", fontsize=10)

    ax = fig.add_subplot(G[2, 0]); X = np.arange(len(names))
    co = [outcomes[p].kpi["churn_opt"] for p in names]; cd = [outcomes[p].kpi["churn_disruption"] for p in names]
    ax.bar(X, co, color="#fd8d3c", label="planned observations moved/removed by optimisation"); ax.bar(X, cd, bottom=co, color="#d62728", label="invalidated by a disruption")
    for i, p in enumerate(names): ax.text(i, co[i] + cd[i] + 10, f"{outcomes[p].kpi['wall_s']:.0f}s compute", ha="center", fontsize=8)
    ax.set_xticks(X); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("# future observations changed"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Plan stability: what operations has to re-uplink (churn) vs. compute spent", fontsize=10)

    ax = fig.add_subplot(G[2, 1]); o = outcomes[focus]
    lat_r = [(td - ev_by_id[tk].arrival_t) / H for tk, td, on, v in o.deliveries if ev_by_id[tk].priority == "rush"]
    lat_s = [(td - ev_by_id[tk].arrival_t) / H for tk, td, on, v in o.deliveries if ev_by_id[tk].priority == "standard"]
    bins = np.linspace(0, 48, 49)
    ax.hist(lat_s, bins=bins, color="#9ecae1", label=f"standard (n={len(lat_s)})"); ax.hist(lat_r, bins=bins, color="#d62728", alpha=.85, label=f"rush (n={len(lat_r)}, SLA 3-6 h)")
    ax.set_xlabel("hours from ORDER to delivered product"); ax.set_ylabel("# orders"); ax.legend(fontsize=8)
    ax.set_title(f"Order-to-delivery latency, policy '{focus}' (SLA clock starts when the order arrives)", fontsize=10)
    fig.suptitle("Rolling-horizon tasking: how should an already-tasked fleet react to new SLAs and disruptions?", fontsize=13)
    fig.savefig(path, dpi=100); plt.close(fig)


# ===================================================================================== resources under disruptions
def fig_ops_dashboard(world, outcome, path, storage_sat="VX-B6", battery_sat="VX-A4"):
    sidx = {s.name: i for i, s in enumerate(world.fleet)}; hrs = np.arange(world.n_steps) * world.cfg.dt / H
    fig = plt.figure(figsize=(19, 15)); G = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.18, left=0.05, right=0.98, top=0.94, bottom=0.06)
    # storage with the capacity step caused by the degradation disruption
    s = sidx[storage_sat]; sat = world.fleet[s]; ax = fig.add_subplot(G[0, 0])
    ax.plot(hrs, outcome.hist_q[s], color="#e45756", lw=1.2)
    cap = np.full(len(hrs), sat.storage_gb)
    for d in world.disruptions:
        if d.kind == "sat_capacity" and d.target == storage_sat and "storage_cap_gb" in d.params: cap[hrs >= d.announce_t / H] = d.params["storage_cap_gb"]
    ax.plot(hrs, cap, "k--", lw=1, label="usable storage")
    for c in outcome.contacts.get(storage_sat, []): ax.axvspan(c.t0 / H, c.t1 / H, color="gold", alpha=.35, lw=0)
    _mark_disruptions(ax, world); ax.axvline(24, color="0.4", ls=":"); ax.set_xlim(0, 30); ax.set_ylim(0, sat.storage_gb * 1.1)
    ax.set_xlabel("hours since 00:00 UTC"); ax.set_ylabel("onboard data (GB)"); ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"{storage_sat}: storage (actual trajectory); gold = downlink contacts; capacity drops when the partition degrades", fontsize=10)
    # battery with telemetry correction / eclipse
    s2 = sidx[battery_sat]; sat2 = world.fleet[s2]; ax = fig.add_subplot(G[0, 1])
    ax.plot(hrs, outcome.hist_soc[s2], color="#4c78a8", lw=1.2)
    bcap = np.full(len(hrs), sat2.battery_wh)
    for d in world.disruptions:
        if d.kind == "sat_capacity" and d.target == battery_sat and "battery_cap_wh" in d.params: bcap[hrs >= d.announce_t / H] = d.params["battery_cap_wh"]
    ax.axhline(sat2.soc_min_wh, color="r", ls="--", lw=1, label="battery floor"); ax.plot(hrs, bcap, "k--", lw=1, label="usable capacity")
    ecl = np.array([False] * world.n_steps)
    from ..satellites.resources import SatResources
    ecl = SatResources(sat2, world.n_steps, world.cfg.dt).eclipse.astype(int); ch = np.diff(np.concatenate([[0], ecl, [0]]))
    for a, b in zip(np.where(ch == 1)[0], np.where(ch == -1)[0]): ax.axvspan(hrs[a], hrs[min(b, len(hrs) - 1)], color="0.6", alpha=.3, lw=0)
    _mark_disruptions(ax, world); ax.set_xlim(0, 30); ax.set_xlabel("hours since 00:00 UTC"); ax.set_ylabel("battery (Wh)"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"{battery_sat}: battery (grey = eclipse); capacity fades at 14 h, telemetry correction at 16 h", fontsize=10)
    # split downlinks
    ax = fig.add_subplot(G[1, :]); tr = outcome.transfers[s]; cons = sorted(tr, key=lambda c: c.t0); cons = [c for c in cons if c.t0 / H < 24][:16]
    pal = plt.cm.tab20(np.linspace(0, 1, 20)); colour = {}
    for xi, c in enumerate(cons):
        bottom = 0.0
        for tk, gb in tr[c].items():
            col = "0.75" if tk is None else colour.setdefault(tk, pal[len(colour) % 20])
            ax.bar(xi, gb, bottom=bottom, color=col, edgecolor="k", linewidth=.4); bottom += gb
        ax.plot([xi - .42, xi + .42], [c.capacity_gb] * 2, color="k", lw=1.5)
    ax.set_xticks(range(len(cons))); ax.set_xticklabels([f"{c.gs.name}\n{hhmm(c.t0)}" for c in cons], fontsize=7); ax.set_ylabel("GB transferred")
    ax.set_title(f"{storage_sat}: each colour = one product; products are split across consecutive passes and DIFFERENT stations (grey = old backlog, black = contact capacity)", fontsize=10)
    # peak storage + fuel per satellite
    ax = fig.add_subplot(G[2, 0]); nm = [x.name for x in world.fleet]; X = np.arange(len(nm))
    pk = [100 * np.nanmax(outcome.hist_q[i]) / world.fleet[i].storage_gb for i in range(len(nm))]
    ax.bar(X - .2, pk, .4, color="#e45756", label="peak storage (% of original capacity)")
    fu = [100 * outcome.fuel[n][0] / world.fleet[i].propellant_kg for i, n in enumerate(nm)]
    ax.bar(X + .2, fu, .4, color="#8c6d31", label="propellant left (%) after 30 h")
    ax.set_xticks(X); ax.set_xticklabels(nm, rotation=90, fontsize=7); ax.set_ylim(0, 125); ax.legend(fontsize=8, ncol=2, loc="upper center")
    ax.set_title("Peak storage per satellite, and propellant left (drag make-up + the VX-A2 maintenance burn: tiny over one day, but tracked)", fontsize=10)
    # ground-station usage
    ax = fig.add_subplot(G[2, 1]); cap_g, used = {}, {}
    for s2_, lst in outcome.contacts.items():
        for c in lst: cap_g[c.gs.name] = cap_g.get(c.gs.name, 0) + c.capacity_gb
    for i in range(len(nm)):
        for c, dct in outcome.transfers[i].items(): used[c.gs.name] = used.get(c.gs.name, 0) + sum(dct.values())
    gn = sorted(cap_g, key=lambda n: -cap_g[n]); X = np.arange(len(gn))
    ax.bar(X, [cap_g[n] for n in gn], color="0.82", label="booked contact capacity"); ax.bar(X, [min(used.get(n, 0), cap_g[n]) for n in gn], color="#54a24b", label="GB downlinked")
    ax.set_xticks(X); ax.set_xticklabels(gn, rotation=45, ha="right", fontsize=8); ax.set_ylabel("GB"); ax.legend(fontsize=8)
    ax.set_title("Ground-segment usage (Svalbard capacity is lost during its 08-14 h outage and partly re-booked elsewhere)", fontsize=10)
    fig.suptitle(f"Resources under disruptions - actual trajectories of policy '{outcome.policy}'", fontsize=13)
    fig.savefig(path, dpi=100); plt.close(fig)


# ===================================================================================== interactive 3D globe
_JS = r"""
(function () {
  var gd = document.getElementById('{plot_id}');
  var D = __DATA__;
  var F = D.labels.length, cur = 0, timer = null, policy = D.focus;
  var ui = document.createElement('div');
  ui.style.cssText = 'position:fixed;left:14px;bottom:14px;z-index:20;color:#eee;font:13px sans-serif;background:rgba(10,16,32,.88);padding:8px 12px;border-radius:8px;display:flex;gap:10px;align-items:center';
  ui.innerHTML = '<select id="sar_pol"></select><button id="sar_play" style="min-width:74px">&#9654; play</button>' +
                 '<input id="sar_sl" type="range" min="0" max="' + (F - 1) + '" value="0" style="width:320px"><span id="sar_lab" style="min-width:110px"></span>' +
                 '<select id="sar_spd"><option value="1">1x</option><option value="2">2x</option><option value="4">4x</option></select>';
  document.body.appendChild(ui);
  var sel = document.getElementById('sar_pol'), btn = document.getElementById('sar_play'), sl = document.getElementById('sar_sl'),
      lab = document.getElementById('sar_lab'), spd = document.getElementById('sar_spd');
  D.policies.forEach(function (p) { var o = document.createElement('option'); o.value = p; o.textContent = D.names[p]; sel.appendChild(o); });
  sel.value = policy;
  // Only the three MOVING traces are updated (satellites, downlink beams, imaging lines). Plotly.restyle never touches the camera,
  // so the globe can be rotated / zoomed freely while the animation runs.
  function draw(f) {
    cur = f; sl.value = f; lab.textContent = D.labels[f];
    Plotly.restyle(gd, {x: [D.sx[f], D.bx[f], D.ox[policy][f]], y: [D.sy[f], D.by[f], D.oy[policy][f]], z: [D.sz[f], D.bz[f], D.oz[policy][f]]},
                   [D.i_sat, D.i_beam, D.i_obs]);
  }
  function setPolicy(p) {
    policy = p;
    D.policies.forEach(function (q) { Plotly.restyle(gd, {visible: q === p}, D.static_idx[q]); });
    draw(cur);
  }
  function stop() { if (timer) { clearInterval(timer); timer = null; } btn.innerHTML = '&#9654; play'; }
  function start() {
    btn.innerHTML = '&#10074;&#10074; pause';
    timer = setInterval(function () { draw(cur + 1 >= F ? 0 : cur + 1); }, 90 / parseFloat(spd.value));
  }
  btn.onclick = function () { if (timer) stop(); else start(); };
  sl.oninput = function () { draw(parseInt(sl.value)); };
  spd.onchange = function () { if (timer) { stop(); start(); } };
  sel.onchange = function () { setPolicy(sel.value); };
  window.__sar = {draw: draw, setPolicy: setPolicy, start: start, stop: stop, state: function () { return {cur: cur, policy: policy, playing: !!timer}; }};
  draw(0);
})();
"""


def globe_script(D) -> str:
    """The JavaScript that drives the globe (controls + animation). Kept separate so it can be unit-tested with Node."""
    return _JS.replace("__DATA__", json.dumps(D, separators=(",", ":")))


def html_globe(world, outcomes, rings, path, focus="hybrid", frame_s=120.0, t_total=30 * H):
    import plotly.graph_objects as go
    fleet = world.fleet; S = len(fleet)
    fig = go.Figure(); R3 = lambda a: [round(float(x), 1) for x in a]
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 90), np.linspace(0, np.pi, 45))
    fig.add_trace(go.Surface(x=RE * np.cos(u) * np.sin(v), y=RE * np.sin(u) * np.sin(v), z=RE * np.cos(v), showscale=False,
                             colorscale=[[0, "#16263d"], [1, "#16263d"]], hoverinfo="skip", lighting=dict(ambient=.9, diffuse=.2), name="Earth"))
    cx, cy, cz = [], [], []
    for r in rings:
        p = latlon_to_xyz(r[:, 1], r[:, 0], RE * 1.002); cx += list(p[:, 0]) + [None]; cy += list(p[:, 1]) + [None]; cz += list(p[:, 2]) + [None]
    fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode="lines", line=dict(color="#7fb89a", width=1.5), hoverinfo="skip", name="coast"))
    gx = np.stack([g.xyz * 1.004 for g in world.stations])
    fig.add_trace(go.Scatter3d(x=gx[:, 0], y=gx[:, 1], z=gx[:, 2], mode="markers+text", text=[g.name for g in world.stations], textfont=dict(color="gold", size=9),
                               marker=dict(size=5, color="gold", symbol="diamond"), hoverinfo="text", name="ground stations"))
    for pl in "ABCD":
        sat = next(s for s in fleet if s.plane == pl); p = sat.ecef(np.arange(0, sat.period, 30))[0]
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines", line=dict(color=PLANE_COLORS[pl], width=2), opacity=.4, name=f"plane {pl} orbit", hoverinfo="name"))
    tf = np.arange(0, t_total + 1, frame_s); F = len(tf)
    sp = np.stack([s.ecef(tf)[0] for s in fleet])                                   # (S, F, 3)
    i_sat = len(fig.data)
    fig.add_trace(go.Scatter3d(x=R3(sp[:, 0, 0]), y=R3(sp[:, 0, 1]), z=R3(sp[:, 0, 2]), mode="markers", marker=dict(size=5, color=[PLANE_COLORS[s.plane] for s in fleet], line=dict(color="white", width=1)),
                               text=[f"{s.name} ({s.gen}, {s.inclination_deg:.0f}°)" for s in fleet], hoverinfo="text", name="satellites"))
    i_beam = len(fig.data); fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="gold", width=4), name="downlink beam", hoverinfo="skip"))
    i_obs = len(fig.data); fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="#ff4d4d", width=3), name="imaging (selected policy)", hoverinfo="skip"))
    sidx = {s.name: i for i, s in enumerate(fleet)}
    def seg(lst):
        x, y, z = [], [], []
        for a, b in lst: x += [round(float(a[0]), 1), round(float(b[0]), 1), None]; y += [round(float(a[1]), 1), round(float(b[1]), 1), None]; z += [round(float(a[2]), 1), round(float(b[2]), 1), None]
        return x, y, z
    # beams (same contact plan for every policy)
    contacts = next(iter(outcomes.values())).contacts; beams = [[] for _ in range(F)]
    for name, lst in contacts.items():
        s = sidx[name]
        for c in lst:
            for f in range(int(np.ceil(c.t0 / frame_s)), min(F, int(c.t1 // frame_s) + 1)): beams[f].append((sp[s, f], c.gs.xyz))
    bx = [seg(b) for b in beams]
    D = dict(focus=focus, policies=list(outcomes), names={}, static_idx={}, ox={}, oy={}, oz={}, i_sat=i_sat, i_beam=i_beam, i_obs=i_obs)
    for p, o in outcomes.items():
        cs = served_obs(world, o); k = o.kpi
        D["names"][p] = f"{p}  (value {k['objective']:.0f}, SLA {100 * k['sla_rate']:.0f}%, rush {100 * k['rush_rate']:.0f}%)"
        first = len(fig.data); vis = (p == focus)
        fig.add_trace(go.Scatter3d(x=[c.ev.xyz[0] * 1.003 for c in cs], y=[c.ev.xyz[1] * 1.003 for c in cs], z=[c.ev.xyz[2] * 1.003 for c in cs], mode="markers",
                                   marker=dict(size=[2.5 + .45 * c.ev.reward for c in cs], color=[CAT_COLORS[c.ev.category] for c in cs], line=dict(color="white", width=.3)),
                                   text=[f"{c.ev.id} [{c.ev.category}{', RUSH' if c.ev.priority == 'rush' else ''}] reward {c.ev.reward:.1f}, SLA {c.ev.sla_h:.0f} h<br>{c.sat.name} {c.mode.mod} at {hhmm(c.t)} UTC, roll {c.angle:+.0f}°, {c.mode.data_gb:.0f} GB" for c in cs],
                                   hoverinfo="text", name=f"delivered orders ({p})", visible=vis))
        pos = {}                                                                     # satellite positions at imaging time (vectorised per satellite)
        for s, sat in enumerate(fleet):
            idx = [i for i, c in enumerate(cs) if c.sat is sat]
            if idx: P_ = sat.ecef(np.array([cs[i].t for i in idx]))[0]; pos.update({i: P_[j] for j, i in enumerate(idx)})
        lx, ly, lz = seg([(pos[i], c.ev.xyz) for i, c in enumerate(cs)])
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color="rgba(255,255,255,.25)", width=1), hoverinfo="skip", name=f"lines of sight ({p})", visible=vis))
        D["static_idx"][p] = [first, first + 1]
        per = [[] for _ in range(F)]
        for i, c in enumerate(cs):
            f = int(c.t // frame_s)
            for ff in (f, f + 1):
                if ff < F: per[ff].append((pos[i], c.ev.xyz))
        sg = [seg(b) for b in per]; D["ox"][p] = [a[0] for a in sg]; D["oy"][p] = [a[1] for a in sg]; D["oz"][p] = [a[2] for a in sg]
    D["sx"], D["sy"], D["sz"] = [R3(sp[:, f, 0]) for f in range(F)], [R3(sp[:, f, 1]) for f in range(F)], [R3(sp[:, f, 2]) for f in range(F)]
    D["bx"], D["by"], D["bz"] = [b[0] for b in bx], [b[1] for b in bx], [b[2] for b in bx]
    D["labels"] = [f"{hhmm(t)} UTC" + (" (+1d)" if t >= 86400 else "") for t in tf]
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1020", height=900, margin=dict(l=0, r=0, t=50, b=0), uirevision="keep",
                      title="SAR constellation, rolling operations - drag to rotate at ANY time (also while playing) · bottom-left: policy, play/pause, time, speed",
                      scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor="#0b1020", uirevision="keep",
                                 camera=dict(eye=dict(x=1.5, y=1.1, z=0.7))), legend=dict(font=dict(size=10), x=0.99, xanchor="right", y=0.99))
    fig.write_html(path, include_plotlyjs=True, post_script=globe_script(D), config={"scrollZoom": True})
    return D
