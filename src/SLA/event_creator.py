"""
event_creator.py
================
Creates customer TASKS ("events"): WHERE to image, WHAT product is acceptable, HOW MUCH it is worth.
The contractual/timing side (arrival time, SLA, priority) is attached afterwards by `sla_generator_utils`.

Spatial demand is clustered around real-world hotspots (straits, energy regions, flood basins ...) plus a thin uniform background.
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass, field
import numpy as np
from ..earth_sun_orbits.geo import latlon_to_xyz


@dataclass(eq=False)
class Event:
    """A customer task (area of interest)."""
    id: str
    category: str
    position: tuple                       # (lat, lon)
    res_required: list                    # m; any of these resolutions is acceptable
    mod_required: list                    # acceptable product families
    angle_required: list                  # preferred |off-nadir| angles (deg); the further from them, the lower p_obs
    reward: float                         # value if delivered in time
    times_revisited: int = 0              # history of repeat requests (enters the cost model)
    seasonality: dict = field(default_factory=dict)   # month -> reward multiplier
    angle_tol: float = 6.0
    # --- contract / timing, filled in by sla_generator_utils.assign_slas -----------------------------------
    arrival_t: float = 0.0                # when the order reaches the planner (s since t=0). The planner knows nothing before.
    sla_h: float = 24.0                   # SLA: hours from ORDER to DELIVERED product
    priority: str = "standard"            # "standard" | "rush"

    def __post_init__(self):
        self.xyz = latlon_to_xyz(np.array(self.position[0]), np.array(self.position[1]))

    @property
    def deadline_t(self) -> float:
        """Absolute time (s) by which the product must be fully on the ground."""
        return self.arrival_t + self.sla_h * 3600.0

    def season_factor(self, month): return self.seasonality.get(month, 1.0)


# (name, lat, lon, spread_deg, category, weight)
HOTSPOTS = [
    ("Hormuz", 26.5, 56.3, 2.5, "maritime", 3), ("Red Sea", 14.0, 42.5, 3.0, "maritime", 3), ("Suez", 30.5, 32.5, 1.5, "maritime", 1.5),
    ("Black Sea", 44.5, 33.0, 3.0, "maritime", 2), ("Taiwan Strait", 24.5, 119.5, 2.5, "maritime", 2),
    ("South China Sea", 13.0, 114.0, 4.0, "maritime", 2.5), ("Malacca", 3.5, 101.0, 2.5, "maritime", 2),
    ("Gulf of Guinea", 3.0, 3.0, 4.0, "maritime", 1.5), ("North Sea/Baltic", 57.0, 8.0, 4.0, "maritime", 2),
    ("Gulf of Mexico", 27.0, -90.0, 3.5, "maritime", 1.5), ("Mediterranean", 36.0, 17.0, 5.0, "maritime", 2),
    ("Permian", 31.5, -102.0, 3.0, "infrastructure", 2), ("W. Siberia", 62.0, 72.0, 5.0, "infrastructure", 2),
    ("Gulf energy", 27.5, 49.5, 3.0, "infrastructure", 2), ("Pilbara", -22.5, 118.0, 3.0, "infrastructure", 1.5),
    ("Atacama", -24.0, -69.0, 2.5, "infrastructure", 1.5), ("Eastern Ukraine", 48.5, 36.5, 2.5, "infrastructure", 3),
    ("Europe", 48.0, 10.0, 8.0, "infrastructure", 2.5), ("E. China", 32.0, 118.0, 5.0, "infrastructure", 2.5),
    ("Yangtze floods", 29.5, 113.5, 3.0, "disaster", 2), ("Bengal delta", 23.0, 90.0, 2.5, "disaster", 1.5),
    ("Caribbean storms", 22.0, -75.0, 5.0, "disaster", 1.5), ("California fires", 37.0, -121.0, 4.0, "disaster", 1.5),
    ("Indonesia quakes", -3.0, 120.0, 6.0, "disaster", 1.5), ("Amazon", -6.0, -58.0, 6.0, "environment", 3),
    ("US Midwest", 41.5, -94.0, 5.0, "environment", 2), ("Indo-Gangetic", 29.0, 76.0, 4.0, "environment", 2),
    ("Sahel", 13.5, 2.0, 6.0, "environment", 2), ("Congo basin", -2.0, 23.0, 5.0, "environment", 2),
    ("Arctic routes", 73.0, 100.0, 8.0, "environment", 2), ("Greenland", 70.0, -40.0, 6.0, "environment", 1),
    ("Antarctic coast", -70.0, 20.0, 6.0, "environment", 1),
]

# What each customer category typically orders: acceptable product families, resolutions, reward range
CATEGORY_PROFILE = {
    "maritime":       dict(mods=["spotlight", "stripmap"], res=[1.0, 3.0], reward=(4, 9)),
    "infrastructure": dict(mods=["dwell", "spotlight"], res=[0.5, 1.0], reward=(2, 7)),
    "disaster":       dict(mods=["stripmap", "scan"], res=[3.0, 15.0], reward=(7, 14)),
    "environment":    dict(mods=["scan", "stripmap"], res=[3.0, 15.0, 20.0], reward=(1, 4)),
}


def make_events(n: int = 3000, seed: int = 7, month: int = 11) -> list[Event]:

    """Create n tasks (location + product requirements + reward). No timing yet: see sla_generator_utils."""
    rng = random.Random(seed)
    weights = [h[5] for h in HOTSPOTS]
    evs = []
    for i in range(n):
        if rng.random() < 0.12:                                     # uniform background demand
            lat, lon = math.degrees(math.asin(rng.uniform(-0.93, 0.93))), rng.uniform(-180, 180)
            cat = rng.choice(list(CATEGORY_PROFILE))
        else:                                                       # clustered demand around a hotspot
            _, la, lo, sp, cat, _w = rng.choices(HOTSPOTS, weights)[0]
            lat = max(-80, min(80, rng.gauss(la, sp)))
            lon = (rng.gauss(lo, sp / max(0.2, math.cos(math.radians(la)))) + 180) % 360 - 180
        c = CATEGORY_PROFILE[cat]
        mods = rng.sample(c["mods"], rng.choice([1, 2]) if len(c["mods"]) > 1 else 1)
        evs.append(Event(id=f"T{i:04d}", category=cat, position=(lat, lon),
                         res_required=sorted(rng.sample(c["res"], rng.choice([1, 2]))), mod_required=mods,
                         angle_required=[rng.uniform(20, 50) for _ in range(rng.choice([1, 2]))],
                         reward=rng.uniform(*c["reward"]), times_revisited=rng.randint(0, 6),
                         seasonality={month: rng.choice([1.0, 1.0, 1.0, 1.4, 0.6])}, angle_tol=rng.choice([4.0, 6.0, 10.0])))
    return evs
