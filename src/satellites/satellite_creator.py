"""
satellite_creator.py
====================
Everything needed to CREATE satellites: imaging products (modes), hardware generations and the fleet itself.

A `Satellite` object is the *static specification + orbit* of a spacecraft. Anything that changes during operations
(battery charge, data on board, fuel, outages ...) lives in `satellite_update_utils.SatState`, not here.

All numbers are plausible engineering assumptions, not proprietary data.
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass, field
import numpy as np
from ..earth_sun_orbits.geo import RE, MU, W_E, THETA0, raan_from_ltan, xyz_to_latlon


# --------------------------------------------------------------------------- imaging products
@dataclass(frozen=True)
class Mode:
    """An imaging product. Products differ in resolution, acquisition time, DATA VOLUME and POWER - which is why
    storage and energy constraints make the choice of product matter."""
    mod: str            # family: dwell | spotlight | stripmap | scan
    res: float          # best achievable resolution, m
    dwell: float        # acquisition time, s
    data_gb: float      # data volume produced on board, GB
    power_w: float      # extra electrical power drawn while imaging, W


# Economic cost factor per product family (higher = more expensive to produce / process)
MOD_COST = {"dwell": 5.0, "spotlight": 3.0, "stripmap": 1.5, "scan": 1.0}

DWELL = Mode("dwell", 0.25, 25, 30.0, 1300)
SPOT = Mode("spotlight", 0.5, 12, 14.0, 1000)
STRIP = Mode("stripmap", 3.0, 20, 22.0, 850)
SCAN = Mode("scan", 15.0, 30, 16.0, 750)


# --------------------------------------------------------------------------- satellite specification
@dataclass(eq=False)
class Satellite:
    """A satellite's fixed design and orbit; its live battery and data state is stored separately."""
    name: str
    plane: str
    gen: str
    # orbit (circular)
    altitude_km: float
    inclination_deg: float
    raan_deg: float
    phase_deg: float
    products: list
    # pointing / agility
    min_angle: float = 15.0              # off-nadir limits (deg)
    max_angle: float = 55.0
    slew_s_per_deg: float = 0.5          # time to rotate one degree of roll
    settle_s: float = 2.0                # settling time after each slew
    current_angle: float = 0.0           # signed roll at t=0
    # data subsystem
    storage_gb: float = 150.0
    backlog_gb: float = 20.0             # un-downlinked data already on board at t=0
    dl_gb_s: float = 0.10                # downlink rate, GB/s
    # electrical power subsystem
    battery_wh: float = 100.0
    soc_min_wh: float = 45.0             # never discharge below this (depth-of-discharge limit)
    soc_init_wh: float = 90.0
    solar_w: float = 130.0               # orbit-averaged array output in sunlight
    bus_w: float = 55.0                  # always-on housekeeping load
    dl_w: float = 70.0                   # extra load while the downlink radio is on
    slew_wh_per_deg: float = 0.02
    # propulsion (orbit maintenance only - SAR tasking itself uses reaction wheels, not propellant)
    propellant_kg: float = 2.0
    fuel_reserve_kg: float = 0.3         # below this the satellite needs operator attention

    def __post_init__(self):
        # Derive orbit radius, speed, and repeat time from the satellite's altitude.
        self.a = RE + self.altitude_km
        self.n = math.sqrt(MU / self.a ** 3)             # mean motion, rad/s
        self.period = 2 * math.pi / self.n

    # ---- orbit propagation -------------------------------------------------------------
    def eci(self, t):
        """Position / velocity in the inertial frame (km, km/s) for a time array t (s since t=0)."""
        # Move around a circular orbit, then rotate the orbital plane into 3D space.
        t = np.atleast_1d(np.asarray(t, float))
        u = math.radians(self.phase_deg) + self.n * t
        i, om = math.radians(self.inclination_deg), math.radians(self.raan_deg)
        x, y = self.a * np.cos(u), self.a * np.sin(u)
        vx, vy = -self.a * self.n * np.sin(u), self.a * self.n * np.cos(u)
        def lift(x, y):                                  # orbital plane -> inertial frame (inclination, then RAAN)
            y2, z2 = y * math.cos(i), y * math.sin(i)
            return np.stack([x * math.cos(om) - y2 * math.sin(om), x * math.sin(om) + y2 * math.cos(om), z2], -1)
        return lift(x, y), lift(vx, vy)

    def ecef(self, t):
        """Position / velocity in the Earth-fixed frame (the Earth rotates under the orbit)."""
        # Rotate the position with Earth so it can be compared with ground locations.
        t = np.atleast_1d(np.asarray(t, float))
        r, v = self.eci(t)
        th = THETA0 + W_E * t
        c, s = np.cos(th), np.sin(th)
        rot = lambda q: np.stack([c * q[:, 0] + s * q[:, 1], -s * q[:, 0] + c * q[:, 1], q[:, 2]], -1)
        return rot(r), rot(v)

    def latlon(self, t): return xyz_to_latlon(self.ecef(t)[0])

    # ---- derived quantities ---------------------------------------------------------------
    def max_central_deg(self) -> float:
        """Largest Earth-central angle between the sub-satellite point and a target that is still inside the off-nadir limit."""
        eta = math.radians(self.max_angle)
        return math.degrees(math.asin(min(1.0, self.a / RE * math.sin(eta))) - eta)

    def energy_wh(self, mode: Mode, roll: float) -> float:
        """Energy of one observation: transmitter/processing during the dwell plus the slew out to the roll angle."""
        # Combine imaging power over time with the energy used to change the viewing angle.
        return mode.power_w * mode.dwell / 3600.0 + self.slew_wh_per_deg * abs(roll)

    def slew_need(self, a1: float, a2: float) -> float:
        """Seconds needed between the end of one observation and the start of the next one."""
        return abs(a2 - a1) * self.slew_s_per_deg + self.settle_s

    @property
    def drag_kg_per_day(self) -> float:
        """Propellant used for drag make-up: grows quickly as the orbit gets lower."""
        return 0.004 * math.exp(-(self.altitude_km - 560.0) / 70.0)


# --------------------------------------------------------------------------- hardware generations
GEN2 = dict(gen="G2", products=[SPOT, STRIP, SCAN], storage_gb=150, dl_gb_s=0.06, battery_wh=110, soc_min_wh=45,
            soc_init_wh=100, solar_w=130, bus_w=55, dl_w=70, slew_s_per_deg=0.5, propellant_kg=2.0)
GEN3 = dict(gen="G3", products=[DWELL, SPOT, STRIP, SCAN], storage_gb=300, dl_gb_s=0.12, battery_wh=160, soc_min_wh=70,
            soc_init_wh=145, solar_w=220, bus_w=70, dl_w=90, slew_s_per_deg=0.25, max_angle=58.0, propellant_kg=3.5)


def make_fleet(seed: int = 3) -> list[Satellite]:
    """24 satellites in 4 planes: three sun-synchronous planes (10:30, 22:30 and dawn-dusk local time of ascending node)
    and one 45-degree plane that mixes both hardware generations. Satellites in a plane are evenly phased."""
    rng = random.Random(seed)
    # Each plane has its own orbit and equipment mix; satellites are spaced evenly around it.
    planes = [  # name, LTAN (h) or None, inclination, altitude, generation(s), count
        ("A", 10.5, 97.7, 560, [GEN2], 6),
        ("B", 22.5, 97.7, 580, [GEN2], 6),
        ("C", 6.0, 97.8, 600, [GEN3], 6),      # dawn-dusk: permanent sunlight -> no eclipse, energy-rich
        ("D", None, 45.0, 570, [GEN2, GEN3], 6),
    ]
    fleet = []
    for pi, (pl, ltan, inc, alt, gens, n) in enumerate(planes):
        raan = raan_from_ltan(ltan) if ltan is not None else 40.0
        for k in range(n):
            g = dict(gens[k % len(gens)])
            fleet.append(Satellite(name=f"VX-{pl}{k + 1}", plane=pl, altitude_km=alt, inclination_deg=inc, raan_deg=raan,
                                   phase_deg=(360.0 / n) * k + 17.0 * pi,
                                   backlog_gb=rng.uniform(0.05, 0.25) * g["storage_gb"], **g))
    return fleet
