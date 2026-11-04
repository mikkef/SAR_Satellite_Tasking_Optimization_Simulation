"""Earth, Sun and coordinate helpers (spherical Earth, circular orbits)."""
import math
import numpy as np

RE = 6371.0                 # km
MU = 398600.4418            # km^3/s^2
W_E = 7.2921159e-5          # rad/s

# Sun on 2 Nov 2026 (ecliptic longitude ~219.6 deg). Simulation t=0 is ~00:00 UTC.
EPS = math.radians(23.439)
SUN_LON = math.radians(219.6)
SUN_ECI = np.array([math.cos(SUN_LON), math.sin(SUN_LON) * math.cos(EPS), math.sin(SUN_LON) * math.sin(EPS)])
SUN_RA = math.atan2(math.sin(SUN_LON) * math.cos(EPS), math.cos(SUN_LON))
THETA0 = SUN_RA - math.pi   # Earth rotation angle at t=0 so Greenwich is at local solar midnight


def raan_from_ltan(ltan_hours: float) -> float:
    """RAAN (deg) of a sun-synchronous plane with the given local time of ascending node."""
    return math.degrees(SUN_RA + math.radians(15.0 * (ltan_hours - 12.0))) % 360.0


def latlon_to_xyz(lat, lon, r=RE):
    lat, lon = np.radians(lat), np.radians(lon)
    return r * np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat) * np.ones_like(lon)], -1)


def xyz_to_latlon(p):
    p = np.asarray(p)
    return np.degrees(np.arcsin(p[..., 2] / np.linalg.norm(p, axis=-1))), np.degrees(np.arctan2(p[..., 1], p[..., 0]))


def eclipse_mask(r_eci: np.ndarray) -> np.ndarray:
    """True where the satellite is in Earth's cylindrical shadow."""
    along = r_eci @ SUN_ECI
    perp = np.linalg.norm(r_eci - along[:, None] * SUN_ECI[None], axis=1)
    return (along < 0) & (perp < RE)


def hhmm(t_seconds: float) -> str:
    return f"{int(t_seconds // 3600) % 24:02d}:{int(t_seconds % 3600 // 60):02d}"
