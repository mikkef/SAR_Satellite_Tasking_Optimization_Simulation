"""
ground_station_creator.py
=========================
Creates the ground segment: a set of downlink stations, each with a number of antennas and a data rate.
(The illustrative sites below are typical polar + mid-latitude commercial station locations, not a statement about any real contract.)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..earth_sun_orbits.geo import latlon_to_xyz


@dataclass(eq=False)
class GroundStation:
    name: str
    lat: float
    lon: float
    antennas: int = 1            # an antenna serves ONE satellite at a time -> contention
    rate_gb_s: float = 0.25      # station-side max rate; the link uses min(satellite rate, station rate)
    min_elev: float = 5.0        # elevation mask, deg

    def __post_init__(self): self.xyz = latlon_to_xyz(np.array(self.lat), np.array(self.lon))


def make_stations() -> list[GroundStation]:
    """12 stations. Polar sites (Svalbard, Kiruna, Fairbanks, Inuvik, Troll) see a sun-synchronous satellite on almost every orbit."""
    S = GroundStation
    return [S("Svalbard", 78.23, 15.39, antennas=2), S("Kiruna", 67.86, 20.96), S("Fairbanks", 64.84, -147.72),
            S("Inuvik", 68.36, -133.72), S("Awarua", -46.53, 168.38), S("Troll", -72.0, 2.53), S("PuntaArenas", -53.0, -70.9),
            S("Hawaii", 19.0, -155.7), S("Singapore", 1.35, 103.8), S("Oman", 23.6, 58.5), S("Hartebeesthoek", -25.89, 27.69),
            S("Dongara", -29.2, 114.9)]
