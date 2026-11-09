"""Synthetic NDVI generator with realistic imperfections.

The generator is used only for (a) the ML training set and (b) the explicitly
labelled fallback when Sentinel-2 is unavailable.  Real demo fields come from
``data/real_fields.json``; they are never overwritten by this module.

The curves are deliberately not identical to the reference profile: each
sample gets inter-annual timing, amplitude and canopy-shape variation, sensor
noise and cloud gaps.  This prevents suspiciously perfect 100/100 matches.
"""
from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta
from typing import Sequence

import numpy as np

from . import config
from .features import CROP_IDS, STEP_DAYS, interpolate_series, season_dates

# Literature-informed phenology for northern Kazakhstan.  These are broad
# priors, not measurements from named farms: spring wheat is usually sown in
# mid-May and peaks in the first half of July; barley is earlier; sunflower is
# the latest crop; rapeseed peaks around late June/early July.
CROP_PROFILES: dict[str, dict] = {
    "wheat": {
        "sowing": date(2026, 5, 15),
        "peak": date(2026, 7, 12),
        "peak_ndvi": 0.74,
        "baseline": 0.13,
        "sigma_left": 34.0,
        "sigma_right": 26.0,
        "peak_offset_days": 58,
        "harvest": "конец августа",
    },
    "barley": {
        "sowing": date(2026, 5, 5),
        "peak": date(2026, 6, 28),
        "peak_ndvi": 0.69,
        "baseline": 0.12,
        "sigma_left": 29.0,
        "sigma_right": 20.0,
        "peak_offset_days": 54,
        "harvest": "начало августа",
    },
    "sunflower": {
        "sowing": date(2026, 5, 20),
        "peak": date(2026, 8, 3),
        "peak_ndvi": 0.80,
        "baseline": 0.13,
        "sigma_left": 38.0,
        "sigma_right": 28.0,
        "peak_offset_days": 75,
        "harvest": "середина сентября",
    },
    "rapso": {
        "sowing": date(2026, 5, 1),
        "peak": date(2026, 6, 30),
        "peak_ndvi": 0.77,
        "baseline": 0.12,
        "sigma_left": 30.0,
        "sigma_right": 21.0,
        "peak_offset_days": 60,
        "harvest": "начало августа",
    },
}

FAR_CROP = {
    "wheat": "sunflower",
    "barley": "rapso",
    "sunflower": "wheat",
    "rapso": "wheat",
}


def asymmetric_gaussian(
    dates: Sequence[date],
    peak_date: date,
    peak_ndvi: float,
    baseline: float = 0.12,
    sigma_left: float = 32.0,
    sigma_right: float = 23.0,
) -> np.ndarray:
    offsets = np.array([(d - peak_date).days for d in dates], dtype=float)
    sigma = np.where(offsets < 0, sigma_left, sigma_right)
    values = baseline + (peak_ndvi - baseline) * np.exp(-0.5 * (offsets / sigma) ** 2)
    return np.clip(values, 0.0, 1.0)


def profile_values(crop: str) -> np.ndarray:
    p = CROP_PROFILES[crop]
    return asymmetric_gaussian(
        season_dates(config.SEASON_YEAR),
        p["peak"],
        p["peak_ndvi"],
        p["baseline"],
        p["sigma_left"],
        p["sigma_right"],
    )


def _smooth_perturbation(rng: random.Random, size: int, amplitude: float) -> np.ndarray:
    """Low-frequency canopy-shape variation (management, soil, uneven stand)."""
    if size < 4 or amplitude <= 0:
        return np.zeros(size)
    knots = np.array([rng.uniform(-1.0, 1.0) for _ in range(4)], dtype=float)
    x_knots = np.linspace(0.0, size - 1, len(knots))
    x = np.arange(size)
    return np.interp(x, x_knots, knots) * amplitude


def _with_gaps_and_noise(
    values: np.ndarray,
    rng: random.Random,
    noise: float,
    cloud_gaps: bool = True,
    gap_weeks: int | None = None,
) -> tuple[np.ndarray, list[bool], list[float]]:
    arr = np.asarray(values, dtype=float)
    arr = arr + np.array([rng.gauss(0.0, noise) for _ in range(arr.size)])
    arr += _smooth_perturbation(rng, arr.size, noise * 1.6)
    measured = [True] * arr.size
    fractions = [round(rng.uniform(0.72, 1.0), 2) for _ in range(arr.size)]
    if cloud_gaps and arr.size > 5:
        gap_count = gap_weeks if gap_weeks is not None else rng.choice([0, 1, 1, 2, 2, 3])
        if gap_count:
            start = rng.randint(2, max(2, arr.size - gap_count - 1))
            for idx in range(start, min(arr.size, start + gap_count)):
                arr[idx] = np.nan
                measured[idx] = False
                fractions[idx] = round(rng.uniform(0.0, 0.28), 2)
    filled = interpolate_series(arr)
    return np.clip(filled, 0.03, 0.95), measured, fractions


def generate_crop_sample(
    crop: str,
    rng: random.Random,
    shift_days: int | None = None,
    stress: str | None = None,
) -> dict:
    p = CROP_PROFILES[crop]
    if shift_days is None:
        shift_days = rng.randint(-10, 10)
    peak_date = p["peak"] + timedelta(days=shift_days)
    amplitude = p["peak_ndvi"] + rng.uniform(-0.10, 0.06)
    sigma_left = p["sigma_left"] * rng.uniform(0.86, 1.16)
    sigma_right = p["sigma_right"] * rng.uniform(0.86, 1.15)
    baseline = p["baseline"] + rng.uniform(-0.03, 0.03)
    if stress == "late_low":
        peak_date = p["peak"] + timedelta(days=27 + rng.randint(-5, 8))
        amplitude = p["peak_ndvi"] - rng.uniform(0.18, 0.30)
        sigma_left *= rng.uniform(0.82, 0.98)
        sigma_right *= rng.uniform(0.80, 0.96)
    elif stress == "early_stress":
        peak_date = p["peak"] - timedelta(days=13 + rng.randint(0, 9))
        amplitude = p["peak_ndvi"] - rng.uniform(0.10, 0.22)
        sigma_left *= rng.uniform(0.88, 1.04)
    amplitude = min(0.90, max(0.30, amplitude))
    values = asymmetric_gaussian(
        season_dates(config.SEASON_YEAR),
        peak_date,
        amplitude,
        baseline,
        sigma_left,
        sigma_right,
    )
    values, measured, fractions = _with_gaps_and_noise(
        values, rng, 0.032 if stress is None else 0.038, cloud_gaps=True
    )
    return {
        "values": values.tolist(),
        "measured": measured,
        "valid_fraction": fractions,
        "sowing_date": (p["sowing"] + timedelta(days=shift_days)).isoformat(),
        "peak_date": peak_date.isoformat(),
        "stress": stress,
    }


def generate_bare_sample(rng: random.Random) -> dict:
    baseline = rng.uniform(0.07, 0.20)
    trend = rng.uniform(-0.03, 0.03)
    values = np.array(
        [baseline + trend * i / 20.0 + rng.gauss(0, 0.025) for i in range(config.N_POINTS)],
        dtype=float,
    )
    values = np.clip(values, 0.04, 0.30)
    filled, measured, fractions = _with_gaps_and_noise(
        values, rng, 0.012, cloud_gaps=True
    )
    return {"values": filled.tolist(), "measured": measured, "valid_fraction": fractions}


def generate_two_wave_sample(rng: random.Random) -> dict:
    dates = season_dates(config.SEASON_YEAR)
    first_peak = date(2026, 6, 1) + timedelta(days=rng.randint(0, 28))
    second_peak = date(2026, 7, 28) + timedelta(days=rng.randint(0, 28))
    if second_peak <= first_peak + timedelta(days=30):
        second_peak = first_peak + timedelta(days=44)
    baseline = rng.uniform(0.09, 0.14)
    first_amp = rng.uniform(0.34, 0.55)
    second_amp = rng.uniform(0.42, 0.68)
    left_sigma = rng.uniform(14.0, 22.0)
    right_sigma = rng.uniform(14.0, 22.0)
    first = asymmetric_gaussian(
        dates, first_peak, baseline + first_amp, baseline, left_sigma, left_sigma
    )
    second = asymmetric_gaussian(
        dates, second_peak, baseline + second_amp, baseline, right_sigma, right_sigma
    )
    values = np.clip(np.maximum(first, second), 0.05, 0.90)
    filled, measured, fractions = _with_gaps_and_noise(
        values, rng, 0.028, cloud_gaps=True
    )
    return {"values": filled.tolist(), "measured": measured, "valid_fraction": fractions}


def center_seed(lat: float, lon: float) -> int:
    raw = f"{lat:.4f}|{lon:.4f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


SCENARIO_WEIGHTS = [
    ("normal", 46),
    ("late_low", 14),
    ("bare", 10),
    ("wrong_crop", 12),
    ("double_wave", 10),
    ("early_stress", 8),
]


def _pick_scenario(rng: random.Random) -> str:
    total = sum(weight for _, weight in SCENARIO_WEIGHTS)
    point = rng.uniform(0, total)
    upto = 0.0
    for name, weight in SCENARIO_WEIGHTS:
        upto += weight
        if point <= upto:
            return name
    return "normal"


def synthetic_series_for_field(
    lat: float,
    lon: float,
    declared_crop: str,
    sowing_date: date,
    scenario: str | None = None,
    year: int | None = None,
) -> dict:
    """Deterministic synthetic NDVI for a drawn polygon.

    The scenario is derived from the centre coordinates.  The declared crop
    and sowing date anchor the curve; changing them does not change the seed
    but does change the plausible agronomic timing.
    """
    declared_crop = declared_crop if declared_crop in CROP_IDS else "wheat"
    year = year or sowing_date.year
    dates = season_dates(year)
    seed = center_seed(lat, lon)
    rng = random.Random(seed)
    chosen = scenario or _pick_scenario(rng)
    base = CROP_PROFILES[declared_crop]
    actual_crop = declared_crop
    measured: list[bool]
    fractions: list[float]

    if chosen == "bare":
        sample = generate_bare_sample(rng)
        values, measured, fractions = sample["values"], sample["measured"], sample["valid_fraction"]
    elif chosen == "double_wave":
        baseline = rng.uniform(0.09, 0.14)
        p1 = sowing_date + timedelta(days=rng.randint(30, 42))
        p2 = sowing_date + timedelta(days=rng.randint(80, 98))
        a1 = rng.uniform(0.34, 0.50)
        a2 = rng.uniform(0.46, 0.64)
        s1 = rng.uniform(14, 21)
        s2 = rng.uniform(15, 22)
        first = asymmetric_gaussian(dates, p1, baseline + a1, baseline, s1, s1)
        second = asymmetric_gaussian(dates, p2, baseline + a2, baseline, s2, s2)
        clean = np.clip(np.maximum(first, second), 0.05, 0.90)
        values, measured, fractions = _with_gaps_and_noise(
            clean, rng, 0.027, cloud_gaps=True
        )
    elif chosen == "wrong_crop":
        actual_crop = FAR_CROP[declared_crop]
        actual = CROP_PROFILES[actual_crop]
        base_peak = date(year, actual["peak"].month, actual["peak"].day)
        peak_date = base_peak + timedelta(days=rng.randint(-7, 7))
        amplitude = actual["peak_ndvi"] + rng.uniform(-0.08, 0.04)
        clean = asymmetric_gaussian(
            dates,
            peak_date,
            amplitude,
            actual["baseline"],
            actual["sigma_left"] * rng.uniform(0.90, 1.10),
            actual["sigma_right"] * rng.uniform(0.90, 1.10),
        )
        values, measured, fractions = _with_gaps_and_noise(
            clean, rng, 0.030, cloud_gaps=True
        )
    else:
        peak_date = sowing_date + timedelta(
            days=base["peak_offset_days"] + rng.randint(-7, 7)
        )
        amplitude = base["peak_ndvi"] + rng.uniform(-0.07, 0.04)
        sigma_left = base["sigma_left"] * rng.uniform(0.90, 1.12)
        sigma_right = base["sigma_right"] * rng.uniform(0.88, 1.12)
        if chosen == "late_low":
            peak_date += timedelta(days=28 + rng.randint(-4, 7))
            amplitude = base["peak_ndvi"] - rng.uniform(0.20, 0.29)
            sigma_left *= 0.90
            sigma_right *= 0.85
        elif chosen == "early_stress":
            peak_date -= timedelta(days=14 + rng.randint(0, 9))
            amplitude = base["peak_ndvi"] - rng.uniform(0.11, 0.21)
        clean = asymmetric_gaussian(
            dates, peak_date, amplitude, base["baseline"], sigma_left, sigma_right
        )
        values, measured, fractions = _with_gaps_and_noise(
            clean, rng, 0.028 if chosen == "normal" else 0.034, cloud_gaps=True
        )

    values = np.clip(values, 0.02, 0.95)
    return {
        "dates": dates,
        "values": [round(float(v), 3) for v in values],
        "measured": measured,
        "valid_fraction": fractions,
        "scenario": chosen,
        "actual_crop": actual_crop,
        "seed": seed,
    }
