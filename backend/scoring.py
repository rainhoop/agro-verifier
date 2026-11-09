"""Heuristic application-match score and transparent verdict rules.

The score is intentionally simple:
    shape 50% + sowing-date alignment 25% + peak level 25%.
It measures agreement with the *declared* application, not whether the crop
is actually correct.  Crop identity and double-wave anomalies are separate
verdict guards based on the ML model and a shape check.  All thresholds live
in ``backend.config`` and are exposed by ``GET /api/config``.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import config
from .features import CROP_IDS, iso_to_date

DATA_DIR = Path(__file__).resolve().parent / "data"
PROFILES_PATH = DATA_DIR / "profiles.json"

_profiles_cache: dict | None = None


def load_profiles(force: bool = False) -> dict:
    global _profiles_cache
    if _profiles_cache is None or force:
        with PROFILES_PATH.open("r", encoding="utf-8") as fh:
            _profiles_cache = json.load(fh)
    return _profiles_cache


def profile_values(crop: str) -> list[float] | None:
    profiles = load_profiles()
    crop_data = profiles.get("crops", {}).get(crop)
    if not crop_data:
        return None
    return [float(item["ndvi"]) for item in crop_data.get("series", [])]


def reference_series(crop: str, dates: list[date]) -> list[dict]:
    profiles = load_profiles()
    crop_data = profiles.get("crops", {}).get(crop)
    if not crop_data:
        return []
    items = crop_data.get("series", [])
    result = []
    for offset, item in enumerate(items):
        if offset >= len(dates):
            break
        result.append({"date": dates[offset].isoformat(), "ndvi": round(float(item["ndvi"]), 3)})
    return result


def reference_peak(crop: str) -> float:
    values = profile_values(crop)
    return max(values) if values else 0.75


def calculate_score(
    features: dict, reference_peak_value: float, sowing_date: date
) -> tuple[int, dict]:
    shape = max(0.0, min(1.0, float(features.get("correlation_with_reference") or 0.0)))
    growth = iso_to_date(features.get("growth_start_date"))
    if growth is None:
        timing = 0.0
    else:
        diff = abs((growth - sowing_date).days)
        if diff <= config.TIMING_FULL_DAYS:
            timing = 1.0
        else:
            timing = max(
                0.0,
                1.0
                - (diff - config.TIMING_FULL_DAYS)
                / max(1.0, config.TIMING_ZERO_DAYS - config.TIMING_FULL_DAYS),
            )
    peak = float(features.get("peak_ndvi") or 0.0)
    peak_ratio = max(0.0, min(1.0, peak / reference_peak_value)) if reference_peak_value else 0.0

    raw = (
        config.SCORE_WEIGHTS["shape"] * shape
        + config.SCORE_WEIGHTS["timing"] * timing
        + config.SCORE_WEIGHTS["peak"] * peak_ratio
    )
    score = int(round(raw * 100))
    breakdown = {
        "shape": round(50.0 * shape, 1),
        "timing": round(25.0 * timing, 1),
        "peak": round(25.0 * peak_ratio, 1),
        "peak_ratio": round(peak_ratio, 3),
        "timing_diff_days": (
            abs((growth - sowing_date).days) if growth is not None else None
        ),
    }
    return max(0, min(100, score)), breakdown


def decide_verdict(
    features: dict,
    score: int,
    ml_result: dict,
    declared_crop: str,
    reference_peak_value: float | None = None,
) -> tuple[str, str]:
    """Return (verdict, reason_code).

    Order matters.  A good shape score cannot overrule a confident crop
    mismatch, a double wave, or an almost flat canopy.
    """
    peak = float(features.get("peak_ndvi") or 0.0)
    growth = features.get("growth_start_date")
    vegetation_days = int(features.get("vegetation_days") or 0)
    reference_peak = reference_peak_value or 0.0
    peak_ratio = peak / reference_peak if reference_peak > 0 else 0.0

    # 1. No meaningful growth at all.
    if peak < config.NO_GROWTH_PEAK and (growth is None or vegetation_days < 20):
        return "not_found", "no_growth"

    predicted = ml_result.get("predicted")
    probability = float(ml_result.get("probability") or 0.0)

    # 2. A clearly bimodal curve needs a human.  The transparent shape check
    # is primary; a model-only "double_wave" label is trusted only when the
    # classifier is reasonably confident.  This avoids turning a weak model
    # guess on real Sentinel-2 data into a hard verdict.
    if config.DOUBLE_WAVE_GUARD and (
        features.get("two_wave_detected")
        or (predicted == "double_wave" and probability >= config.CROP_MISMATCH_PROBABILITY)
    ):
        return "needs_check", "double_wave"

    # 3. Confident crop mismatch.  "bare" and "double_wave" are anomalies,
    # not alternative crops, so they never trigger this rule.
    if (
        predicted in CROP_IDS
        and predicted != declared_crop
        and probability >= config.CROP_MISMATCH_PROBABILITY
    ):
        return "crop_mismatch", "crop_mismatch"

    # 4. Almost flat canopy: usually fallow, pasture or a failed field.
    if peak <= config.FLAT_PEAK_MAX:
        return "not_found", "flat_canopy"

    # 5. A weak peak cannot be confirmed even when the remaining curve has a
    # familiar shape.  This is the guard that keeps a high shape score from
    # masking a low canopy.
    if reference_peak > 0 and peak_ratio < config.LOW_PEAK_RATIO:
        return "doubtful", "low_peak"

    # 6. Score thresholds.
    if score >= config.CONFIRMED_SCORE:
        return "confirmed", "score_confirmed"
    if score >= config.DOUBTFUL_SCORE:
        return "doubtful", "score_doubtful"
    return "not_found", "score_low"
