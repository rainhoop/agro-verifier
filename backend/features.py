"""Feature extraction from a 10-day NDVI time series.

The module is deliberately dependency-light (numpy only) so it can also be
used by the data generator and by unit tests.  Dates follow the real Sentinel-2
aggregation grid: 1 April through 21 October 2026 (21 ten-day composites).
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from . import config

SEASON_YEAR = config.SEASON_YEAR
SEASON_START = date(config.SEASON_YEAR, config.SEASON_START[0], config.SEASON_START[1])
STEP_DAYS = config.STEP_DAYS
N_POINTS = config.N_POINTS

CROP_IDS = ("wheat", "barley", "sunflower", "rapso")
CLASS_IDS = CROP_IDS + ("bare", "double_wave")

TEMPORAL_NDVI_COLORS = (
    "#E9E1BF",
    "#D8D79A",
    "#BCD07F",
    "#8FBE6C",
    "#5FA462",
    "#3A8660",
    "#1F6F5C",
)
TEMPORAL_DEVIATION_COLORS = (
    "#596267",
    "#777E82",
    "#969A9C",
    "#F1F0EC",
    "#969A9C",
    "#777E82",
    "#596267",
)
GROWTH_NDVI_THRESHOLD = 0.25
PEAK_NDVI_FRACTION = 0.9
FLAT_CURVE_RANGE = 0.03

CROP_LABELS: dict[str, dict[str, str]] = {
    "wheat": {"ru": "Пшеница яровая", "kz": "Жаздық бидай"},
    "barley": {"ru": "Ячмень", "kz": "Арпа"},
    "sunflower": {"ru": "Подсолнечник", "kz": "Күнбағыс"},
    "rapso": {"ru": "Рапс", "kz": "Рапс"},
    "bare": {"ru": "Нет посева / пустое поле", "kz": "Егіс жоқ / бос алқап"},
    "double_wave": {"ru": "Две волны роста", "kz": "Екі өсу толқыны"},
}

CROP_ALIASES: dict[str, str] = {
    "wheat": "wheat",
    "пшеница": "wheat",
    "пшеница яровая": "wheat",
    "бидай": "wheat",
    "barley": "barley",
    "ячмень": "barley",
    "арпа": "barley",
    "sunflower": "sunflower",
    "подсолнечник": "sunflower",
    "күнбағыс": "sunflower",
    "rapso": "rapso",
    "rape": "rapso",
    "rapeseed": "rapso",
    "рапс": "rapso",
}


def normalize_crop(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("ё", "е")
    return CROP_ALIASES.get(key)


def crop_label(crop: str, lang: str = "ru") -> str:
    labels = CROP_LABELS.get(crop, {"ru": crop, "kz": crop})
    return labels.get("kz" if lang == "kz" else "ru", crop)


def season_dates(year: int = SEASON_YEAR) -> list[date]:
    start = date(year, config.SEASON_START[0], config.SEASON_START[1])
    return [start + timedelta(days=i * STEP_DAYS) for i in range(N_POINTS)]


def temporal_stage(
    dates: Sequence[date],
    values: Sequence[float | None],
    index: int,
    extracted: dict,
    measured: Sequence[bool] | None = None,
) -> str | None:
    """Classify a composite from the already extracted seasonal curve boundaries."""
    length = min(len(dates), len(values))
    if index < 0 or index >= length:
        return None

    valid = [
        float(value)
        for offset, value in enumerate(values[:length])
        if value is not None
        and np.isfinite(float(value))
        and (measured is None or (offset < len(measured) and measured[offset]))
    ]
    if len(valid) < 3 or max(valid) - min(valid) < FLAT_CURVE_RANGE:
        return None

    current = dates[index]
    try:
        current_value = float(values[index]) if values[index] is not None else None
    except (TypeError, ValueError):
        current_value = None
    if current_value is None or not np.isfinite(current_value):
        return None

    growth = date.fromisoformat(extracted["growth_start_date"]) if extracted.get("growth_start_date") else None
    peak = date.fromisoformat(extracted["peak_date"]) if extracted.get("peak_date") else None
    peak_value = extracted.get("peak_ndvi")
    if peak_value is None:
        peak_value = max(valid)
    peak_value = float(peak_value)

    if current_value < GROWTH_NDVI_THRESHOLD:
        if peak is not None and current > peak:
            return "after_harvest"
        return "before_growth"
    if growth is None or current < growth:
        return "before_growth"
    if (
        peak is not None
        and abs((current - peak).days) <= STEP_DAYS
        and current_value >= peak_value * PEAK_NDVI_FRACTION
    ):
        return "peak"
    if peak is None or current < peak:
        return "growth"
    return "decline"


def temporal_color(value: float | None, mode: str = "ndvi") -> str:
    """Pick a seven-step map color; deviation uses magnitude, not verdict colors."""
    if value is None:
        return "#D8DADD"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "#D8DADD"
    if not np.isfinite(number):
        return "#D8DADD"

    if mode == "deviation":
        position = (min(max(number, -0.5), 0.5) + 0.5) / 1.0
        palette = TEMPORAL_DEVIATION_COLORS
    else:
        position = min(max(number, 0.0), 1.0)
        palette = TEMPORAL_NDVI_COLORS
    index = min(round(position * (len(palette) - 1)), len(palette) - 1)
    return palette[index]


def interpolate_series(values: Sequence[float | None]) -> np.ndarray:
    """Linearly fill cloud gaps and clip NDVI to a physically sane range."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, 0.0, 1.0)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not finite.any():
        return np.full_like(arr, 0.0)
    idx = np.arange(arr.size)
    arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return arr


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return 0.0
    a, b = a[mask], b[mask]
    if float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(value):
        return 0.0
    return value


def _cross_up(dates: Sequence[date], values: np.ndarray, threshold: float) -> date | None:
    if values.size == 0:
        return None
    if values[0] >= threshold:
        return dates[0]
    for i in range(1, values.size):
        if values[i - 1] < threshold <= values[i]:
            span = float(values[i] - values[i - 1])
            frac = (threshold - float(values[i - 1])) / span if span else 0.0
            offset = (dates[i] - dates[i - 1]).days * max(0.0, min(1.0, frac))
            return dates[i - 1] + timedelta(days=round(offset))
    return None


def _cross_down(
    dates: Sequence[date], values: np.ndarray, peak_idx: int, threshold: float
) -> date | None:
    for i in range(peak_idx + 1, values.size):
        if values[i - 1] >= threshold > values[i]:
            span = float(values[i - 1] - values[i])
            frac = (float(values[i - 1]) - threshold) / span if span else 0.0
            offset = (dates[i] - dates[i - 1]).days * max(0.0, min(1.0, frac))
            return dates[i - 1] + timedelta(days=round(offset))
    return None


def _smoothed(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values.copy()
    padded = np.pad(values, (1, 1), mode="edge")
    return np.convolve(padded, np.ones(3) / 3.0, mode="valid")


def _significant_peaks(values: np.ndarray) -> list[int]:
    if values.size < 3:
        return []
    smooth = _smoothed(values)
    candidates = [
        i
        for i in range(1, smooth.size - 1)
        if smooth[i] >= smooth[i - 1]
        and smooth[i] >= smooth[i + 1]
        and smooth[i] >= 0.28
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda idx: float(smooth[idx]), reverse=True)
    chosen: list[int] = []
    for idx in candidates:
        if all(abs(idx - other) >= 3 for other in chosen):
            chosen.append(idx)
        if len(chosen) == 3:
            break
    return chosen


def two_wave_detected(values: Sequence[float]) -> bool:
    arr = interpolate_series(values)
    peaks = _significant_peaks(arr)
    if len(peaks) < 2:
        return False
    smooth = _smoothed(arr)
    first, second = sorted(peaks[:2])
    high = max(float(smooth[first]), float(smooth[second]))
    low = min(float(smooth[first]), float(smooth[second]))
    if low < max(0.28, 0.55 * high):
        return False
    valley = float(np.min(smooth[first : second + 1]))
    return valley <= low - 0.06 and valley <= high - 0.10


def _second_peak_ratio(values: np.ndarray) -> float:
    peaks = _significant_peaks(values)
    if len(peaks) < 2:
        return 0.0
    smooth = _smoothed(values)
    top = float(np.max(smooth))
    if top <= 0:
        return 0.0
    return float(smooth[peaks[1]] / top)


def build_features(
    dates: Sequence[date],
    values: Sequence[float | None],
    reference_values: Sequence[float] | None = None,
    sowing_date: date | None = None,
    weather_summary: dict | None = None,
    measured: Sequence[bool] | None = None,
) -> dict:
    """Build the human-readable and scoring features for one field."""
    clean = interpolate_series(values)
    peak_idx = int(np.argmax(clean)) if clean.size else 0
    peak_value = float(clean[peak_idx]) if clean.size else 0.0
    peak_date = dates[peak_idx] if clean.size and peak_idx < len(dates) else None
    growth_start = _cross_up(dates, clean, GROWTH_NDVI_THRESHOLD)
    harvest = _cross_down(dates, clean, peak_idx, GROWTH_NDVI_THRESHOLD)
    duration = 0
    if growth_start is not None:
        end = harvest if harvest is not None else dates[-1]
        duration = max(0, (end - growth_start).days)
    correlation = pearson(clean, reference_values) if reference_values is not None else 0.0
    shifted = None
    if growth_start is not None and sowing_date is not None:
        shifted = (growth_start - sowing_date).days

    two_wave = two_wave_detected(clean)
    features: dict = {
        "growth_start_date": growth_start.isoformat() if growth_start else None,
        "peak_date": peak_date.isoformat() if peak_date else None,
        "peak_ndvi": round(peak_value, 3),
        "harvest_date": harvest.isoformat() if harvest else None,
        "vegetation_days": int(duration),
        "correlation_with_reference": round(correlation, 3),
        "mean_ndvi": round(float(np.mean(clean)) if clean.size else 0.0, 3),
        "two_wave_detected": bool(two_wave),
        "sowing_shift_days": shifted,
        "reference_peak_ndvi": (
            round(float(np.max(interpolate_series(reference_values))), 3)
            if reference_values is not None and len(reference_values)
            else None
        ),
        "precipitation_index_pct": (
            weather_summary.get("precipitation_index") if weather_summary else None
        ),
        "measured_points": (
            int(sum(1 for value in measured if value)) if measured is not None else None
        ),
    }
    return features


def model_feature_names() -> list[str]:
    raw = [f"ndvi_{i:02d}" for i in range(N_POINTS)]
    extra = [
        "peak",
        "peak_idx",
        "growth_idx",
        "harvest_idx",
        "mean",
        "std",
        "auc",
        "max_slope",
        "two_wave",
        "second_peak_ratio",
        "early_mean",
        "late_mean",
        "mid_mean",
        "corr_wheat",
        "corr_barley",
        "corr_sunflower",
        "corr_rapso",
    ]
    return raw + extra


_PROFILE_CACHE: dict[str, list[float]] | None = None


def _profile_rows() -> dict[str, list[float]]:
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None:
        try:
            path = Path(__file__).resolve().parent / "data" / "profiles.json"
            payload = json.loads(path.read_text("utf-8"))
            _PROFILE_CACHE = {
                crop: [float(item["ndvi"]) for item in data.get("series", [])]
                for crop, data in payload.get("crops", {}).items()
            }
        except Exception:
            _PROFILE_CACHE = {}
    return _PROFILE_CACHE


def model_feature_vector(values: Sequence[float | None]) -> np.ndarray:
    """Fixed-length vector used by the RandomForest classifier.

    The series is lightly smoothed and supplemented with correlations to the
    four crop reference profiles.  That makes the classifier less sensitive to
    single cloudy composites, which is important when it is applied to real
    Sentinel-2 curves outside the synthetic training distribution.
    """
    arr = interpolate_series(values)
    if arr.size != N_POINTS:
        arr = np.interp(
            np.linspace(0.0, 1.0, N_POINTS),
            np.linspace(0.0, 1.0, max(1, arr.size)),
            arr if arr.size else np.zeros(N_POINTS),
        )
    arr = _smoothed(arr)
    dates = season_dates()
    peak_idx = int(np.argmax(arr)) if arr.size else 0
    growth_start = _cross_up(dates, arr, 0.25)
    harvest = _cross_down(dates, arr, peak_idx, 0.25)
    growth_idx = dates.index(growth_start) if growth_start in dates else -1
    harvest_idx = dates.index(harvest) if harvest in dates else -1
    if growth_idx < 0 and growth_start is not None:
        growth_idx = int(round((growth_start - dates[0]).days / STEP_DAYS))
    if harvest_idx < 0 and harvest is not None:
        harvest_idx = int(round((harvest - dates[0]).days / STEP_DAYS))
    peak = float(np.max(arr)) if arr.size else 0.0
    std = float(np.std(arr)) if arr.size else 0.0
    auc = (
        float(np.sum((arr[:-1] + arr[1:]) / 2.0) / max(1, arr.size - 1))
        if arr.size > 1
        else 0.0
    )
    diff = np.diff(arr) if arr.size else np.array([0.0])
    max_slope = float(np.max(np.abs(diff))) if diff.size else 0.0
    two_wave = 1.0 if two_wave_detected(arr) else 0.0
    second_ratio = _second_peak_ratio(arr) if arr.size else 0.0
    early = float(np.mean(arr[:6])) if arr.size >= 6 else 0.0
    late = float(np.mean(arr[-5:])) if arr.size >= 5 else 0.0
    mid = float(np.mean(arr[6:15])) if arr.size >= 15 else 0.0
    profile_corrs = [
        pearson(arr, _profile_rows().get(crop, []))
        for crop in CROP_IDS
    ]
    extras = np.array(
        [
            peak,
            peak_idx / max(1, N_POINTS - 1),
            growth_idx / max(1, N_POINTS - 1) if growth_idx >= 0 else -1.0,
            harvest_idx / max(1, N_POINTS - 1) if harvest_idx >= 0 else -1.0,
            float(np.mean(arr)) if arr.size else 0.0,
            std,
            auc,
            max_slope,
            two_wave,
            second_ratio,
            early,
            late,
            mid,
            *profile_corrs,
        ],
        dtype=float,
    )
    vector = np.concatenate([arr, extras])
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def iso_to_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None
