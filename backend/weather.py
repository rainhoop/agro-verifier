"""Season weather via Open-Meteo with a labelled offline fallback.

Order of preference:
1. a previously downloaded Open-Meteo entry from the disk cache;
2. Open-Meteo Historical API (no key, 5 s timeout);
3. a climate-normal estimate for northern Kazakhstan, clearly labelled.

Set AGRO_DISABLE_NETWORK=1 to force the offline path.  Climate-norm estimates
are never written to the cache, so a later online request can still obtain the
real observation.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

from . import config

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_PATH = DATA_DIR / "weather_cache.json"
NORM_PRECIPITATION_MM = 220.0
NORM_TEMPERATURE_C = 15.5
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_lock = threading.Lock()
_memory_cache: dict[str, dict] | None = None


def make_key(latitude: float, longitude: float, start: date, end: date) -> str:
    return f"{latitude:.3f},{longitude:.3f}:{start.isoformat()}:{end.isoformat()}"


def _load_cache() -> dict[str, dict]:
    global _memory_cache
    if _memory_cache is not None:
        return _memory_cache
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            _memory_cache = dict(payload.get("entries", {}))
            return _memory_cache
        except Exception as exc:
            logger.warning("Cannot read weather cache: %s", exc)
    _memory_cache = {}
    return _memory_cache


def _save_cache(cache: dict[str, dict]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "description": (
                    "Open-Meteo 2026 season summaries for AgroVerifier demo fields. "
                    "Climate-norm estimates are not cached."
                ),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            "entries": cache,
        }
        tmp = CACHE_PATH.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        tmp.replace(CACHE_PATH)
    except OSError as exc:
        logger.warning("Cannot write weather cache: %s", exc)


def make_summary(
    precipitation_mm: float,
    temperature_mean_c: float,
    source: str,
    start: date,
    end: date,
    synthetic: bool = False,
    estimated: bool = False,
) -> dict:
    precipitation = round(float(precipitation_mm), 1)
    temperature = round(float(temperature_mean_c), 1)
    index = (
        round(precipitation / NORM_PRECIPITATION_MM * 100.0, 1)
        if NORM_PRECIPITATION_MM
        else 0.0
    )
    return {
        "precipitation_mm": precipitation,
        "norm_precipitation_mm": NORM_PRECIPITATION_MM,
        "precipitation_index": index,
        "temperature_mean_c": temperature,
        "norm_temperature_c": NORM_TEMPERATURE_C,
        "source": source,
        "synthetic": synthetic,
        "estimated": estimated,
        "cached": False,
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _climate_norm(start: date, end: date) -> dict:
    summary = make_summary(
        NORM_PRECIPITATION_MM,
        NORM_TEMPERATURE_C,
        source="climate_norm",
        start=start,
        end=end,
        synthetic=False,
        estimated=True,
    )
    summary["cached"] = False
    return summary


def get_weather(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    allow_network: bool | None = None,
) -> dict:
    key = make_key(latitude, longitude, start, end)
    cache = _load_cache()
    cached = cache.get(key)
    if cached and cached.get("source") == "open-meteo":
        result = dict(cached)
        result["cached"] = True
        return result

    if config.network_disabled():
        allow_network = False
    elif allow_network is None:
        allow_network = True

    if allow_network:
        try:
            response = httpx.get(
                ARCHIVE_URL,
                params={
                    "latitude": round(latitude, 3),
                    "longitude": round(longitude, 3),
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "daily": "precipitation_sum,temperature_2m_mean",
                    "timezone": "UTC",
                },
                timeout=config.WEATHER_TIMEOUT,
            )
            response.raise_for_status()
            daily = response.json().get("daily", {})
            rain = [
                float(x)
                for x in (daily.get("precipitation_sum") or [])
                if x is not None
            ]
            temps = [
                float(x)
                for x in (daily.get("temperature_2m_mean") or [])
                if x is not None
            ]
            if not rain or not temps:
                raise ValueError("Open-Meteo returned no daily values")
            summary = make_summary(
                sum(rain),
                sum(temps) / len(temps),
                source="open-meteo",
                start=start,
                end=end,
                synthetic=False,
                estimated=False,
            )
            with _lock:
                cache[key] = summary
                _save_cache(cache)
            return summary
        except Exception as exc:
            logger.info("Open-Meteo unavailable, using fallback: %s", exc)

    return _climate_norm(start, end)
