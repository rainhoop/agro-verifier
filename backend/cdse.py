"""Copernicus Data Space Ecosystem (Sentinel Hub) Statistical API client.

Only the two endpoints needed by the service are implemented:

* OAuth2 client-credentials token, cached until 60 s before expiry;
* Sentinel-2 L2A statistical NDVI for a polygon, 10-day composites.

The module never logs or returns the client id/secret or the access token.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from . import config, geo
from .features import interpolate_series

logger = logging.getLogger(__name__)

STATS_URL = config.CDSE_STATS_URL
TOKEN_URL = config.CDSE_TOKEN_URL
CACHE_DIR = config.CDSE_CACHE_DIR

_token_lock = threading.Lock()
_token_value: str | None = None
_token_expires_at = 0.0

EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL", "dataMask"], units: "DN" }],
    output: [
      { id: "ndvi", bands: 1, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(s) {
  let denom = s.B08 + s.B04;
  let ndvi = denom > 0 ? (s.B08 - s.B04) / denom : 0;
  let invalidScl = (s.SCL === 3 || s.SCL === 8 || s.SCL === 9 ||
                    s.SCL === 10 || s.SCL === 11);
  return { ndvi: [ndvi], dataMask: [s.dataMask * (invalidScl ? 0 : 1)] };
}
"""


class CdseError(RuntimeError):
    """Base class for user-facing CDSE failures."""


class CdseAuthError(CdseError):
    pass


class CdseRateLimitError(CdseError):
    pass


class CdseDataError(CdseError):
    pass


def _credentials_or_raise() -> tuple[str, str]:
    credentials = config.cdse_credentials()
    if not credentials:
        raise CdseAuthError("CDSE credentials are not configured")
    return credentials


def get_token(force: bool = False) -> str:
    """Return a cached access token, refreshing it under a lock."""
    global _token_value, _token_expires_at
    with _token_lock:
        now = time.time()
        if not force and _token_value and now < _token_expires_at:
            return _token_value
        client_id, client_secret = _credentials_or_raise()
        try:
            response = httpx.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=config.CDSE_TIMEOUT,
            )
        except httpx.TimeoutException as exc:
            raise CdseAuthError("CDSE token request timed out") from exc
        except Exception as exc:
            logger.info("CDSE token request failed: %s", type(exc).__name__)
            raise CdseAuthError("CDSE token request failed") from exc

        if response.status_code >= 400:
            logger.warning("CDSE token endpoint returned HTTP %s", response.status_code)
            raise CdseAuthError("CDSE authentication failed")
        try:
            payload = response.json()
            token = str(payload["access_token"])
        except Exception as exc:
            raise CdseAuthError("CDSE returned an invalid token response") from exc
        expires_in = float(payload.get("expires_in") or 1800)
        _token_value = token
        _token_expires_at = now + max(120.0, expires_in - config.CDSE_TOKEN_SAFETY_SECONDS)
        return token


def invalidate_token() -> None:
    global _token_value, _token_expires_at
    with _token_lock:
        _token_value = None
        _token_expires_at = 0.0


def _pixel_resolution(rings: list[list[list[float]]]) -> tuple[float, float]:
    """Keep the request under the Statistical API pixel budget."""
    min_lon, min_lat, max_lon, max_lat = geo.bbox(rings)
    width_deg = max(1e-9, max_lon - min_lon)
    height_deg = max(1e-9, max_lat - min_lat)
    base_x, base_y = config.CDSE_BASE_RESX, config.CDSE_BASE_RESY
    estimated = (width_deg / base_x) * (height_deg / base_y)
    scale = math.sqrt(estimated / config.CDSE_MAX_PIXELS) if estimated > config.CDSE_MAX_PIXELS else 1.0
    if scale <= 1.0:
        return base_x, base_y
    # Round up slightly to stay below the limit even after raster alignment.
    return round(base_x * scale * 1.02, 8), round(base_y * scale * 1.02, 8)


def _extract_series(payload: dict) -> dict:
    intervals = payload.get("data")
    if not isinstance(intervals, list) or not intervals:
        raise CdseDataError("Sentinel-2 returned no intervals")
    dates: list[str] = []
    raw: list[float | None] = []
    fractions: list[float] = []
    for item in intervals:
        interval = item.get("interval") or {}
        start = str(interval.get("from") or "")[:10]
        if not start:
            continue
        stats = (
            (((item.get("outputs") or {}).get("ndvi") or {}).get("bands") or {})
            .get("B0", {})
            .get("stats", {})
        )
        sample_count = int(stats.get("sampleCount") or 0)
        no_data = int(stats.get("noDataCount") or 0)
        fraction = (sample_count - no_data) / sample_count if sample_count else 0.0
        mean = stats.get("mean")
        measured = fraction >= config.CDSE_MIN_VALID_FRACTION and mean is not None
        dates.append(start)
        fractions.append(round(max(0.0, min(1.0, fraction)), 4))
        raw.append(round(float(mean), 5) if measured else None)

    if not dates:
        raise CdseDataError("Sentinel-2 returned no dated intervals")
    measured_count = sum(1 for value in raw if value is not None)
    if measured_count < config.CDSE_MIN_VALID_POINTS:
        raise CdseDataError(
            f"Only {measured_count} clear Sentinel-2 observations; "
            f"need at least {config.CDSE_MIN_VALID_POINTS}"
        )
    filled = interpolate_series([value if value is not None else float("nan") for value in raw])
    return {
        "dates": dates,
        "ndvi_raw": raw,
        "ndvi": [round(float(value), 3) for value in filled],
        "valid_fraction": fractions,
        "measured": [value is not None for value in raw],
        "valid_points": measured_count,
        "total_points": len(dates),
    }


def _cache_path(geometry: dict, start: date, end: date) -> Path:
    payload = json.dumps(
        {"geometry": geometry, "start": start.isoformat(), "end": end.isoformat()},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:28]
    return CACHE_DIR / f"{digest}.json"


def _read_cache(path: Path) -> dict | None:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.info("Cannot read CDSE cache: %s", exc)
    return None


def _write_cache(path: Path, value: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.info("Cannot write CDSE cache: %s", exc)


def _post_statistics(body: dict, timeout: float) -> dict:
    retries = config.CDSE_RETRIES
    token_refreshed = False
    for attempt in range(retries + 1):
        token = get_token(force=token_refreshed and attempt > 0)
        try:
            response = httpx.post(
                STATS_URL,
                json=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            raise CdseError("Sentinel-2 statistics request timed out") from exc
        except Exception as exc:
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            raise CdseError("Sentinel-2 statistics request failed") from exc

        if response.status_code == 401 and not token_refreshed:
            invalidate_token()
            token_refreshed = True
            continue
        if response.status_code == 429:
            if attempt < retries:
                time.sleep(3 * (2**attempt))
                continue
            raise CdseRateLimitError("Sentinel-2 rate limit exceeded")
        if 500 <= response.status_code < 600:
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            raise CdseError(f"Sentinel-2 service unavailable (HTTP {response.status_code})")
        if response.status_code >= 400:
            logger.warning("CDSE statistics endpoint returned HTTP %s", response.status_code)
            raise CdseError(f"Sentinel-2 request rejected (HTTP {response.status_code})")
        try:
            return response.json()
        except Exception as exc:
            raise CdseError("Sentinel-2 returned invalid JSON") from exc
    raise CdseError("Sentinel-2 request failed")


def statistics_for_geometry(
    geometry: dict[str, Any],
    *,
    start: date | None = None,
    end: date | None = None,
    timeout: float = config.CDSE_DRAWN_TIMEOUT,
    use_cache: bool = True,
) -> dict:
    """Fetch and normalise a 10-day NDVI series for a GeoJSON Polygon."""
    if config.network_disabled():
        raise CdseError("Network is disabled")
    rings = geo.geometry_rings(geometry)
    start = start or date(config.SEASON_YEAR, 4, 1)
    end = end or date(config.SEASON_YEAR, 10, 31)
    cache_path = _cache_path(geometry, start, end)
    if use_cache:
        cached = _read_cache(cache_path)
        if cached and cached.get("source") == "sentinel-2-l2a":
            cached["cached"] = True
            return cached

    resx, resy = _pixel_resolution(rings)
    body = {
        "input": {
            "bounds": {
                "geometry": geometry,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "mosaickingOrder": "leastCC",
                        "maxCloudCoverage": 85,
                    },
                }
            ],
        },
        "aggregation": {
            "timeRange": {
                "from": f"{start.isoformat()}T00:00:00Z",
                "to": f"{end.isoformat()}T23:59:59Z",
            },
            "aggregationInterval": {"of": "P10D"},
            "evalscript": EVALSCRIPT,
            "resx": resx,
            "resy": resy,
        },
        "calculations": {
            "default": {"statistics": {"default": {"percentiles": {"k": [25, 50, 75]}}}}
        },
    }
    payload = _post_statistics(body, timeout)
    series = _extract_series(payload)
    result = {
        **series,
        "source": "sentinel-2-l2a",
        "season_year": start.year,
        "aggregation": "P10D",
        "resolution_deg": [resx, resy],
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cached": False,
    }
    if use_cache:
        _write_cache(cache_path, result)
    return result


def warm_token() -> bool:
    """Used by /api/health without exposing the token."""
    if config.network_disabled() or not config.cdse_credentials():
        return False
    try:
        get_token()
        return True
    except CdseError:
        return False
