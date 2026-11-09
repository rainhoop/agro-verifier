"""Geometry helpers and a Nominatim geocoding proxy.

The browser cannot set a custom User-Agent on a cross-origin request, so
place search goes through this small backend endpoint.  Nominatim requests are
identified, cached in memory and throttled to at most one request per second.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import date
from typing import Any

import httpx

from . import config

logger = logging.getLogger(__name__)
EARTH_RADIUS_M = 6_371_008.8

_geocode_lock = threading.Lock()
_geocode_cache: dict[str, list[dict]] = {}
_geocode_last_call = 0.0


class GeoError(ValueError):
    """Invalid or unsupported geometry."""


def extract_geometry(geojson: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(geojson, dict):
        raise GeoError("GeoJSON must be an object")
    kind = geojson.get("type")
    if kind == "Feature":
        geometry = geojson.get("geometry") or {}
        if not isinstance(geometry, dict):
            raise GeoError("Feature without geometry")
        return geometry
    if kind == "FeatureCollection":
        features = geojson.get("features") or []
        if not features:
            raise GeoError("Empty FeatureCollection")
        return extract_geometry(features[0])
    return geojson


def rings_from_geometry(geometry: dict[str, Any]) -> list[list[list[float]]]:
    kind = geometry.get("type")
    coords = geometry.get("coordinates")
    if kind == "Polygon" and coords:
        return [coords[0]]
    if kind == "MultiPolygon" and coords:
        return [polygon[0] for polygon in coords if polygon]
    if kind == "Point":
        return []
    raise GeoError("Unsupported geometry; draw a polygon")


def _normalize_ring(ring: list[Any]) -> list[list[float]]:
    points: list[list[float]] = []
    for point in ring:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise GeoError("Invalid coordinate pair")
        lon, lat = float(point[0]), float(point[1])
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise GeoError("Coordinate outside valid range")
        points.append([lon, lat])
    if points and points[0] != points[-1]:
        points.append(points[0])
    if len(points) < 4:
        raise GeoError("A polygon needs at least 3 vertices")
    return points


def geometry_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    rings = [_normalize_ring(ring) for ring in rings_from_geometry(geometry)]
    if not rings:
        raise GeoError("Polygon has no coordinates")
    return rings


def ring_area_ha(ring: list[list[float]]) -> float:
    """Spherical polygon area (Chamberlain & Duquette), hectares."""
    if len(ring) < 4:
        return 0.0
    total = 0.0
    for first, second in zip(ring, ring[1:]):
        lon1, lat1 = math.radians(first[0]), math.radians(first[1])
        lon2, lat2 = math.radians(second[0]), math.radians(second[1])
        total += (lon2 - lon1) * (2.0 + math.sin(lat1) + math.sin(lat2))
    area_m2 = abs(total) * EARTH_RADIUS_M * EARTH_RADIUS_M / 2.0
    return area_m2 / 10_000.0


def area_ha(rings: list[list[list[float]]]) -> float:
    return round(sum(ring_area_ha(ring) for ring in rings), 1)


def center(rings: list[list[list[float]]]) -> tuple[float, float]:
    points = [point for ring in rings for point in ring[:-1]]
    if not points:
        raise GeoError("Polygon has no coordinates")
    lon = sum(point[0] for point in points) / len(points)
    lat = sum(point[1] for point in points) / len(points)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise GeoError("Invalid coordinates")
    return lat, lon


def bbox(rings: list[list[list[float]]]) -> tuple[float, float, float, float]:
    points = [point for ring in rings for point in ring]
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    return min(lons), min(lats), max(lons), max(lats)


def point_in_ring(point: tuple[float, float], ring: list[list[float]]) -> bool:
    lon, lat = point
    inside = False
    for first, second in zip(ring, ring[1:]):
        x1, y1 = first
        x2, y2 = second
        if (y1 > lat) != (y2 > lat):
            cross = (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-15) + x1
            if lon < cross:
                inside = not inside
    return inside


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    def orient(p, q, r):
        value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        if abs(value) < 1e-12:
            return 0
        return 1 if value > 0 else 2

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12
            and min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12
        )

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(a, c, b):
        return True
    if o2 == 0 and on_segment(a, d, b):
        return True
    if o3 == 0 and on_segment(c, a, d):
        return True
    if o4 == 0 and on_segment(c, b, d):
        return True
    return False


def ring_is_simple(ring: list[list[float]]) -> bool:
    if len(ring) < 5:
        return True
    segments = list(zip(ring, ring[1:]))
    for i, (a, b) in enumerate(segments):
        for j, (c, d) in enumerate(segments):
            if j <= i + 1 or (i == 0 and j == len(segments) - 1):
                continue
            if _segments_intersect(
                (a[0], a[1]), (b[0], b[1]), (c[0], c[1]), (d[0], d[1])
            ):
                return False
    return True


def validate_polygon(
    geometry: dict[str, Any],
    *,
    min_ha: float = config.AREA_MIN_HA,
    max_ha: float = config.AREA_MAX_HA,
) -> tuple[list[list[list[float]]], float, tuple[float, float]]:
    rings = geometry_rings(geometry)
    for ring in rings:
        if not ring_is_simple(ring):
            raise GeoError("Polygon must not intersect itself")
    polygon_area = area_ha(rings)
    if polygon_area < min_ha:
        raise GeoError(f"Area {polygon_area:.1f} ha is below the minimum {min_ha:.0f} ha")
    if polygon_area > max_ha:
        raise GeoError(f"Area {polygon_area:.1f} ha is above the maximum {max_ha:.0f} ha")
    return rings, polygon_area, center(rings)


def geocode(query: str, limit: int = 5, lang: str = "ru") -> list[dict]:
    text = (query or "").strip()
    if len(text) < 3:
        raise GeoError("Query is too short")
    if len(text) > 200:
        raise GeoError("Query is too long")
    cache_key = f"{lang}:{text.lower()}"
    with _geocode_lock:
        if cache_key in _geocode_cache:
            return _geocode_cache[cache_key]

    if config.network_disabled():
        raise GeoError("Network is disabled")

    global _geocode_last_call
    with _geocode_lock:
        wait = 1.05 - (time.monotonic() - _geocode_last_call)
        if wait > 0:
            time.sleep(wait)
        _geocode_last_call = time.monotonic()

    try:
        response = httpx.get(
            config.NOMINATIM_URL,
            params={
                "q": text,
                "format": "jsonv2",
                "limit": max(1, min(limit, 8)),
                "addressdetails": 0,
                "accept-language": "ru,kk,en",
            },
            headers={
                "User-Agent": config.NOMINATIM_USER_AGENT,
                "Accept": "application/json",
            },
            timeout=config.NOMINATIM_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        raise GeoError("Geocoder timeout") from exc
    except Exception as exc:
        logger.info("Nominatim unavailable: %s", exc)
        raise GeoError("Geocoder unavailable") from exc

    results = []
    for item in payload if isinstance(payload, list) else []:
        try:
            results.append(
                {
                    "display_name": str(item.get("display_name") or "").strip(),
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "boundingbox": [
                        float(value) for value in (item.get("boundingbox") or [])
                    ],
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    with _geocode_lock:
        _geocode_cache[cache_key] = results
    return results


def season_dates(year: int = config.SEASON_YEAR) -> list[date]:
    from datetime import timedelta

    start = date(year, config.SEASON_START[0], config.SEASON_START[1])
    return [start + timedelta(days=index * config.STEP_DAYS) for index in range(config.N_POINTS)]
