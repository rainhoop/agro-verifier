"""Geometry validation and geodesic-area tests."""
from __future__ import annotations

import pytest

from backend import config, geo


def _square(size: float, lon: float = 70.0, lat: float = 51.0) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon, lat],
                [lon + size, lat],
                [lon + size, lat + size],
                [lon, lat + size],
                [lon, lat],
            ]
        ],
    }


def test_geodesic_area_and_center() -> None:
    geometry = _square(0.01)
    rings = geo.geometry_rings(geometry)
    area = geo.area_ha(rings)
    assert 70 <= area <= 85
    lat, lon = geo.center(rings)
    assert lat == pytest.approx(51.005, abs=0.001)
    assert lon == pytest.approx(70.005, abs=0.001)


def test_polygon_is_closed_automatically() -> None:
    geometry = {
        "type": "Polygon",
        "coordinates": [[[70.0, 51.0], [70.01, 51.0], [70.01, 51.01], [70.0, 51.01]]],
    }
    rings = geo.geometry_rings(geometry)
    assert rings[0][0] == rings[0][-1]
    assert len(rings[0]) == 5


def test_feature_and_feature_collection_are_accepted() -> None:
    geometry = _square(0.01)
    feature = {"type": "Feature", "properties": {}, "geometry": geometry}
    collection = {
        "type": "FeatureCollection",
        "features": [feature],
    }
    assert geo.extract_geometry(feature) == geometry
    assert geo.extract_geometry(collection) == geometry


def test_self_intersection_is_rejected() -> None:
    bowtie = {
        "type": "Polygon",
        "coordinates": [
            [
                [70.0, 51.0],
                [70.02, 51.02],
                [70.02, 51.0],
                [70.0, 51.02],
                [70.0, 51.0],
            ]
        ],
    }
    rings = geo.geometry_rings(bowtie)
    assert not geo.ring_is_simple(rings[0])
    with pytest.raises(geo.GeoError, match="intersect"):
        geo.validate_polygon(bowtie)


def test_valid_polygon_not_rejected_for_provisional_closing_edge() -> None:
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [70.0, 51.0],
                [70.02, 51.0],
                [70.02, 51.02],
                [70.03, 51.01],
                [70.0, 51.05],
                [70.0, 51.0],
            ]
        ],
    }

    rings, area, _center = geo.validate_polygon(geometry)

    assert geo.ring_is_simple(rings[0])
    assert area == pytest.approx(583.3, abs=1.0)


def test_area_limits_are_enforced() -> None:
    with pytest.raises(geo.GeoError, match="below the minimum"):
        geo.validate_polygon(_square(0.001))
    with pytest.raises(geo.GeoError, match="above the maximum"):
        geo.validate_polygon(_square(1.0))
    rings, area, _center = geo.validate_polygon(_square(0.01))
    assert rings
    assert config.AREA_MIN_HA <= area <= config.AREA_MAX_HA


def test_invalid_coordinates_are_rejected() -> None:
    bad = {
        "type": "Polygon",
        "coordinates": [[[270.0, 91.0], [270.1, 91.0], [270.1, 91.1], [270.0, 91.0]]],
    }
    with pytest.raises(geo.GeoError):
        geo.geometry_rings(bad)


def test_geocode_offline_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGRO_DISABLE_NETWORK", "1")
    monkeypatch.setattr(config, "network_disabled", lambda: True)
    with pytest.raises(geo.GeoError, match="Network is disabled"):
        geo.geocode("Астана")
