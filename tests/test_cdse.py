"""CDSE Statistical API client tests with a mocked HTTP layer.

No test in this module may touch the network: every httpx call is replaced by
a small deterministic fake.  The fake access token is deliberately obvious and
is never written to the project cache.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from backend import cdse, config


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def _stats_payload(means: list[float | None], fractions: list[float]) -> dict:
    intervals = []
    for index, (mean, fraction) in enumerate(zip(means, fractions)):
        sample_count = 100
        no_data = int(round((1.0 - fraction) * sample_count))
        stats: dict = {"sampleCount": sample_count, "noDataCount": no_data}
        if mean is not None:
            stats["mean"] = mean
        start = date(2026, 4, 1) + timedelta(days=index * 10)
        end = start + timedelta(days=9)
        intervals.append(
            {
                "interval": {
                    "from": f"{start.isoformat()}T00:00:00Z",
                    "to": f"{end.isoformat()}T23:59:59Z",
                },
                "outputs": {"ndvi": {"bands": {"B0": {"stats": stats}}}},
            }
        )
    return {"data": intervals}


GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [70.0, 51.0],
            [70.02, 51.0],
            [70.02, 51.01],
            [70.0, 51.01],
            [70.0, 51.0],
        ]
    ],
}


@pytest.fixture(autouse=True)
def reset_cdse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cdse.invalidate_token()
    monkeypatch.setattr(cdse, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "network_disabled", lambda: False)
    monkeypatch.setenv("CDSE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "test-client-secret")
    yield
    cdse.invalidate_token()


def _patch_http(monkeypatch: pytest.MonkeyPatch, stats_payload: dict, stats_status: int = 200):
    calls: list[tuple[str, dict]] = []

    def fake_post(url: str, **kwargs):
        calls.append((url, kwargs))
        if url == cdse.TOKEN_URL:
            return FakeResponse(200, {"access_token": "fake-token", "expires_in": 1800})
        return FakeResponse(stats_status, stats_payload)

    monkeypatch.setattr(cdse.httpx, "post", fake_post)
    return calls


def test_token_is_cached_between_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_http(monkeypatch, _stats_payload([0.2, 0.3, 0.4, 0.5, 0.6], [1, 1, 1, 1, 1]))
    cdse.get_token()
    cdse.get_token()
    token_calls = [call for call in calls if call[0] == cdse.TOKEN_URL]
    assert len(token_calls) == 1
    token_body = token_calls[0][1]["data"]
    assert token_body["grant_type"] == "client_credentials"
    assert "client_secret" in token_body


def test_statistics_request_and_series_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    means = [0.11, 0.22, 0.33, 0.44, 0.55]
    fractions = [1.0, 0.2, 0.9, 0.8, 0.7]
    calls = _patch_http(monkeypatch, _stats_payload(means, fractions))

    result = cdse.statistics_for_geometry(GEOMETRY, use_cache=False)

    assert result["source"] == "sentinel-2-l2a"
    assert result["season_year"] == 2026
    assert result["aggregation"] == "P10D"
    assert len(result["dates"]) == 5
    assert len(result["ndvi"]) == 5
    assert result["measured"] == [True, False, True, True, True]
    assert result["valid_points"] == 4
    assert result["valid_fraction"][1] == 0.2
    assert result["ndvi_raw"][1] is None
    assert all(0.0 <= value <= 1.0 for value in result["ndvi"])

    stats_calls = [call for call in calls if call[0] == cdse.STATS_URL]
    assert len(stats_calls) == 1
    body = stats_calls[0][1]["json"]
    assert body["input"]["data"][0]["type"] == "sentinel-2-l2a"
    assert body["input"]["data"][0]["dataFilter"]["mosaickingOrder"] == "leastCC"
    assert (
        body["input"]["bounds"]["properties"]["crs"]
        == "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
    )
    assert body["aggregation"]["aggregationInterval"] == {"of": "P10D"}
    assert body["aggregation"]["resx"] <= config.CDSE_BASE_RESX
    assert body["aggregation"]["resy"] <= config.CDSE_BASE_RESY
    assert "B08" in body["aggregation"]["evalscript"]
    assert "dataMask" in body["aggregation"]["evalscript"]


def test_disk_cache_avoids_second_http_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_http(monkeypatch, _stats_payload([0.2, 0.3, 0.4, 0.5], [1, 1, 1, 1]))
    first = cdse.statistics_for_geometry(GEOMETRY)
    assert first["cached"] is False

    def must_not_call(url: str, **kwargs):  # pragma: no cover - assertion path
        raise AssertionError("a cached series must not hit the network")

    monkeypatch.setattr(cdse.httpx, "post", must_not_call)
    second = cdse.statistics_for_geometry(GEOMETRY)
    assert second["cached"] is True
    assert second["source"] == "sentinel-2-l2a"
    assert second["ndvi"] == first["ndvi"]


def test_transient_5xx_is_retried_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_post(url: str, **kwargs):
        calls.append((url, kwargs))
        if url == cdse.TOKEN_URL:
            return FakeResponse(200, {"access_token": "fake-token", "expires_in": 1800})
        stats_attempts = len([call for call in calls if call[0] == cdse.STATS_URL])
        if stats_attempts == 1:
            return FakeResponse(503, {"error": "temporary"})
        return FakeResponse(200, _stats_payload([0.2, 0.3, 0.4, 0.5], [1, 1, 1, 1]))

    monkeypatch.setattr(cdse.httpx, "post", fake_post)
    monkeypatch.setattr(cdse.time, "sleep", lambda _seconds: None)

    result = cdse.statistics_for_geometry(GEOMETRY)
    assert result["valid_points"] == 4
    assert len([call for call in calls if call[0] == cdse.STATS_URL]) == 2


def test_rate_limit_after_retry_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **kwargs):
        if url == cdse.TOKEN_URL:
            return FakeResponse(200, {"access_token": "fake-token", "expires_in": 1800})
        return FakeResponse(429, {"error": "quota"})

    monkeypatch.setattr(cdse.httpx, "post", fake_post)
    monkeypatch.setattr(cdse.time, "sleep", lambda _seconds: None)

    with pytest.raises(cdse.CdseRateLimitError):
        cdse.statistics_for_geometry(GEOMETRY)


def test_too_few_clear_observations_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_http(
        monkeypatch,
        _stats_payload([0.2, 0.3, 0.4, 0.5], [1.0, 0.1, 0.1, 0.1]),
    )
    with pytest.raises(cdse.CdseDataError):
        cdse.statistics_for_geometry(GEOMETRY, use_cache=False)


def test_large_polygon_increases_pixel_size() -> None:
    small_rings = [GEOMETRY["coordinates"][0]]
    large_rings = [
        [
            [60.0, 40.0],
            [62.0, 40.0],
            [62.0, 42.0],
            [60.0, 42.0],
            [60.0, 40.0],
        ]
    ]
    small = cdse._pixel_resolution(small_rings)
    large = cdse._pixel_resolution(large_rings)
    assert small == (config.CDSE_BASE_RESX, config.CDSE_BASE_RESY)
    assert large[0] > small[0]
    assert large[1] > small[1]


def test_auth_failure_does_not_leak_secret(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "super-secret-value")

    def fake_post(url: str, **kwargs):
        return FakeResponse(401, {"error": "invalid_client"})

    monkeypatch.setattr(cdse.httpx, "post", fake_post)
    with caplog.at_level("DEBUG"):
        with pytest.raises(cdse.CdseAuthError):
            cdse.get_token()
    assert "super-secret-value" not in caplog.text
    assert "test-client-secret" not in caplog.text


def test_network_disabled_prevents_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "network_disabled", lambda: True)
    with pytest.raises(cdse.CdseError):
        cdse.statistics_for_geometry(GEOMETRY)
