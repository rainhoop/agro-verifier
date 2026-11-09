"""Weather cache/fallback and ML guard-rail tests."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from backend import config, model, weather

ROOT = Path(__file__).resolve().parents[1]


def test_demo_weather_cache_hit() -> None:
    summary = weather.get_weather(
        53.920, 69.600, date(2026, 4, 1), date(2026, 10, 18)
    )
    assert summary["source"] == "open-meteo"
    assert summary["cached"] is True
    assert summary["synthetic"] is False
    assert summary["estimated"] is False
    assert summary["precipitation_mm"] > 0
    assert 0 <= summary["precipitation_index"] <= 300
    assert summary["period"] == {"start": "2026-04-01", "end": "2026-10-18"}


def test_offline_fallback_never_calls_network(monkeypatch) -> None:
    calls = []

    def fake_get(*args, **kwargs):  # pragma: no cover - must not be reached
        calls.append((args, kwargs))
        raise AssertionError("network must be disabled")

    monkeypatch.setenv("AGRO_DISABLE_NETWORK", "1")
    monkeypatch.setattr(weather.httpx, "get", fake_get)
    summary = weather.get_weather(
        47.123, 65.456, date(2026, 4, 1), date(2026, 10, 18)
    )
    assert calls == []
    assert summary["source"] == "climate_norm"
    assert summary["estimated"] is True
    assert summary["synthetic"] is False
    assert summary["precipitation_mm"] == weather.NORM_PRECIPITATION_MM
    assert summary["temperature_mean_c"] == weather.NORM_TEMPERATURE_C


def test_weather_index_math() -> None:
    summary = weather.make_summary(
        110.0,
        14.0,
        source="open-meteo",
        start=date(2026, 4, 1),
        end=date(2026, 10, 18),
        synthetic=False,
    )
    assert summary["precipitation_index"] == 50.0
    assert summary["norm_precipitation_mm"] == 220.0
    assert summary["cached"] is False


def test_model_metrics_and_prediction(demo_fields) -> None:
    info = model.model_info()
    assert info["available"] is True
    assert info["dataset"] == "synthetic"
    assert info["accuracy"] == 0.825
    assert info["n_samples"] == 1800

    metrics = json.loads(
        (ROOT / "backend" / "data" / "model_metrics.json").read_text("utf-8")
    )
    assert metrics["accuracy"] == info["accuracy"]
    assert metrics["dataset"] == "synthetic"
    assert "not validated on real fields" in metrics["validation"]
    assert len(metrics["labels"]) == 6
    assert len(metrics["confusion_matrix"]) == 6
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == metrics["n_test"]

    wheat_field = next(field for field in demo_fields if field["id"] == "field-01")
    prediction = model.predict_crop(wheat_field["ndvi"])
    assert prediction["predicted"] == "wheat"
    assert prediction["source"] == "model"
    assert len(prediction["top3"]) == 3
    assert abs(sum(prediction["probabilities"].values()) - 1.0) < 0.01
    assert 0 <= prediction["probability"] <= 1


def test_rule_based_fallback(monkeypatch) -> None:
    monkeypatch.setattr(model, "load_bundle", lambda force=False: None)
    prediction = model.predict_crop([0.12] * config.N_POINTS)
    assert prediction["predicted"] == "bare"
    assert prediction["source"] == "fallback"
    assert len(prediction["top3"]) == 3
    assert abs(sum(prediction["probabilities"].values()) - 1.0) < 0.01


def test_fallback_does_not_trust_double_wave_without_confidence() -> None:
    prediction = model.predict_crop([0.12] * config.N_POINTS)
    assert prediction["predicted"] == "bare"
