"""API contract and real-demo verdict tests (all offline)."""
from __future__ import annotations

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from backend import config, features
from backend.main import DEMO_FIELDS


def _verify_payload(field: dict, lang: str = "ru") -> dict:
    return {
        "demo_id": field["id"],
        "crop": field["crop"],
        "sowing_date": field["sowing_date"],
        "lang": lang,
    }


def _polygon(lon: float, lat: float, size: float) -> dict:
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


VERDICTS = {"confirmed", "doubtful", "not_found", "crop_mismatch", "needs_check"}


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "agro-verifier"
    assert payload["version"] == "0.2.0"
    assert payload["data_mode"] == "sentinel-2-l2a"
    assert payload["demo_fields"] == 6
    assert payload["real_demo_fields"] == 5
    assert payload["model"]["available"] is True
    assert payload["model"]["dataset"] == "synthetic"
    assert payload["model"]["accuracy"] == pytest.approx(0.825)
    assert payload["model"]["n_samples"] == 1800
    assert payload["network_enabled"] is False
    assert isinstance(payload["llm_configured"], bool)


def test_public_config_contract(client: TestClient) -> None:
    response = client.get("/api/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["season"]["year"] == 2026
    assert payload["season"]["start"] == "2026-04-01"
    assert payload["season"]["end"] == "2026-10-31"
    assert payload["season"]["points"] == 21
    assert payload["score"]["weights"] == {"shape": 0.5, "timing": 0.25, "peak": 0.25}
    assert payload["score"]["confirmed_min"] == config.CONFIRMED_SCORE
    assert payload["score"]["doubtful_min"] == config.DOUBTFUL_SCORE
    assert payload["verdict"]["crop_mismatch_probability"] == config.CROP_MISMATCH_PROBABILITY
    assert payload["area"] == {"min_ha": 10.0, "max_ha": 5000.0}
    assert payload["data"]["source"] == "sentinel-2-l2a"
    assert payload["data"]["network_enabled"] is False
    assert payload["model"]["dataset"] == "synthetic"
    assert payload["model"]["low_confidence_threshold"] == config.LOW_MODEL_CONFIDENCE


def test_demo_fields_contract(client: TestClient) -> None:
    response = client.get("/api/demo-fields")
    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["source"] == "sentinel-2-l2a"
    for field in payload["fields"]:
        assert field["sowing_date_conditional"] is True
        source = next(item for item in DEMO_FIELDS if item["id"] == field["id"])
        extracted = features.build_features(
            [date.fromisoformat(value) for value in source["dates"]],
            source["ndvi"],
            source["ndvi"],
            date.fromisoformat(source["sowing_date"]),
        )
        if extracted["growth_start_date"] is None or source["source"] == "synthetic":
            assert source["source"] == "synthetic"
            assert "синтетический ряд" in source["honesty_note_ru"].lower()
        else:
            days_before_growth = (
                date.fromisoformat(extracted["growth_start_date"])
                - date.fromisoformat(source["sowing_date"])
            ).days
            assert 10 <= days_before_growth <= 20
    assert payload["meta"]["synthetic"] is True
    assert payload["meta"]["real_fields"] == 5
    assert payload["meta"]["synthetic_fields"] == 1
    assert "условные" in payload["meta"]["note"]
    assert len(payload["fields"]) == 6
    assert len({field["id"] for field in payload["fields"]}) == 6
    for field in payload["fields"]:
        assert field["crop"] in {"wheat", "barley", "sunflower", "rapso"}
        assert field["source"] in {"sentinel-2-l2a", "synthetic"}
        assert field["source_label"] == config.SOURCE_LABELS[field["source"]]["ru"]
        assert len(field["polygon"]) >= 4
        assert field["area_ha"] >= 10.0
        assert field["area_ha"] <= config.DEMO_AREA_MAX_HA
        assert field["crop_label"]
        assert "условн" in field["name"].lower()
        assert field["district"]
        assert "условн" in field["note"].lower()
        assert 0 < field["measured_points"] <= field["total_points"] == 21
        assert field["expected_verdict"] in VERDICTS


def test_demo_fields_kazakh_localisation(client: TestClient) -> None:
    response = client.get("/api/demo-fields?lang=kz")
    assert response.status_code == 200
    fields = response.json()["fields"]
    assert fields[0]["name"].startswith("№1")
    assert fields[0]["crop_label"] == "Жаздық бидай"
    assert fields[0]["district"]
    assert fields[0]["note"]
    assert "шартты" in fields[0]["note"].lower()


@pytest.mark.parametrize(
    ("lat", "lon", "expected_warning"),
    [(52.0, 70.0, False), (0.0, 0.0, True)],
)
def test_drawn_contour_outside_reference_region_is_allowed(
    client: TestClient,
    lat: float,
    lon: float,
    expected_warning: bool,
) -> None:
    response = client.post(
        "/api/verify",
        json={
            "geojson": _polygon(lon, lat, 0.01),
            "crop": "wheat",
            "sowing_date": "2026-05-15",
            "lang": "ru",
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["field"].get("region_warning", False) is expected_warning
    if expected_warning:
        assert result["field"]["area_ha"] >= 10
        assert result["source"] == "synthetic"


@pytest.mark.parametrize("field", DEMO_FIELDS, ids=[field["id"] for field in DEMO_FIELDS])
def test_each_demo_verdict(client: TestClient, field: dict) -> None:
    response = client.post("/api/verify", json=_verify_payload(field))
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["verdict"] == field["expected_verdict"]
    assert payload["verdict_label"]
    assert payload["verdict_reason_code"]
    assert payload["source"] == field["source"]
    assert payload["source_label"] == config.SOURCE_LABELS[field["source"]]["ru"]
    assert payload["data_source_note"]
    assert payload["demo_mode"] is True
    assert payload["lang"] == "ru"

    assert payload["meta"]["synthetic"] is (field["source"] == "synthetic")
    assert payload["meta"]["data_source"] == field["source"]
    assert payload["meta"]["season_year"] == 2026
    assert payload["meta"]["model_dataset"] == "synthetic"
    assert payload["meta"]["total_points"] == 21
    assert payload["meta"]["measured_points"] == sum(payload["measured"])

    assert len(payload["series"]) == 21
    assert len(payload["reference"]) == 21
    assert payload["series"][0]["date"] == "2026-04-01"
    assert payload["series"][-1]["date"] == "2026-10-18"
    assert payload["temporal_palettes"] == {
        "ndvi": list(features.TEMPORAL_NDVI_COLORS),
        "deviation": list(features.TEMPORAL_DEVIATION_COLORS),
    }
    assert all(item["measured"] in (True, False) for item in payload["series"])
    assert all(0.0 <= item["ndvi"] <= 1.0 for item in payload["series"])
    assert all(item["stage"] in {
        "before_growth", "growth", "peak", "decline", "after_harvest"
    } for item in payload["series"])
    assert all(item["ndvi_color"] in features.TEMPORAL_NDVI_COLORS for item in payload["series"])
    assert all(
        item["deviation_color"] in features.TEMPORAL_DEVIATION_COLORS
        for item in payload["series"]
    )
    assert all(0.0 <= item["ndvi"] <= 1.0 for item in payload["reference"])
    assert len(payload["ndvi_raw"]) == 21
    assert len(payload["valid_fraction"]) == 21

    assert isinstance(payload["score"], int)
    assert 0 <= payload["score"] <= 100
    assert payload["score_label"] == "Соответствие заявке"
    components = payload["score_components"]
    assert 0 <= components["shape"] <= 50
    assert 0 <= components["timing"] <= 25
    assert 0 <= components["peak"] <= 25
    assert 0 <= components["peak_ratio"] <= 1
    assert components["timing_diff_days"] is None or components["timing_diff_days"] >= 0

    assert payload["ml"]["confidence_label"] == "Уверенность модели"
    assert payload["ml"]["source"] in {"model", "fallback"}
    assert 0 <= float(payload["ml"]["probability"]) <= 1
    assert len(payload["top3"]) == 3
    probabilities = payload["probabilities"]
    assert set(probabilities) == {
        "bare",
        "barley",
        "double_wave",
        "rapso",
        "sunflower",
        "wheat",
    }
    assert abs(sum(probabilities.values()) - 1.0) < 0.01

    weather = payload["weather"]
    assert weather["source"] in {"open-meteo", "climate_norm"}
    if weather["source"] == "open-meteo":
        assert weather["cached"] is True
        assert weather["estimated"] is False
    else:
        assert weather["cached"] is False
        assert weather["estimated"] is True
    assert weather["synthetic"] is False
    assert weather["source_label"]
    assert payload["explanation"]["summary"]
    assert payload["explanation"]["reasons"]
    assert payload["explanation"]["recommendation"]
    assert payload["explanation_source"] == "template"

    assert payload["field"]["name"] == field["name_ru"]
    assert payload["field"]["district"] == field["district_ru"]
    assert payload["field"]["honesty_note"]
    assert "условн" in payload["field"]["honesty_note"].lower()
    assert payload["field"]["area_ha"] == field["area_ha"]
    assert payload["field"]["crop_declared"] == field["crop"]
    assert payload["field"]["source"] == field["source"]
    assert payload["field"]["center"]["lat"] == pytest.approx(field["center"]["lat"])


def test_high_score_does_not_override_crop_mismatch(client: TestClient) -> None:
    field = next(item for item in DEMO_FIELDS if item["id"] == "field-04")
    payload = client.post("/api/verify", json=_verify_payload(field)).json()
    assert payload["score"] >= config.CONFIRMED_SCORE
    assert payload["verdict"] == "crop_mismatch"
    assert payload["verdict_reason_code"] == "crop_mismatch"
    assert payload["ml"]["predicted_crop"] != field["crop"]
    assert "Подсолнечник" in payload["explanation"]["summary"]
    assert "Пшеница" in payload["explanation"]["summary"]
    assert any("ML-модель" in reason for reason in payload["explanation"]["reasons"])


def test_demo_verdict_kazakh(client: TestClient) -> None:
    field = next(item for item in DEMO_FIELDS if item["id"] == "field-02")
    response = client.post("/api/verify", json=_verify_payload(field, lang="kz"))
    assert response.status_code == 200
    payload = response.json()
    assert payload["lang"] == "kz"
    assert payload["verdict"] == "doubtful"
    assert payload["verdict_label"] == "Күмәнді"
    assert payload["score_label"] == "Өтінімге сәйкестік"
    assert payload["ml"]["confidence_label"] == "Модель сенімділігі"
    assert payload["source_label"] == config.SOURCE_LABELS["sentinel-2-l2a"]["kz"]
    assert payload["field"]["crop_declared_label"] == "Рапс"
    assert payload["field"]["name"].startswith("№2")
    assert "шартты" in payload["field"]["honesty_note"].lower()
    assert payload["weather"]["source_label"]


def test_verify_drawn_geojson_offline(client: TestClient) -> None:
    body = {
        "geojson": _polygon(66.123, 50.321, 0.03),
        "crop": "barley",
        "sowing_date": "2026-05-06",
        "lang": "ru",
    }
    first = client.post("/api/verify", json=body)
    assert first.status_code == 200, first.text
    payload = first.json()

    assert payload["source"] == "synthetic"
    assert payload["source_label"] == "Синтетика (сервис спутниковых данных недоступен)"
    assert "недоступен" in payload["data_source_note"]
    assert payload["field"]["source"] == "synthetic"
    assert payload["field"]["synthetic_reason"] == "service_unavailable"
    assert payload["field"]["source_detail"]
    assert payload["meta"]["synthetic"] is True
    assert payload["meta"]["data_source"] == "synthetic"
    assert payload["demo_mode"] is False
    assert payload["field"]["area_ha"] > 0
    assert len(payload["series"]) == 21
    assert payload["weather"]["source"] == "climate_norm"
    assert payload["weather"]["estimated"] is True
    assert payload["verdict"] in VERDICTS
    assert "нарисован" in payload["field"]["honesty_note"].lower()

    second = client.post("/api/verify", json=body)
    assert second.status_code == 200
    assert second.json()["series"] == payload["series"]
    assert second.json()["score"] == payload["score"]


def test_drawn_kazakh_fallback_label(client: TestClient) -> None:
    response = client.post(
        "/api/verify",
        json={
            "geojson": _polygon(66.123, 50.321, 0.03),
            "crop": "barley",
            "sowing_date": "2026-05-06",
            "lang": "kz",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "synthetic"
    assert payload["source_label"] == "Синтетика (спутниктік деректер қызметі қолжетімсіз)"
    assert payload["field"]["name"] == "Өз контуры"
    assert payload["verdict_label"]


def test_drawn_area_limits(client: TestClient) -> None:
    tiny = client.post(
        "/api/verify",
        json={
            "geojson": _polygon(66.123, 50.321, 0.001),
            "crop": "wheat",
            "sowing_date": "2026-05-15",
        },
    )
    assert tiny.status_code == 422
    assert "минимум" in tiny.json()["detail"]

    huge = client.post(
        "/api/verify",
        json={
            "geojson": _polygon(60.0, 45.0, 1.0),
            "crop": "wheat",
            "sowing_date": "2026-05-15",
        },
    )
    assert huge.status_code == 422
    assert "максимум" in huge.json()["detail"]


def test_drawn_self_intersection_rejected(client: TestClient) -> None:
    bowtie = {
        "type": "Polygon",
        "coordinates": [
            [
                [66.10, 50.30],
                [66.13, 50.33],
                [66.13, 50.30],
                [66.10, 50.33],
                [66.10, 50.30],
            ]
        ],
    }
    response = client.post(
        "/api/verify",
        json={"geojson": bowtie, "crop": "wheat", "sowing_date": "2026-05-15"},
    )
    assert response.status_code == 422
    assert "пересекает" in response.json()["detail"]


def test_verify_rejects_bad_input(client: TestClient) -> None:
    missing = client.post(
        "/api/verify",
        json={"crop": "wheat", "sowing_date": "2026-05-15"},
    )
    assert missing.status_code == 422

    bad_crop = client.post(
        "/api/verify",
        json={"demo_id": "field-01", "crop": "corn", "sowing_date": "2026-05-15"},
    )
    assert bad_crop.status_code == 400

    bad_demo = client.post(
        "/api/verify",
        json={"demo_id": "missing", "crop": "wheat", "sowing_date": "2026-05-15"},
    )
    assert bad_demo.status_code == 404

    bad_geometry = client.post(
        "/api/verify",
        json={
            "geojson": {"type": "Polygon", "coordinates": []},
            "crop": "wheat",
            "sowing_date": "2026-05-15",
        },
    )
    assert bad_geometry.status_code == 422

    two_vertices = client.post(
        "/api/verify",
        json={
            "geojson": _polygon(66.123, 50.321, 0.03)
            | {"coordinates": [[[66.123, 50.321], [66.153, 50.321]]]},
            "crop": "wheat",
            "sowing_date": "2026-05-15",
        },
    )
    assert two_vertices.status_code == 422


def test_batch_json_curated_scenarios(client: TestClient) -> None:
    rows = [
        {"id": "n", "crop": "wheat", "sowing_date": "2026-05-15", "lat": 51.88, "lon": 70.10, "scenario": "normal"},
        {"id": "l", "crop": "wheat", "sowing_date": "2026-05-12", "lat": 52.62, "lon": 69.10, "scenario": "late_low"},
        {"id": "b", "crop": "wheat", "sowing_date": "2026-05-17", "lat": 51.30, "lon": 70.80, "scenario": "bare"},
        {"id": "w", "crop": "sunflower", "sowing_date": "2026-05-22", "lat": 52.70, "lon": 70.20, "scenario": "wrong_crop"},
        {"id": "d", "crop": "wheat", "sowing_date": "2026-05-15", "lat": 52.45, "lon": 71.50, "scenario": "double_wave"},
    ]
    response = client.post("/api/batch", json={"rows": rows, "lang": "ru"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["count"] == 5
    assert payload["demo_mode"] is True
    assert payload["source"] == "synthetic"
    assert payload["meta"]["synthetic"] is True
    assert "синтетические" in payload["meta"]["note"].lower()

    by_id = {item["id"]: item for item in payload["rows"]}
    assert by_id["n"]["verdict"] == "confirmed"
    assert by_id["b"]["verdict"] == "not_found"
    assert by_id["d"]["verdict"] == "needs_check"
    assert by_id["w"]["verdict"] == "crop_mismatch"
    assert by_id["l"]["verdict"] in {"doubtful", "crop_mismatch"}

    assert sum(payload["counts"].values()) == 5
    order = {"crop_mismatch": 0, "not_found": 1, "needs_check": 2, "doubtful": 3, "confirmed": 4}
    ranks = [item["risk_rank"] for item in payload["rows"]]
    assert ranks == sorted(ranks)
    assert all(item["risk_rank"] == order[item["verdict"]] for item in payload["rows"])
    for item in payload["rows"]:
        assert isinstance(item["score"], int)
        assert 0 <= item["score"] <= 100
        assert len(item["top3"]) == 3
        assert item["summary"]
        assert item["recommendation"]
        assert item["source"] == "synthetic"
        assert item["source_label"] == config.SOURCE_LABELS["synthetic"]["ru"]
        assert item["crop_label"]


def test_batch_csv_upload_kazakh(client: TestClient) -> None:
    csv_text = (
        "id,crop,sowing_date,lat,lon\n"
        "k1,wheat,2026-05-15,51.88,70.10\n"
        "k2,barley,2026-05-06,51.55,71.40\n"
    )
    response = client.post(
        "/api/batch",
        files={"file": ("portfolio.csv", csv_text.encode("utf-8"), "text/csv")},
        data={"lang": "kz"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["lang"] == "kz"
    assert payload["count"] == 2
    assert {item["id"] for item in payload["rows"]} == {"k1", "k2"}
    assert all(item["crop_label"] for item in payload["rows"])
    assert all(item["source"] == "synthetic" for item in payload["rows"])


def test_batch_validation(client: TestClient) -> None:
    too_many = [
        {"id": f"r{i}", "crop": "wheat", "sowing_date": "2026-05-15", "lat": 51.0, "lon": 70.0}
        for i in range(201)
    ]
    response = client.post("/api/batch", json={"rows": too_many})
    assert response.status_code == 413

    bad_crop = client.post(
        "/api/batch",
        json={"rows": [{"id": "x", "crop": "corn", "sowing_date": "2026-05-15", "lat": 51, "lon": 70}]},
    )
    assert bad_crop.status_code == 422

    bad_date = client.post(
        "/api/batch",
        json={"rows": [{"id": "x", "crop": "wheat", "sowing_date": "not-a-date", "lat": 51, "lon": 70}]},
    )
    assert bad_date.status_code == 422

    empty = client.post("/api/batch", json={"rows": []})
    assert empty.status_code == 422

    wrong_type = client.post("/api/batch", content="id,crop", headers={"Content-Type": "text/plain"})
    assert wrong_type.status_code == 415


def test_sample_batch_and_csv_template(client: TestClient) -> None:
    response = client.get("/api/sample-batch")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 20
    assert len(payload["rows"]) == 20
    assert payload["source"] == "synthetic"
    assert payload["meta"]["synthetic"] is True
    assert all(row["scenario"] for row in payload["rows"])

    csv_response = client.get("/api/sample-batch?format=csv")
    assert csv_response.status_code == 200
    assert "text/csv" in csv_response.headers["content-type"]
    assert "attachment" in csv_response.headers["content-disposition"]
    text = csv_response.content.decode("utf-8-sig")
    lines = text.strip().splitlines()
    assert lines[0] == "id,crop,sowing_date,lat,lon,scenario"
    assert len(lines) == 21
    assert lines[1].startswith("AKM-001,")


def test_geocode_offline_returns_actionable_error(client: TestClient) -> None:
    response = client.get("/api/geocode", params={"q": "Астана"})
    assert response.status_code == 503
    assert "интернет" in response.json()["detail"].lower()


def test_frontend_is_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "АгроВерификатор" in page.text
    assert '<div id="map"' in page.text
    assert 'lang="ru"' in page.text
    assert "apple-mobile-web-app" not in page.text

    for asset in (
        "/app.js",
        "/styles.css",
        "/i18n.json",
        "/vendor/leaflet.js",
        "/vendor/leaflet.css",
        "/vendor/chart.umd.js",
        "/vendor/fonts/inter-400-cyrillic.woff2",
    ):
        response = client.get(asset)
        assert response.status_code == 200, asset
        assert response.content


def test_openapi_parse(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema["paths"]
    for path in (
        "/api/health",
        "/api/chat/config",
        "/api/chat",
        "/api/config",
        "/api/demo-fields",
        "/api/geocode",
        "/api/verify",
        "/api/batch",
        "/api/sample-batch",
    ):
        assert path in paths
    json.dumps(schema)
