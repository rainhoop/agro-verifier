"""Build six demo fields from Sentinel-2 L2A observations where criteria match.

The script downloads (or reuses) a pool of real 2026 NDVI series for a fixed
grid of candidate rectangles in Akmola and North Kazakhstan, derives features,
and assigns one field to each demo scenario.  A scenario is filled with a
synthetic fallback only when no real candidate passes its criteria.

Run from the project root:
    python -m backend.scripts.fetch_real_demo
    python -m backend.scripts.fetch_real_demo --refresh

``--refresh`` calls the CDSE Statistical API again.  Without it the script
uses ``backend/data/candidate_pool.json`` (real downloads, kept for offline
reproducibility).  Secrets are read through ``backend.config`` and are never
printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import cdse, config, features, geo, model, scoring, synthetic, weather

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"
POOL_PATH = DATA_DIR / "candidate_pool.json"
REAL_PATH = DATA_DIR / "real_fields.json"
DEMO_PATH = DATA_DIR / "demo_fields.json"

# Half-size of approximately 450 x 500 m demo rectangles (~22 ha).
HALF_LON = 0.0038
HALF_LAT = 0.00225
DEMO_MIN_AREA_HA = 10.0

CANDIDATES: dict[str, dict] = {
    "Akmola-Shortandy-1": {
        "lat": 51.75,
        "lon": 71.05,
        "district_ru": "Шортандинский район",
        "district_kz": "Шортанды ауданы",
    },
    "Akmola-Shortandy-2": {
        "lat": 51.86,
        "lon": 70.88,
        "district_ru": "Шортандинский район",
        "district_kz": "Шортанды ауданы",
    },
    "Akmola-Shortandy-3": {
        "lat": 51.66,
        "lon": 71.18,
        "district_ru": "Шортандинский район",
        "district_kz": "Шортанды ауданы",
    },
    "Akmola-Akkol-1": {
        "lat": 52.00,
        "lon": 70.85,
        "district_ru": "Аккольский район",
        "district_kz": "Ақкөл ауданы",
    },
    "Akmola-Akkol-2": {
        "lat": 51.90,
        "lon": 71.02,
        "district_ru": "Аккольский район",
        "district_kz": "Ақкөл ауданы",
    },
    "Akmola-Astrakhanka-1": {
        "lat": 51.53,
        "lon": 69.82,
        "district_ru": "Астраханский район",
        "district_kz": "Астрахан ауданы",
    },
    "Akmola-Astrakhanka-2": {
        "lat": 51.44,
        "lon": 69.98,
        "district_ru": "Астраханский район",
        "district_kz": "Астрахан ауданы",
    },
    "Akmola-Atbasar-1": {
        "lat": 51.80,
        "lon": 68.30,
        "district_ru": "Атбасарский район",
        "district_kz": "Атбасар ауданы",
    },
    "Akmola-Atbasar-2": {
        "lat": 51.70,
        "lon": 68.20,
        "district_ru": "Атбасарский район",
        "district_kz": "Атбасар ауданы",
    },
    "Akmola-Yereymentau-1": {
        "lat": 51.62,
        "lon": 73.12,
        "district_ru": "Ерейментауский район",
        "district_kz": "Ерейментау ауданы",
    },
    "Akmola-Yereymentau-2": {
        "lat": 51.50,
        "lon": 73.24,
        "district_ru": "Ерейментауский район",
        "district_kz": "Ерейментау ауданы",
    },
    "Akmola-Zerenda-1": {
        "lat": 52.90,
        "lon": 69.12,
        "district_ru": "Зерендинский район",
        "district_kz": "Зеренді ауданы",
    },
    "Akmola-Burabay-1": {
        "lat": 53.00,
        "lon": 70.20,
        "district_ru": "Бурабайский район",
        "district_kz": "Бурабай ауданы",
    },
    "SKO-Petropavl-1": {
        "lat": 54.95,
        "lon": 69.00,
        "district_ru": "Кызылжарский район",
        "district_kz": "Қызылжар ауданы",
    },
    "SKO-Esil-1": {
        "lat": 51.96,
        "lon": 66.40,
        "district_ru": "Есильский район",
        "district_kz": "Есіл ауданы",
    },
    "SKO-Esil-2": {
        "lat": 51.88,
        "lon": 66.58,
        "district_ru": "Есильский район",
        "district_kz": "Есіл ауданы",
    },
    "SKO-Taiynsha-1": {
        "lat": 53.86,
        "lon": 69.76,
        "district_ru": "Тайыншинский район",
        "district_kz": "Тайынша ауданы",
    },
    "SKO-Taiynsha-2": {
        "lat": 53.92,
        "lon": 69.60,
        "district_ru": "Тайыншинский район",
        "district_kz": "Тайынша ауданы",
    },
    "SKO-Sergeyevka-1": {
        "lat": 53.65,
        "lon": 67.40,
        "district_ru": "Район Шал акына",
        "district_kz": "Шал ақын ауданы",
    },
    "SKO-Novoisishimka-1": {
        "lat": 53.20,
        "lon": 66.80,
        "district_ru": "Район имени Габита Мусрепова",
        "district_kz": "Ғабит Мүсірепов ауданы",
    },
}


def rectangle_polygon(lat: float, lon: float) -> list[list[float]]:
    return [
        [round(lon - HALF_LON, 7), round(lat - HALF_LAT, 7)],
        [round(lon + HALF_LON, 7), round(lat - HALF_LAT, 7)],
        [round(lon + HALF_LON, 7), round(lat + HALF_LAT, 7)],
        [round(lon - HALF_LON, 7), round(lat + HALF_LAT, 7)],
        [round(lon - HALF_LON, 7), round(lat - HALF_LAT, 7)],
    ]


def _clean_series(points: list[dict]) -> dict:
    dates = [str(point["date"]) for point in points]
    raw = [
        (round(float(point["mean"]), 5) if point.get("mean") is not None else None)
        for point in points
    ]
    fractions = [round(float(point.get("valid") or 0.0), 4) for point in points]
    measured = [
        value is not None and fraction >= config.CDSE_MIN_VALID_FRACTION
        for value, fraction in zip(raw, fractions)
    ]
    return {
        "dates": dates,
        "ndvi_raw": [value if flag else None for value, flag in zip(raw, measured)],
        "valid_fraction": fractions,
        "measured": measured,
    }


def fetch_candidate_pool() -> dict:
    candidates: dict[str, dict] = {}
    for name, meta in CANDIDATES.items():
        polygon = rectangle_polygon(meta["lat"], meta["lon"])
        geometry = {"type": "Polygon", "coordinates": [polygon]}
        print(f"  fetching {name} ...", flush=True)
        try:
            series = cdse.statistics_for_geometry(
                geometry, timeout=config.CDSE_TIMEOUT, use_cache=True
            )
        except Exception as exc:  # noqa: BLE001 - a pool candidate may be unusable
            print(f"    skipped: {type(exc).__name__}: {exc}")
            continue
        candidates[name] = {
            "lat": meta["lat"],
            "lon": meta["lon"],
            "polygon": polygon,
            "dates": series["dates"],
            "ndvi_raw": series["ndvi_raw"],
            "valid_fraction": series["valid_fraction"],
            "measured": series["measured"],
            "source": "sentinel-2-l2a",
        }
    if not candidates:
        raise SystemExit("No candidate could be downloaded from CDSE")
    return {
        "meta": {
            "source": "sentinel-2-l2a",
            "api": "Copernicus Data Space Ecosystem Statistical API",
            "collection": "sentinel-2-l2a",
            "season": "2026-04-01/2026-10-18",
            "aggregation": "P10D",
            "mosaicking_order": "leastCC",
            "scl_invalid_classes": [3, 8, 9, 10, 11],
            "min_valid_fraction": config.CDSE_MIN_VALID_FRACTION,
            "candidate_count": len(candidates),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Реальные ряды NDVI из CDSE. Кандидаты — условные прямоугольные контуры "
                "~82.5 га; принадлежность к хозяйствам не устанавливалась."
            ),
        },
        "candidates": candidates,
    }


def load_pool(refresh: bool) -> dict:
    if refresh or not POOL_PATH.exists():
        print("Downloading a fresh candidate pool from CDSE ...")
        pool = fetch_candidate_pool()
        POOL_PATH.write_text(
            json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"Saved {POOL_PATH.relative_to(ROOT)}")
        return pool
    try:
        pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Cannot read {POOL_PATH}: {exc}") from exc
    if not pool.get("candidates"):
        raise SystemExit("Candidate pool is empty; run with --refresh")
    return pool


def candidate_features(item: dict, declared_crop: str) -> dict:
    dates = [date.fromisoformat(value) for value in item["dates"]]
    raw = list(item["ndvi_raw"])
    reference = scoring.profile_values(declared_crop) or []
    return features.build_features(
        dates,
        raw,
        reference,
        None,
        {"precipitation_index": 100.0},
        item["measured"],
    )


def _date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _is_non_arable(item: dict, feat: dict) -> str | None:
    if sum(1 for flag in item["measured"] if flag) < config.CDSE_MIN_VALID_POINTS:
        return "too few clear observations"
    peak = float(feat["peak_ndvi"])
    mean = float(feat["mean_ndvi"])
    if peak < 0.15:
        return "no vegetation (water or bare surface)"
    if mean > config.FLAT_PEAK_MAX and peak - mean < 0.10:
        return "dense, almost constant canopy (forest or wetland)"
    return None


def _normal_ok(feat: dict, rec: dict) -> bool:
    start = _date(feat["growth_start_date"])
    return (
        float(feat["peak_ndvi"]) >= 0.55
        and not feat["two_wave_detected"]
        and start is not None
        and date(2026, 5, 1) <= start <= date(2026, 6, 15)
        and (
            (rec["predicted"] == "wheat" and rec["probability"] >= 0.40)
            or float(feat["correlation_with_reference"]) >= 0.85
        )
    )


def _late_weak_ok(feat: dict) -> bool:
    peak_date = _date(feat["peak_date"])
    return (
        0.30 <= float(feat["peak_ndvi"]) <= 0.50
        and peak_date is not None
        and peak_date >= date(2026, 7, 1)
        and not feat["two_wave_detected"]
    )


def _flat_ok(feat: dict) -> bool:
    return (
        float(feat["peak_ndvi"]) <= 0.30
        and float(feat["mean_ndvi"]) <= 0.28
        and not feat["two_wave_detected"]
    )


def _mismatch_ok(feat: dict, rec: dict) -> bool:
    return (
        rec["predicted"] in features.CROP_IDS
        and rec["predicted"] != "wheat"
        and rec["probability"] >= config.CROP_MISMATCH_PROBABILITY
        and float(feat["peak_ndvi"]) >= 0.50
        and not feat["two_wave_detected"]
    )


def _early_barley_ok(feat: dict, rec: dict) -> bool:
    peak_date = _date(feat["peak_date"])
    return (
        rec["predicted"] == "barley"
        and rec["probability"] >= 0.40
        and peak_date is not None
        and peak_date <= date(2026, 7, 5)
        and float(feat["peak_ndvi"]) >= 0.45
        and not feat["two_wave_detected"]
    )


def _double_wave_ok(feat: dict) -> bool:
    return bool(feat["two_wave_detected"])


SCENARIOS: list[dict] = [
    {
        "key": "normal",
        "crop": "wheat",
        "sowing_date": "2026-05-01",
        "label_ru": "Нормальный посев",
        "label_kz": "Қалыпты егіс",
        "priority": ["SKO-Taiynsha-2", "Akmola-Shortandy-2", "SKO-Novoisishimka-1"],
        "criterion": lambda feat, rec: _normal_ok(feat, rec),
        "fallback_scenario": "normal",
        "fallback_lat": 51.72,
        "fallback_lon": 70.90,
    },
    {
        "key": "late_weak",
        "crop": "rapso",
        "sowing_date": "2026-04-04",
        "label_ru": "Слабый и поздний пик",
        "label_kz": "Әлсіз және кешіккен шың",
        "priority": ["SKO-Esil-1", "Akmola-Astrakhanka-2", "Akmola-Shortandy-3"],
        "criterion": lambda feat, rec: _late_weak_ok(feat),
        "fallback_scenario": "late_low",
        "fallback_lat": 51.40,
        "fallback_lon": 69.90,
    },
    {
        "key": "flat",
        "crop": "wheat",
        "sowing_date": "2026-05-05",
        "label_ru": "Плоская кривая (пар или залежь)",
        "label_kz": "Тегіс қисық (пар немесе тыңайған жер)",
        "priority": ["Akmola-Shortandy-3", "Akmola-Astrakhanka-2"],
        "criterion": lambda feat, rec: _flat_ok(feat),
        "fallback_scenario": "bare",
        "fallback_lat": 51.20,
        "fallback_lon": 70.40,
    },
    {
        "key": "crop_mismatch",
        "crop": "wheat",
        "sowing_date": "2026-04-12",
        "label_ru": "Несоответствие заявленной культуры",
        "label_kz": "Мәлімделген дақылға сәйкессіздік",
        "priority": ["SKO-Petropavl-1", "SKO-Sergeyevka-1", "Akmola-Akkol-1"],
        "criterion": lambda feat, rec: _mismatch_ok(feat, rec),
        "fallback_scenario": "wrong_crop",
        "fallback_lat": 52.20,
        "fallback_lon": 70.20,
    },
    {
        "key": "early_barley",
        "crop": "barley",
        "sowing_date": "2026-05-20",
        "label_ru": "Ранний пик ячменя",
        "label_kz": "Арпаның ерте шыңы",
        "priority": ["Akmola-Akkol-2", "SKO-Petropavl-1"],
        "criterion": lambda feat, rec: _early_barley_ok(feat, rec),
        "fallback_scenario": "early_stress",
        "fallback_lat": 51.84,
        "fallback_lon": 70.90,
    },
    {
        "key": "double_wave",
        "crop": "wheat",
        "sowing_date": "2026-05-08",
        "label_ru": "Две волны роста",
        "label_kz": "Екі өсу толқыны",
        "priority": ["SKO-Esil-2", "Akmola-Yereymentau-2"],
        "criterion": lambda feat, rec: _double_wave_ok(feat),
        "fallback_scenario": "double_wave",
        "fallback_lat": 51.60,
        "fallback_lon": 71.30,
    },
]


def _series_from_pool(item: dict) -> tuple[list[date], list[float | None], list[bool]]:
    dates = [date.fromisoformat(value) for value in item["dates"]]
    raw = list(item["ndvi_raw"])
    measured = list(item["measured"])
    return dates, raw, measured


def _weather_for(center: tuple[float, float], allow_network: bool) -> dict:
    return weather.get_weather(
        center[0],
        center[1],
        date(config.SEASON_YEAR, 4, 1),
        date(config.SEASON_YEAR, 10, 18),
        allow_network=allow_network,
    )


def _build_field(
    *,
    index: int,
    scenario: dict,
    candidate_id: str,
    item: dict,
    pool_meta: dict,
    allow_network: bool,
) -> dict:
    polygon = item.get("polygon") or rectangle_polygon(item["lat"], item["lon"])
    geometry = {"type": "Polygon", "coordinates": [polygon]}
    rings, area, (lat, lon) = geo.validate_polygon(
        geometry, min_ha=DEMO_MIN_AREA_HA, max_ha=config.DEMO_AREA_MAX_HA
    )
    dates, raw, measured = _series_from_pool(item)
    clean = features.interpolate_series(raw)
    crop = scenario["crop"]
    sowing = date.fromisoformat(scenario["sowing_date"])
    reference = scoring.profile_values(crop) or []
    weather_summary = _weather_for((lat, lon), allow_network)
    feat = features.build_features(
        dates, raw, reference, sowing, weather_summary, measured
    )
    expected_verdict, reason = scoring.decide_verdict(
        feat,
        scoring.calculate_score(feat, scoring.reference_peak(crop), sowing)[0],
        model.predict_crop(clean),
        crop,
        scoring.reference_peak(crop),
    )
    notes = {
        "ru": (
            "Контур и заявка условные. Истинная культура и дата сева на реальном поле "
            "неизвестны; контур не привязан к хозяйству."
        ),
        "kz": (
            "Контур мен өтінім шартты. Нақты алқаптағы шынайы дақыл мен себу күні "
            "белгісіз; контур шаруашылыққа байланыстырылмаған."
        ),
    }
    return {
        "id": f"field-{index:02d}",
        "candidate_id": candidate_id,
        "name_ru": f"Поле №{index} · условный контур",
        "name_kz": f"№{index} алқап · шартты контур",
        "district_ru": CANDIDATES.get(candidate_id, {}).get(
            "district_ru", "Северный Казахстан"
        ),
        "district_kz": CANDIDATES.get(candidate_id, {}).get(
            "district_kz", "Солтүстік Қазақстан"
        ),
        "scenario": scenario["key"],
        "scenario_label_ru": scenario["label_ru"],
        "scenario_label_kz": scenario["label_kz"],
        "crop": crop,
        "crop_label_ru": features.crop_label(crop, "ru"),
        "crop_label_kz": features.crop_label(crop, "kz"),
        "sowing_date": scenario["sowing_date"],
        "sowing_date_conditional": True,
        "center": {"lat": round(lat, 5), "lon": round(lon, 5)},
        "polygon": polygon,
        "area_ha": area,
        "dates": [value.isoformat() for value in dates],
        "ndvi_raw": raw,
        "valid_fraction": item["valid_fraction"],
        "ndvi": [round(float(value), 3) for value in clean],
        "measured": measured,
        "measured_points": sum(1 for flag in measured if flag),
        "total_points": len(dates),
        "source": "sentinel-2-l2a",
        "source_note_ru": "Реальные данные Sentinel-2 L2A, 10-дневные композиты.",
        "source_note_kz": "Sentinel-2 L2A нақты деректері, 10 күндік композиттер.",
        "honesty_note_ru": notes["ru"],
        "honesty_note_kz": notes["kz"],
        "expected_verdict": expected_verdict,
        "expected_reason": reason,
        "selected_with": (
            "реальный кандидат" if item.get("source") == "sentinel-2-l2a" else "синтетика"
        ),
        "fetched_at": pool_meta.get("created_at") or pool_meta.get("fetched_at"),
    }


def _build_synthetic_field(
    *, index: int, scenario: dict, allow_network: bool
) -> dict:
    lat, lon = scenario["fallback_lat"], scenario["fallback_lon"]
    polygon = rectangle_polygon(lat, lon)
    rings, area, (clat, clon) = geo.validate_polygon(
        {"type": "Polygon", "coordinates": [polygon]},
        min_ha=DEMO_MIN_AREA_HA,
        max_ha=config.DEMO_AREA_MAX_HA,
    )
    generated = synthetic.synthetic_series_for_field(
        clat, clon, scenario["crop"], date.fromisoformat(scenario["sowing_date"]),
        scenario=scenario["fallback_scenario"],
    )
    dates = generated["dates"]
    clean = [round(float(value), 3) for value in generated["values"]]
    reference = scoring.profile_values(scenario["crop"]) or []
    weather_summary = _weather_for((clat, clon), allow_network)
    feat = features.build_features(
        dates, clean, reference, date.fromisoformat(scenario["sowing_date"]),
        weather_summary, generated["measured"]
    )
    score = scoring.calculate_score(
        feat, scoring.reference_peak(scenario["crop"]),
        date.fromisoformat(scenario["sowing_date"])
    )[0]
    verdict, reason = scoring.decide_verdict(
        feat, score, model.predict_crop(clean), scenario["crop"],
        scoring.reference_peak(scenario["crop"])
    )
    return {
        "id": f"field-{index:02d}",
        "candidate_id": f"synthetic-{scenario['key']}",
        "name_ru": f"Поле №{index} · синтетический условный контур",
        "name_kz": f"№{index} алқап · синтетикалық шартты контур",
        "district_ru": "Условный район",
        "district_kz": "Шартты аудан",
        "scenario": scenario["key"],
        "scenario_label_ru": scenario["label_ru"],
        "scenario_label_kz": scenario["label_kz"],
        "crop": scenario["crop"],
        "crop_label_ru": features.crop_label(scenario["crop"], "ru"),
        "crop_label_kz": features.crop_label(scenario["crop"], "kz"),
        "sowing_date": scenario["sowing_date"],
        "sowing_date_conditional": True,
        "center": {"lat": round(clat, 5), "lon": round(clon, 5)},
        "polygon": polygon,
        "area_ha": area,
        "dates": [value.isoformat() for value in dates],
        "ndvi_raw": [
            value if flag else None for value, flag in zip(clean, generated["measured"])
        ],
        "valid_fraction": generated["valid_fraction"],
        "ndvi": clean,
        "measured": generated["measured"],
        "measured_points": sum(1 for flag in generated["measured"] if flag),
        "total_points": len(dates),
        "source": "synthetic",
        "source_note_ru": "Синтетика: реальный кандидат для сценария не найден.",
        "source_note_kz": "Синтетика: сценарий үшін нақты кандидат табылмады.",
        "honesty_note_ru": (
            "Синтетический ряд. Не является спутниковыми наблюдениями; контур и заявка условные."
        ),
        "honesty_note_kz": (
            "Синтетикалық қатар. Спутниктік бақылау емес; контур мен өтінім шартты."
        ),
        "expected_verdict": verdict,
        "expected_reason": reason,
        "selected_with": "синтетика (реальный кандидат не найден)",
        "fetched_at": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download all candidates from CDSE instead of using candidate_pool.json",
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="do not request Open-Meteo while building demo fields",
    )
    args = parser.parse_args()

    pool = load_pool(refresh=args.refresh)
    pool_candidates: dict = pool["candidates"]
    pool_meta: dict = pool.get("meta") or {}

    prepared: dict[str, dict] = {}
    rejects: list[tuple[str, str]] = []
    for name, item in pool_candidates.items():
        declared = "wheat"
        feat = candidate_features(item, declared)
        rec = model.predict_crop(features.interpolate_series(item["ndvi_raw"]))
        reason = _is_non_arable(item, feat)
        if reason:
            rejects.append((name, reason))
            continue
        prepared[name] = {"item": item, "feat": feat, "rec": rec}

    chosen: dict[str, str] = {}
    used: set[str] = set()
    for scenario in SCENARIOS:
        match = None
        for candidate_id in scenario["priority"]:
            if candidate_id in used or candidate_id not in prepared:
                continue
            record = prepared[candidate_id]
            generic_feat = record["feat"]
            rec = record["rec"]
            try:
                ok = scenario["criterion"](generic_feat, rec)
            except Exception:
                ok = False
            if ok:
                match = candidate_id
                break
        if match is None:
            # Fall back to any unused candidate that passes the criterion.
            for candidate_id, record in prepared.items():
                if candidate_id in used:
                    continue
                if scenario["criterion"](record["feat"], record["rec"]):
                    match = candidate_id
                    break
        if match:
            chosen[scenario["key"]] = match
            used.add(match)
        else:
            print(f"! no real candidate for scenario {scenario['key']}; using synthetic")

    fields: list[dict] = []
    for index, scenario in enumerate(SCENARIOS, start=1):
        candidate_id = chosen.get(scenario["key"])
        if candidate_id:
            field = _build_field(
                index=index,
                scenario=scenario,
                candidate_id=candidate_id,
                item=pool_candidates[candidate_id],
                pool_meta=pool_meta,
                allow_network=not args.skip_weather,
            )
        else:
            field = _build_synthetic_field(
                index=index, scenario=scenario, allow_network=not args.skip_weather
            )
        fields.append(field)

    real_count = sum(1 for field in fields if field["source"] == "sentinel-2-l2a")
    meta = {
        "season_year": config.SEASON_YEAR,
        "season_start": f"{config.SEASON_YEAR}-04-01",
        "season_end": "2026-10-18",
        "step_days": config.STEP_DAYS,
        "source": "sentinel-2-l2a",
        "real_fields": real_count,
        "synthetic_fields": len(fields) - real_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api": "Copernicus Data Space Ecosystem Statistical API",
        "aggregation": "P10D",
        "honesty_note": (
            "Истинная культура и дата сева на реальных контурах неизвестны; заявка условна. "
            "Контуры не привязаны к хозяйствам. Модель обучена на синтетике."
        ),
        "rejected_candidates": [
            {"candidate": name, "reason": reason} for name, reason in rejects
        ],
        "selected_candidates": chosen,
    }
    real_payload = {"meta": meta, "fields": fields}
    REAL_PATH.write_text(
        json.dumps(real_payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    demo_payload = {
        "meta": {
            "season_year": config.SEASON_YEAR,
            "synthetic": real_count != len(fields),
            "real_fields": real_count,
            "note": (
                "Демо-поля: реальные ряды Sentinel-2 L2A там, где это указано в source; "
                "контур и заявка условные. Синтетика явно помечена."
            ),
        },
        "fields": fields,
    }
    DEMO_PATH.write_text(
        json.dumps(demo_payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"\nWrote {REAL_PATH.relative_to(ROOT)} and {DEMO_PATH.relative_to(ROOT)}")
    print(f"Real fields: {real_count}/{len(fields)}")
    for field in fields:
        print(
            f"  {field['id']}  {field['scenario']:14s}  {field['candidate_id']:20s}  "
            f"{field['crop']:9s}  {field['area_ha']:5.1f} ha  "
            f"source={field['source']:16s}  expected={field['expected_verdict']} "
            f"({field['expected_reason']})"
        )
    if rejects:
        print("\nRejected as non-arable / unusable:")
        for name, reason in rejects:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
