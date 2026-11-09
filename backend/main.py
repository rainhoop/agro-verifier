"""AgroVerifier FastAPI application: API + static one-page frontend.

Data policy
-----------
* Demo fields use pre-downloaded Sentinel-2 L2A series from 2026 where a real
  candidate matches the scenario; synthetic fallbacks are explicitly labelled.
* A contour drawn by the user is sent to the CDSE Statistical API live.
* If CDSE is unavailable, the response falls back to a synthetic series and is
  explicitly labelled ``source = "synthetic"``.  Synthetic data is never
  presented as a satellite observation.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import chat, cdse, config, explain, features, geo, model, scoring, synthetic, weather
from .features import CROP_IDS, crop_label, normalize_crop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agro-verifier")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
FRONTEND_DIR = ROOT / "frontend"
DEMO_PATH = DATA_DIR / "demo_fields.json"
REAL_PATH = DATA_DIR / "real_fields.json"
VERSION = "0.2.0"
MAX_BATCH_ROWS = 200
NORTH_KZ_REGION = (48.4, 57.8, 57.5, 84.0)
RISK_ORDER = {
    "crop_mismatch": 0,
    "not_found": 1,
    "needs_check": 2,
    "doubtful": 3,
    "confirmed": 4,
}
VERDICT_LABELS = {
    "ru": {
        "confirmed": "Подтверждено",
        "doubtful": "Сомнительно",
        "not_found": "Посева не обнаружено",
        "crop_mismatch": "Несоответствие культуры",
        "needs_check": "Требует проверки",
    },
    "kz": {
        "confirmed": "Расталды",
        "doubtful": "Күмәнді",
        "not_found": "Егіс табылмады",
        "crop_mismatch": "Дақыл сәйкессіздігі",
        "needs_check": "Тексеруді қажет етеді",
    },
}
SOURCE_NOTE = {
    "sentinel-2-l2a": {
        "ru": (
            "Sentinel-2 L2A, 10-дневные композиты, 2026. Облачные интервалы "
            "восстановлены линейно; измеренные точки отмечены на графике."
        ),
        "kz": (
            "Sentinel-2 L2A, 10 күндік композиттер, 2026. Бұлтты аралықтар "
            "сызықтық толтырылды; өлшенген нүктелер графикте белгіленген."
        ),
    },
    "synthetic": {
        "ru": (
            "Синтетика. Ряд не является наблюдением Sentinel-2; используйте его "
            "только для демонстрации."
        ),
        "kz": (
            "Синтетика. Қатар Sentinel-2 бақылауы емес; тек демонстрация үшін "
            "қолданылады."
        ),
    },
}
SYNTHETIC_UNAVAILABLE = {
    "ru": {
        "label": "Синтетика (сервис спутниковых данных недоступен)",
        "note": (
            "Синтетика (сервис спутниковых данных недоступен). Ряд не является "
            "наблюдением Sentinel-2; используйте его только для демонстрации."
        ),
    },
    "kz": {
        "label": "Синтетика (спутниктік деректер қызметі қолжетімсіз)",
        "note": (
            "Синтетика (спутниктік деректер қызметі қолжетімсіз). Қатар Sentinel-2 "
            "бақылауы емес; тек демонстрация үшін қолданылады."
        ),
    },
}
DISCLAIMER = {
    "ru": (
        "Результат не является юридическим доказательством. Истинная культура на "
        "реальных демо-контурах неизвестна, заявка условна."
    ),
    "kz": (
        "Нәтиже заңды дәлел болып саналмайды. Нақты демо-контурлардағы шынайы дақыл "
        "белгісіз, өтінім шартты."
    ),
}
DRAW_NAME = {"ru": "Свой контур", "kz": "Өз контуры"}

app = FastAPI(
    title="АгроВерификатор / AgroVerifier",
    version=VERSION,
    description=(
        "MVP кредитной верификации посевов по Sentinel-2 L2A. "
        "Демо-контуры и заявки условные; синтетические fallback-ряды помечаются."
    ),
)


class VerifyRequest(BaseModel):
    demo_id: str | None = None
    geojson: dict[str, Any] | None = None
    crop: str
    sowing_date: date
    lang: Literal["ru", "kz"] = "ru"


def _load_demo_fields() -> list[dict]:
    for path in (DEMO_PATH, REAL_PATH):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fields = payload.get("fields") or []
            if fields:
                return fields
        except Exception as exc:
            logger.warning("Cannot read %s: %s", path.name, exc)
    return []


DEMO_FIELDS = _load_demo_fields()


def _find_demo(demo_id: str) -> dict | None:
    return next((field for field in DEMO_FIELDS if field["id"] == demo_id), None)


def _error(lang: str, ru: str, kz: str, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=kz if lang == "kz" else ru)


def _source_label(source: str, lang: str) -> str:
    entry = config.SOURCE_LABELS.get(source, config.SOURCE_LABELS["synthetic"])
    return entry.get("kz" if lang == "kz" else "ru", entry["ru"])


def _source_note(source: str, lang: str) -> str:
    return SOURCE_NOTE.get(source, SOURCE_NOTE["synthetic"])[
        "kz" if lang == "kz" else "ru"
    ]


def _localized_demo(field: dict, lang: str) -> dict:
    suffix = "kz" if lang == "kz" else "ru"
    source = field.get("source", "synthetic")
    return {
        "id": field["id"],
        "name": field.get(f"name_{suffix}", field.get("name_ru", field["id"])),
        "name_ru": field.get("name_ru"),
        "name_kz": field.get("name_kz"),
        "district": field.get(f"district_{suffix}", field.get("district_ru")),
        "district_ru": field.get("district_ru"),
        "district_kz": field.get("district_kz"),
        "scenario": field.get("scenario"),
        "scenario_label": field.get(
            f"scenario_label_{suffix}", field.get("scenario_label_ru")
        ),
        "crop": field["crop"],
        "crop_label": crop_label(field["crop"], lang),
        "sowing_date": field["sowing_date"],
        "sowing_date_conditional": field.get("sowing_date_conditional", False),
        "center": field["center"],
        "polygon": field["polygon"],
        "area_ha": field.get("area_ha"),
        "source": source,
        "source_label": _source_label(source, lang),
        "note": field.get(f"honesty_note_{suffix}") or _source_note(source, lang),
        "measured_points": field.get("measured_points"),
        "total_points": field.get("total_points"),
        "expected_verdict": field.get("expected_verdict"),
    }


def _rings_from_geometry(geometry: dict[str, Any]) -> list[list[list[float]]]:
    return geo.geometry_rings(geometry)


def _validate_drawn(lang: str, geometry: dict[str, Any]) -> tuple[list, float, tuple[float, float]]:
    try:
        rings = geo.geometry_rings(geometry)
    except geo.GeoError as exc:
        message = str(exc)
        if "at least 3" in message:
            raise _error(
                lang,
                "В контуре меньше трёх вершин. Поставьте минимум три точки.",
                "Контурда үш төбеден аз. Кемінде үш нүкте қойыңыз.",
                422,
            ) from exc
        raise _error(
            lang,
            "Не удалось прочитать контур. Нарисуйте полигон заново.",
            "Контурды оқу мүмкін болмады. Полигонды қайта сызыңыз.",
            422,
        ) from exc
    for ring in rings:
        if not geo.ring_is_simple(ring):
            raise _error(
                lang,
                "Контур пересекает сам себя. Уберите пересечение и завершите рисование.",
                "Контур өзін қиып өтеді. Қиылысуды алып тастап, сызуды аяқтаңыз.",
                422,
            )
    area = geo.area_ha(rings)
    if area < config.AREA_MIN_HA:
        raise _error(
            lang,
            f"Площадь контура {area:.1f} га — меньше минимума {config.AREA_MIN_HA:.0f} га. "
            "Обведите поле целиком.",
            f"Контур ауданы {area:.1f} га — минимум {config.AREA_MIN_HA:.0f} га-дан аз. "
            "Алқапты толық қамтыңыз.",
            422,
        )
    if area > config.AREA_MAX_HA:
        raise _error(
            lang,
            f"Площадь контура {area:.1f} га — больше максимума {config.AREA_MAX_HA:.0f} га. "
            "Выберите одно поле, а не несколько хозяйств.",
            f"Контур ауданы {area:.1f} га — максимум {config.AREA_MAX_HA:.0f} га-дан көп. "
            "Бір алқапты таңдаңыз.",
            422,
        )
    try:
        center = geo.center(rings)
    except geo.GeoError as exc:
        raise _error(
            lang,
            "Не удалось определить центр контура.",
            "Контур орталығын анықтау мүмкін болмады.",
            422,
        ) from exc
    return rings, area, center


def _is_in_north_kazakhstan(lat: float, lon: float) -> bool:
    south, north, west, east = NORTH_KZ_REGION
    return south <= lat <= north and west <= lon <= east


def _weather_public(summary: dict, lang: str) -> dict:
    payload = dict(summary)
    source = payload.get("source", "climate_norm")
    names = {
        "open-meteo": {
            "ru": "Open-Meteo, исторические данные 2026",
            "kz": "Open-Meteo, 2026 тарихи деректері",
        },
        "climate_norm": {
            "ru": "Климатическая норма (оценка)",
            "kz": "Климаттық норма (бағалау)",
        },
    }
    payload["source_label"] = names.get(source, names["climate_norm"])[
        "kz" if lang == "kz" else "ru"
    ]
    return payload


def _verify_series(
    *,
    dates: list[date],
    raw_values: list[float | None],
    measured: list[bool],
    valid_fraction: list[float],
    crop: str,
    sowing_date: date,
    center: tuple[float, float],
    lang: str,
    field_info: dict,
    source: str,
    allow_network: bool | None = None,
) -> dict:
    reference_values = scoring.profile_values(crop)
    if reference_values is None:
        raise _error(lang, "Неизвестная культура", "Белгісіз дақыл")
    clean_values = features.interpolate_series(raw_values)
    weather_summary = weather.get_weather(
        center[0], center[1], dates[0], dates[-1], allow_network=allow_network
    )
    feat = features.build_features(
        dates, raw_values, reference_values, sowing_date, weather_summary, measured
    )
    reference_peak = scoring.reference_peak(crop)
    score, breakdown = scoring.calculate_score(feat, reference_peak, sowing_date)
    ml_result = model.predict_crop(clean_values)
    verdict, reason_code = scoring.decide_verdict(
        feat, score, ml_result, crop, reference_peak
    )
    explanation, explanation_source = explain.build_explanation(
        verdict,
        score,
        feat,
        weather_summary,
        crop,
        ml_result,
        lang,
        reason_code,
    )
    reference = scoring.reference_series(crop, dates)
    reference_values = [
        float(point["ndvi"]) for point in reference
    ]
    predicted = ml_result.get("predicted")
    source_label = _source_label(source, lang)
    source_note = _source_note(source, lang)
    if source == "synthetic" and field_info.get("synthetic_reason") == "service_unavailable":
        unavailable = SYNTHETIC_UNAVAILABLE["kz" if lang == "kz" else "ru"]
        source_label = unavailable["label"]
        source_note = unavailable["note"]
    field = {
        **field_info,
        "crop_declared": crop,
        "crop_declared_label": crop_label(crop, lang),
        "sowing_date": sowing_date.isoformat(),
        "center": {"lat": round(center[0], 5), "lon": round(center[1], 5)},
        "source": source,
    }
    return {
        "demo_mode": bool(field_info.get("demo")),
        "lang": lang,
        "source": source,
        "source_label": source_label,
        "data_source_note": source_note,
        "field": field,
        "series": [
            {
                "date": d.isoformat(),
                "ndvi": round(float(v), 3),
                "measured": bool(flag),
                "valid_fraction": round(float(fraction), 3),
                "stage": features.temporal_stage(
                    dates, clean_values, index, feat, measured=measured
                ),
                "ndvi_color": features.temporal_color(v),
                "deviation_color": features.temporal_color(
                    float(v) - reference_values[index], mode="deviation"
                )
                if index < len(reference_values)
                else features.temporal_color(None, mode="deviation"),
            }
            for index, (d, v, flag, fraction) in enumerate(
                zip(dates, clean_values, measured, valid_fraction)
            )
        ],
        "ndvi_raw": [None if value is None else round(float(value), 3) for value in raw_values],
        "measured": [bool(flag) for flag in measured],
        "valid_fraction": [round(float(value), 3) for value in valid_fraction],
        "reference": reference,
        "temporal_palettes": {
            "ndvi": list(features.TEMPORAL_NDVI_COLORS),
            "deviation": list(features.TEMPORAL_DEVIATION_COLORS),
        },
        "features": feat,
        "score": score,
        "score_label": (
            "Өтінімге сәйкестік" if lang == "kz" else "Соответствие заявке"
        ),
        "score_components": breakdown,
        "verdict": verdict,
        "verdict_label": VERDICT_LABELS[lang].get(verdict, verdict),
        "verdict_reason_code": reason_code,
        "ml": {
            "predicted_crop": predicted,
            "predicted_label": crop_label(predicted, lang) if predicted else None,
            "probability": ml_result.get("probability"),
            "source": ml_result.get("source"),
            "confidence_label": (
                "Модель сенімділігі" if lang == "kz" else "Уверенность модели"
            ),
        },
        "top3": ml_result.get("top3", []),
        "probabilities": ml_result.get("probabilities", {}),
        "weather": _weather_public(weather_summary, lang),
        "explanation": explanation,
        "explanation_source": explanation_source,
        "meta": {
            "data_source": source,
            "synthetic": source == "synthetic",
            "season_year": dates[0].year,
            "model_dataset": "synthetic",
            "measured_points": int(sum(1 for flag in measured if flag)),
            "total_points": len(dates),
            "min_valid_fraction": config.CDSE_MIN_VALID_FRACTION,
            "disclaimer_ru": DISCLAIMER["ru"],
            "disclaimer_kz": DISCLAIMER["kz"],
        },
    }


def _verify_request(payload: VerifyRequest) -> dict:
    lang = payload.lang
    crop = normalize_crop(payload.crop)
    if not crop:
        raise _error(
            lang,
            "Культура должна быть одной из: wheat, barley, sunflower, rapso",
            "Дақыл wheat, barley, sunflower, rapso ішінен болуы керек",
        )
    if payload.demo_id:
        demo = _find_demo(payload.demo_id)
        if not demo:
            raise _error(lang, "Демо-поле не найдено", "Демо-алқап табылмады", 404)
        source = demo.get("source", "synthetic")
        dates = [date.fromisoformat(value) for value in demo["dates"]]
        raw = list(demo["ndvi_raw"])
        measured = list(demo["measured"])
        fractions = list(demo["valid_fraction"])
        field_info = {
            "id": demo["id"],
            "name": demo.get(f"name_{'kz' if lang == 'kz' else 'ru'}"),
            "district": demo.get(f"district_{'kz' if lang == 'kz' else 'ru'}"),
            "area_ha": demo.get("area_ha"),
            "scenario": demo.get("scenario"),
            "scenario_label": demo.get(
                f"scenario_label_{'kz' if lang == 'kz' else 'ru'}"
            ),
            "honesty_note": demo.get(f"honesty_note_{'kz' if lang == 'kz' else 'ru'}"),
            "source_detail": demo.get(f"source_note_{'kz' if lang == 'kz' else 'ru'}"),
            "demo": True,
        }
        return _verify_series(
            dates=dates,
            raw_values=raw,
            measured=measured,
            valid_fraction=fractions,
            crop=crop,
            sowing_date=payload.sowing_date,
            center=(demo["center"]["lat"], demo["center"]["lon"]),
            lang=lang,
            field_info=field_info,
            source=source,
        )

    if not payload.geojson:
        raise _error(
            lang,
            "Выберите демо-поле или нарисуйте контур на карте",
            "Демо-алқапты таңдаңыз немесе картада контур сызыңыз",
            422,
        )
    try:
        geometry = geo.extract_geometry(payload.geojson)
        _rings, area_ha, (lat, lon) = _validate_drawn(lang, geometry)
    except HTTPException:
        raise
    except Exception as exc:
        raise _error(
            lang,
            f"Некорректный контур: {exc}",
            f"Контур дұрыс емес: {exc}",
            422,
        ) from exc

    source = "sentinel-2-l2a"
    live_note = ""
    try:
        live = cdse.statistics_for_geometry(
            geometry,
            timeout=config.CDSE_DRAWN_TIMEOUT,
            use_cache=True,
        )
        dates = [date.fromisoformat(value) for value in live["dates"]]
        raw = list(live["ndvi_raw"])
        measured = list(live["measured"])
        fractions = list(live["valid_fraction"])
        live_note = (
            "Живой запрос CDSE выполнен."
            if lang == "ru"
            else "CDSE-ге тірі сұраныс орындалды."
        )
    except Exception as exc:
        # Never fail the demo into a blank screen: the fallback is explicit.
        logger.info("CDSE unavailable for drawn field: %s", type(exc).__name__)
        source = "synthetic"
        generated = synthetic.synthetic_series_for_field(
            lat, lon, crop, payload.sowing_date
        )
        dates = list(generated["dates"])
        raw = [
            float(value) if flag else None
            for value, flag in zip(generated["values"], generated["measured"])
        ]
        measured = list(generated["measured"])
        fractions = list(generated["valid_fraction"])
        live_note = (
            "CDSE не ответил; показан синтетический ряд."
            if lang == "ru"
            else "CDSE жауап бермеді; синтетикалық қатар көрсетілді."
        )

    field_info = {
        "id": "drawn-field",
        "name": DRAW_NAME[lang],
        "district": None,
        "area_ha": area_ha,
        "honesty_note": (
            "Контур нарисован пользователем; заявленная культура и дата сева условны."
            if lang == "ru"
            else "Контурды пайдаланушы сызды; мәлімделген дақыл мен себу күні шартты."
        ),
        "source_detail": live_note,
        "synthetic_reason": "service_unavailable" if source == "synthetic" else None,
        "demo": False,
    }
    if not _is_in_north_kazakhstan(lat, lon):
        field_info["region_warning"] = True
    return _verify_series(
        dates=dates,
        raw_values=raw,
        measured=measured,
        valid_fraction=fractions,
        crop=crop,
        sowing_date=payload.sowing_date,
        center=(lat, lon),
        lang=lang,
        field_info=field_info,
        source=source,
    )


@app.get("/api/health")
def health() -> dict:
    info = model.model_info()
    real = sum(1 for field in DEMO_FIELDS if field.get("source") == "sentinel-2-l2a")
    return {
        "status": "ok",
        "service": "agro-verifier",
        "version": VERSION,
        "data_mode": "sentinel-2-l2a" if real else "synthetic",
        "demo_fields": len(DEMO_FIELDS),
        "real_demo_fields": real,
        "model": info,
        "credentials_configured": config.cdse_credentials() is not None,
        "network_enabled": not config.network_disabled(),
        "llm_configured": chat.is_configured(),
    }


@app.get("/api/chat/config")
def chat_config(session_id: str | None = None, lang: Literal["ru", "kz"] = "ru") -> dict:
    if session_id:
        try:
            uuid.UUID(session_id)
        except ValueError as exc:
            raise _error("ru", "Сессия чата недействительна.", "Чат сессиясы жарамсыз.", 400) from exc
    else:
        session_id = str(uuid.uuid4())
    chat.start_session(session_id)
    configured = chat.is_configured(lang=lang)
    return {
        "enabled": True,
        "free_text_enabled": configured,
        "session_id": session_id,
    }


@app.post("/api/chat")
def chat_message(payload: chat.ChatRequest) -> dict:
    if not payload.valid_session_id():
        raise _error(payload.lang, "Сессия чата недействительна.", "Чат сессиясы жарамсыз.", 400)
    try:
        chat.validate_session(payload.session_id)
    except ValueError as exc:
        raise _error(payload.lang, "Сессия чата не найдена. Обновите страницу.", "Чат сессиясы табылмады. Бетті жаңартыңыз.", 400) from exc
    try:
        return chat.respond(payload)
    except ConnectionError as exc:
        raise _error(
            payload.lang,
            "ИИ-помощник недоступен. Запустите модель в LM Studio или выберите вопрос-подсказку.",
            "ИИ-көмекші қолжетімсіз. LM Studio моделін іске қосыңыз немесе ұсынылған сұрақты таңдаңыз.",
            503,
        ) from exc


@app.get("/api/config")
def public_config() -> dict:
    return config.public_config()


@app.get("/api/demo-fields")
def demo_fields(lang: str = "ru") -> dict:
    lang = "kz" if lang == "kz" else "ru"
    real = sum(1 for field in DEMO_FIELDS if field.get("source") == "sentinel-2-l2a")
    return {
        "fields": [_localized_demo(field, lang) for field in DEMO_FIELDS],
        "meta": {
            "count": len(DEMO_FIELDS),
            "real_fields": real,
            "synthetic_fields": len(DEMO_FIELDS) - real,
            "synthetic": real != len(DEMO_FIELDS),
            "source": "sentinel-2-l2a" if real else "synthetic",
            "note": (
                "Контур и заявка условные. Истинная культура реальных полей неизвестна; "
                "контуры не привязаны к хозяйствам."
                if lang == "ru"
                else "Контур мен өтінім шартты. Нақты алқаптардың шынайы дақылы белгісіз; "
                "контурлар шаруашылықтарға байланыстырылмаған."
            ),
        },
    }


@app.get("/api/geocode")
def geocode(q: str = "", lang: str = "ru", limit: int = 5) -> dict:
    lang = "kz" if lang == "kz" else "ru"
    try:
        results = geo.geocode(q, limit=limit, lang=lang)
    except geo.GeoError as exc:
        message = str(exc)
        if message == "Network is disabled":
            raise _error(
                lang,
                "Поиск места недоступен без интернета.",
                "Интернетсіз орын іздеу қолжетімсіз.",
                503,
            ) from exc
        raise _error(
            lang,
            "Не удалось найти место. Проверьте запрос и попробуйте ещё раз.",
            "Орынды табу мүмкін болмады. Сұранысты тексеріп, қайталаңыз.",
            502,
        ) from exc
    return {"results": results, "query": q}


@app.post("/api/verify")
def verify(payload: VerifyRequest) -> dict:
    try:
        return _verify_request(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Verification failed")
        raise _error(
            payload.lang,
            f"Ошибка проверки: {exc}",
            f"Тексеру қатесі: {exc}",
            500,
        ) from exc


def _decode_csv(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Cannot decode CSV; use UTF-8")


def _safe_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip().replace(",", "."))


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Invalid date: {value}")


def _normalize_batch_rows(raw_rows: list[dict], lang: str) -> list[dict]:
    if len(raw_rows) > MAX_BATCH_ROWS:
        raise _error(
            lang,
            f"Максимум {MAX_BATCH_ROWS} полей за один раз",
            f"Бір ретте ең көбі {MAX_BATCH_ROWS} алқап",
            413,
        )
    rows = []
    for index, raw in enumerate(raw_rows):
        clean = {
            str(key).strip().lower(): value
            for key, value in raw.items()
            if key is not None
        }
        row_id = str(clean.get("id") or f"row-{index + 1}").strip()
        crop = normalize_crop(clean.get("crop"))
        if not crop:
            raise _error(
                lang,
                f"Строка {index + 1}: неизвестная культура '{clean.get('crop')}'",
                f"{index + 1}-жол: белгісіз дақыл '{clean.get('crop')}'",
                422,
            )
        try:
            sowing_date = _parse_date(clean.get("sowing_date"))
            lat = _safe_float(clean.get("lat"))
            lon = _safe_float(clean.get("lon"))
        except Exception as exc:
            raise _error(
                lang,
                f"Строка {index + 1}: проверьте sowing_date, lat, lon ({exc})",
                f"{index + 1}-жол: sowing_date, lat, lon мәндерін тексеріңіз ({exc})",
                422,
            ) from exc
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise _error(
                lang,
                f"Строка {index + 1}: некорректные координаты",
                f"{index + 1}-жол: координаталар дұрыс емес",
                422,
            )
        scenario = str(clean.get("scenario") or "").strip().lower() or None
        if scenario not in {name for name, _weight in synthetic.SCENARIO_WEIGHTS}:
            scenario = None
        rows.append(
            {
                "id": row_id,
                "crop": crop,
                "sowing_date": sowing_date,
                "lat": lat,
                "lon": lon,
                "scenario": scenario,
            }
        )
    return rows


async def _parse_batch_request(request: Request) -> tuple[list[dict], str]:
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        lang = "kz" if str(form.get("lang", "ru")) == "kz" else "ru"
        if upload is None:
            raise _error(lang, "CSV-файл не приложен", "CSV файлы қосылмаған", 400)
        data = await upload.read()
        if len(data) > 2_000_000:
            raise _error(
                lang,
                "Файл слишком большой (макс. 2 МБ)",
                "Файл тым үлкен (макс. 2 МБ)",
                413,
            )
        try:
            text = _decode_csv(data)
            sample = text[:4096]
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","
            raw_rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
        except Exception as exc:
            raise _error(
                lang,
                f"Не удалось прочитать CSV: {exc}",
                f"CSV оқылмады: {exc}",
                422,
            ) from exc
        if not raw_rows:
            raise _error(lang, "CSV пуст", "CSV бос", 422)
        return _normalize_batch_rows(raw_rows, lang), lang

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as exc:
            raise _error(
                "ru",
                f"Некорректный JSON: {exc}",
                f"JSON дұрыс емес: {exc}",
                422,
            ) from exc
        lang = "ru"
        if isinstance(body, list):
            raw_rows = body
        elif isinstance(body, dict):
            lang = "kz" if body.get("lang") == "kz" else "ru"
            raw_rows = body.get("rows") or body.get("fields") or []
        else:
            raw_rows = []
        if not raw_rows:
            raise _error(lang, "В JSON нет массива rows", "JSON-да rows массиві жоқ", 422)
        return _normalize_batch_rows(raw_rows, lang), lang

    raise _error(
        "ru",
        "Ожидается multipart/form-data с CSV или application/json",
        "multipart/form-data CSV немесе application/json күтіледі",
        415,
    )


def _batch_row(row: dict, lang: str) -> dict:
    crop = row["crop"]
    generated = synthetic.synthetic_series_for_field(
        row["lat"],
        row["lon"],
        crop,
        row["sowing_date"],
        scenario=row.get("scenario"),
    )
    result = _verify_series(
        dates=list(generated["dates"]),
        raw_values=[
            float(value) if flag else None
            for value, flag in zip(generated["values"], generated["measured"])
        ],
        measured=list(generated["measured"]),
        valid_fraction=list(generated["valid_fraction"]),
        crop=crop,
        sowing_date=row["sowing_date"],
        center=(row["lat"], row["lon"]),
        lang=lang,
        field_info={
            "id": row["id"],
            "name": row["id"],
            "area_ha": None,
            "source_detail": (
                "Синтетический ряд для демонстрации портфеля."
                if lang == "ru"
                else "Портфельді көрсетуге арналған синтетикалық қатар."
            ),
            "demo": False,
        },
        source="synthetic",
        allow_network=False,
    )
    summary = result["explanation"]["summary"]
    return {
        "id": row["id"],
        "crop": crop,
        "crop_label": crop_label(crop, lang),
        "sowing_date": row["sowing_date"].isoformat(),
        "lat": row["lat"],
        "lon": row["lon"],
        "score": result["score"],
        "score_label": result["score_label"],
        "verdict": result["verdict"],
        "verdict_label": result["verdict_label"],
        "reason_code": result["verdict_reason_code"],
        "recommendation": result["explanation"]["recommendation"],
        "summary": summary,
        "ml_predicted": result["ml"]["predicted_crop"],
        "ml_predicted_label": result["ml"]["predicted_label"],
        "ml_probability": result["ml"]["probability"],
        "top3": result["top3"],
        "source": "synthetic",
        "source_label": _source_label("synthetic", lang),
        "risk_rank": RISK_ORDER.get(result["verdict"], 9),
    }


@app.post("/api/batch")
async def batch(request: Request) -> dict:
    rows, lang = await _parse_batch_request(request)
    results = [_batch_row(row, lang) for row in rows]
    results.sort(key=lambda item: (item["risk_rank"], item["score"]))
    counts = {key: 0 for key in RISK_ORDER}
    for item in results:
        counts[item["verdict"]] = counts.get(item["verdict"], 0) + 1
    return {
        "count": len(results),
        "rows": results,
        "counts": counts,
        "lang": lang,
        "demo_mode": True,
        "source": "synthetic",
        "source_label": _source_label("synthetic", lang),
        "meta": {
            "synthetic": True,
            "weather": (
                "климатическая норма или кэш; batch не обращается к сети"
                if lang == "ru"
                else "климаттық норма немесе кэш; batch желіге шықпайды"
            ),
            "note": (
                "Сортировка по риску: несоответствие → нет посева → проверка → сомнительно → подтверждено. "
                "Ряды портфеля синтетические."
                if lang == "ru"
                else "Тәуекел бойынша сұрыптау: сәйкессіздік → егіс жоқ → тексеру → күмәнді → расталды. "
                "Портфель қатарлары синтетикалық."
            ),
        },
    }


SAMPLE_ROWS: list[dict] = [
    {"id": "AKM-001", "crop": "wheat", "sowing_date": "2026-05-15", "lat": 51.88, "lon": 70.10, "scenario": "normal"},
    {"id": "AKM-002", "crop": "wheat", "sowing_date": "2026-05-18", "lat": 52.10, "lon": 70.60, "scenario": "normal"},
    {"id": "AKM-003", "crop": "barley", "sowing_date": "2026-05-06", "lat": 51.55, "lon": 71.40, "scenario": "normal"},
    {"id": "AKM-004", "crop": "barley", "sowing_date": "2026-05-08", "lat": 52.30, "lon": 72.10, "scenario": "early_stress"},
    {"id": "AKM-005", "crop": "sunflower", "sowing_date": "2026-05-20", "lat": 51.70, "lon": 69.60, "scenario": "normal"},
    {"id": "AKM-006", "crop": "sunflower", "sowing_date": "2026-05-22", "lat": 52.70, "lon": 70.20, "scenario": "wrong_crop"},
    {"id": "AKM-007", "crop": "rapso", "sowing_date": "2026-05-01", "lat": 51.95, "lon": 71.90, "scenario": "normal"},
    {"id": "AKM-008", "crop": "wheat", "sowing_date": "2026-05-12", "lat": 52.62, "lon": 69.10, "scenario": "late_low"},
    {"id": "AKM-009", "crop": "wheat", "sowing_date": "2026-05-17", "lat": 51.30, "lon": 70.80, "scenario": "bare"},
    {"id": "AKM-010", "crop": "wheat", "sowing_date": "2026-05-15", "lat": 52.45, "lon": 71.50, "scenario": "double_wave"},
    {"id": "AKM-011", "crop": "barley", "sowing_date": "2026-05-04", "lat": 51.78, "lon": 72.40, "scenario": "normal"},
    {"id": "AKM-012", "crop": "sunflower", "sowing_date": "2026-05-19", "lat": 52.05, "lon": 69.80, "scenario": "late_low"},
    {"id": "AKM-013", "crop": "rapso", "sowing_date": "2026-05-02", "lat": 51.48, "lon": 69.15, "scenario": "wrong_crop"},
    {"id": "AKM-014", "crop": "wheat", "sowing_date": "2026-05-14", "lat": 52.80, "lon": 71.20, "scenario": "normal"},
    {"id": "AKM-015", "crop": "wheat", "sowing_date": "2026-05-20", "lat": 51.10, "lon": 71.70, "scenario": "normal"},
    {"id": "AKM-016", "crop": "barley", "sowing_date": "2026-05-07", "lat": 52.20, "lon": 68.90, "scenario": "early_stress"},
    {"id": "AKM-017", "crop": "sunflower", "sowing_date": "2026-05-21", "lat": 51.60, "lon": 70.30, "scenario": "normal"},
    {"id": "AKM-018", "crop": "wheat", "sowing_date": "2026-05-16", "lat": 52.90, "lon": 70.90, "scenario": "late_low"},
    {"id": "AKM-019", "crop": "rapso", "sowing_date": "2026-05-03", "lat": 51.35, "lon": 72.60, "scenario": "bare"},
    {"id": "AKM-020", "crop": "wheat", "sowing_date": "2026-05-15", "lat": 52.15, "lon": 69.40, "scenario": "double_wave"},
]


@app.get("/api/sample-batch")
def sample_batch(lang: str = "ru", format: str = "json"):
    lang = "kz" if lang == "kz" else "ru"
    if format.lower() == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=["id", "crop", "sowing_date", "lat", "lon", "scenario"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(SAMPLE_ROWS)
        return Response(
            content="\ufeff" + buffer.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="agro-portfolio-template.csv"'
            },
        )
    return {
        "rows": [
            {**row, "crop_label": crop_label(row["crop"], lang)} for row in SAMPLE_ROWS
        ],
        "count": len(SAMPLE_ROWS),
        "source": "synthetic",
        "source_label": _source_label("synthetic", lang),
        "meta": {
            "synthetic": True,
            "note": (
                "20 демонстрационных полей; ряды синтетические и явно помечены."
                if lang == "ru"
                else "20 демонстрациялық алқап; қатарлар синтетикалық және нақты белгіленген."
            ),
        },
    }


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
