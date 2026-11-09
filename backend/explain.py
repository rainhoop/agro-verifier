"""Template explanations (offline default) and optional LLM narration."""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
from typing import Any

import httpx

from .features import CROP_LABELS, crop_label
from .weather import NORM_PRECIPITATION_MM

logger = logging.getLogger(__name__)

_ALLOWED_LATIN_TERMS = {
    "ndvi",
    "sentinel-2",
    "l2a",
    "scl",
    "ml",
    "randomforest",
    "api",
    "pdf",
    "csv",
}
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі0-9-]+")
_TABLE_ROW_RE = re.compile(r"(?m)^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(
    r"(?m)^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_LATEX_RE = re.compile(
    r"\$[^$]+\$|\\\(|\\\)|\\\[|\\\]|\\(?:begin\{|frac\b|sqrt\b|sum\b|int\b|alpha\b|beta\b|gamma\b|cdot\b|times\b)"
)


class ResponseQualityError(ValueError):
    """Raised when generated text is not suitable for display."""


def validate_llm_text(text: str, max_chars: int) -> None:
    if not text.strip():
        raise ResponseQualityError("empty response text")
    if len(text.strip()) > max_chars:
        raise ResponseQualityError("response exceeds length limit")
    if _TABLE_ROW_RE.search(text) or _TABLE_SEPARATOR_RE.search(text):
        raise ResponseQualityError("markdown tables are not allowed")
    if _LATEX_RE.search(text):
        raise ResponseQualityError("LaTeX is not allowed")
    for word in _WORD_RE.findall(text):
        has_latin = re.search(r"[A-Za-z]", word)
        if has_latin and word.lower() not in _ALLOWED_LATIN_TERMS:
            raise ResponseQualityError("unapproved Latin text")


def _fmt_date(value: str | None) -> str:
    if not value:
        return "—"
    try:
        return date.fromisoformat(value).strftime("%d.%m.%Y")
    except ValueError:
        return value


def _days_ru(value: int) -> str:
    n = abs(int(value)) % 100
    if 11 <= n <= 19:
        return "дней"
    n %= 10
    if n == 1:
        return "день"
    if 2 <= n <= 4:
        return "дня"
    return "дней"


def _weather_context(weather: dict, lang: str) -> str:
    precip = weather.get("precipitation_mm", 0)
    norm = weather.get("norm_precipitation_mm", NORM_PRECIPITATION_MM)
    index = weather.get("precipitation_index", 0)
    temp = weather.get("temperature_mean_c", 0)
    source = weather.get("source", "climate_norm")
    if lang == "kz":
        parts = [
            f"Маусымдық жауын-шашын: {precip:.0f} мм ({index:.0f}% норма, {norm:.0f} мм).",
            f"Орташа температура: {temp:.1f} °C.",
        ]
        if index < 70:
            parts.append("Маусым құрғақ болды.")
        elif index > 120:
            parts.append("Маусым ылғалды болды.")
        if source == "open-meteo":
            suffix = " (кэштен)" if weather.get("cached") else ""
            parts.append(f"Дереккөз: Open-Meteo, 2026 тарихи деректері{suffix}.")
        else:
            parts.append("Дереккөз: Солтүстік Қазақстанның климаттық нормасы (бағалау).")
        return " ".join(parts)
    parts = [
        f"Осадки за сезон: {precip:.0f} мм — {index:.0f}% климатической нормы ({norm:.0f} мм).",
        f"Средняя температура: {temp:.1f} °C.",
    ]
    if index < 70:
        parts.append("Сезон засушливый.")
    elif index > 120:
        parts.append("Сезон влажный.")
    if source == "open-meteo":
        suffix = " (из кэша)" if weather.get("cached") else ""
        parts.append(f"Источник: Open-Meteo, исторические данные 2026{suffix}.")
    else:
        parts.append("Источник: климатическая норма Северного Казахстана (оценка).")
    return " ".join(parts)


def _summary(
    verdict: str,
    score: int,
    features: dict,
    weather: dict,
    declared_crop: str,
    ml: dict,
    lang: str,
    reason_code: str | None = None,
) -> str:
    crop = crop_label(declared_crop, lang)
    peak = float(features.get("peak_ndvi") or 0)
    corr = float(features.get("correlation_with_reference") or 0)
    index = float(weather.get("precipitation_index") or 100)
    predicted = ml.get("predicted", "")
    prob = float(ml.get("probability") or 0)
    diff = features.get("sowing_shift_days")
    if lang == "kz":
        if reason_code == "flat_canopy":
            return (
                f"NDVI қисығы барлық маусымда дерлік тегіс (максимум {peak:.2f}). "
                "Бұл егіс емес, пар немесе нашар өскен алқап болуы мүмкін."
            )
        if reason_code == "low_peak":
            return (
                f"Пішін ішінара сәйкеседі, бірақ ең жоғары NDVI {peak:.2f} эталоннан айтарлықтай төмен. "
                "Өтінімге сәйкестік жоғары болғанымен, қорытындыны растауға болмайды."
            )
        if verdict == "confirmed":
            return (
                f"{crop} егісі расталады: NDVI пішіні эталонға жақын (r={corr:.2f}), "
                f"пик {peak:.2f}. Өтінімге сәйкестік {score}/100."
            )
        if verdict == "doubtful":
            if index < 70:
                return (
                    f"Өсу белгілері бар, бірақ құрғақ маусымда (норманың {index:.0f}% жауын) "
                    "пик төмен. Бұл міндетті түрде бұрмалау емес — құжаттарды сұрап, нақтылау қажет."
                )
            if diff is not None and abs(diff) > 20:
                return (
                    f"Өсу мәлімделген себу күнінен {abs(diff)} күнге ауытқыды, "
                    f"нысан пішіні эталонмен ішінара ғана сәйкеседі. Өтінімге сәйкестік {score}/100."
                )
            return f"Өсу бар, бірақ {crop} эталонымен сәйкестік ішінара. Өтінімге сәйкестік {score}/100."
        if verdict == "not_found":
            return (
                f"Контурда тұрақты NDVI өсуі табылмады (максимум {peak:.2f}). "
                "Бұл егіс жоқ немесе деректер жеткіліксіз дегенді білдіреді."
            )
        if verdict == "crop_mismatch":
            return (
                f"Қатар пішіні «{crop}» емес, «{crop_label(predicted, lang)}» дақылына жақын "
                f"(ML сенімділігі {prob*100:.0f}%). Өтінімдегі дақылды тексеру қажет."
            )
        return (
            "Маусымда екі өсу толқыны байқалады: қайта себу, алқаптың біртексіздігі немесе "
            "бұлттық артефактілер болуы мүмкін. Автоматты қорытынды жеткіліксіз."
        )
    if reason_code == "flat_canopy":
        return (
            f"Кривая NDVI почти не поднимается над фоном за весь сезон (максимум {peak:.2f}). "
            "Это похоже на пар, залежь или поле без сформировавшегося посева."
        )
    if reason_code == "low_peak":
        return (
            f"Форма кривой частично совпадает с эталоном, но максимум NDVI {peak:.2f} существенно ниже "
            "обычного для заявленной культуры. Высокое соответствие по форме не позволяет подтвердить посев."
        )
    if verdict == "confirmed":
        return (
            f"Посев {crop.lower()} подтверждается: форма NDVI близка к эталону (r={corr:.2f}), "
            f"пик {peak:.2f}. Соответствие заявке {score}/100."
        )
    if verdict == "doubtful":
        if index < 70:
            return (
                f"Признаки роста есть, но на фоне засушливого сезона ({index:.0f}% нормы осадков) "
                "пик ниже обычного. Это не обязательно искажение — запросите документы и уточните причину."
            )
        if diff is not None and abs(diff) > 20:
            return (
                f"Начало роста отличается от заявленного сева на {abs(diff)} дн., а форма кривой "
                f"лишь частично совпадает с эталоном. Соответствие заявке {score}/100."
            )
        return (
            f"Рост есть, но кривая лишь частично соответствует эталону «{crop}». "
            f"Соответствие заявке {score}/100 — нужна проверка документов."
        )
    if verdict == "not_found":
        return (
            f"На контуре не обнаружен устойчивый рост NDVI (максимум {peak:.2f}). "
            "Признаки заявленного посева отсутствуют или данных недостаточно."
        )
    if verdict == "crop_mismatch":
        return (
            f"Форма ряда ближе к культуре «{crop_label(predicted, lang)}», чем к заявленной "
            f"«{crop}» (уверенность модели {prob*100:.0f}%). Возможна ошибка в заявке или другая культура на поле."
        )
    return (
        "В сезоне видны две волны роста: возможен пересев, неоднородность поля или облачные "
        "артефакты. Автоматического вывода недостаточно, нужна ручная проверка."
    )


def _reasons(
    verdict: str,
    features: dict,
    weather: dict,
    declared_crop: str,
    ml: dict,
    lang: str,
    reason_code: str | None = None,
) -> list[str]:
    reasons: list[str] = []
    crop = crop_label(declared_crop, lang)
    predicted = ml.get("predicted", "")
    prob = float(ml.get("probability") or 0)
    peak = float(features.get("peak_ndvi") or 0)
    ref_peak = float(features.get("reference_peak_ndvi") or 0)
    corr = float(features.get("correlation_with_reference") or 0)
    growth = features.get("growth_start_date")
    harvest = features.get("harvest_date")
    duration = int(features.get("vegetation_days") or 0)
    diff = features.get("sowing_shift_days")
    index = float(weather.get("precipitation_index") or 0)

    if lang == "kz":
        if reason_code == "flat_canopy":
            reasons.append(f"Маусымдық максимум {peak:.2f}: қисық егіс белгісін көрсетпейді.")
        if reason_code == "low_peak":
            reasons.append(
                f"Пик {peak:.2f} эталондық {ref_peak:.2f} деңгейінен төмен; бұл растауға кедергі."
            )
        if growth:
            harvest_text = f", жинау {_fmt_date(harvest)}" if harvest else ""
            reasons.append(
                f"NDVI өсуі {_fmt_date(growth)} басталды, вегетация шамамен {duration} күн{harvest_text}."
            )
        else:
            reasons.append("NDVI қисығы 0.25 шегінен аспайды — тұрақты өсу жоқ.")
        if peak < 0.30:
            reasons.append(f"Маусымдық максимум {peak:.2f} — 0.30 шегінен төмен.")
        elif ref_peak:
            ratio = peak / ref_peak * 100 if ref_peak else 0
            if ratio < 70:
                reasons.append(f"Пик {peak:.2f} эталоннан {100-ratio:.0f}% төмен (эталон {ref_peak:.2f}).")
            elif ratio >= 90:
                reasons.append(f"Пик {peak:.2f} эталонға жақын ({ref_peak:.2f}).")
            else:
                reasons.append(f"Пик {peak:.2f} эталоннан сәл төмен ({ref_peak:.2f}).")
        if corr >= 0.75:
            reasons.append(f"Қисық пішіні «{crop}» эталонымен жақсы сәйкеседі (r={corr:.2f}).")
        elif corr >= 0.40:
            reasons.append(f"Эталонмен орташа сәйкестік (r={corr:.2f}); фенологияның бір бөлігі ерекшеленеді.")
        else:
            reasons.append(f"Пішін сәйкестігі әлсіз (r={corr:.2f}); қисық эталонға ұқсамайды.")
        if diff is not None:
            if abs(diff) <= 15:
                reasons.append(f"Өсу басталуы мәлімделген себу күнінен {abs(diff)} күн шегінде.")
            elif diff > 0:
                reasons.append(f"Өсу мәлімделген себу күнінен {diff} күн кеш басталды.")
            else:
                reasons.append(f"Өсу мәлімделген себу күнінен {abs(diff)} күн ерте басталды.")
        if predicted == "double_wave" or features.get("two_wave_detected"):
            reasons.append(f"Қатарда екі айқын максимум бар (ML сенімділігі {prob*100:.0f}%).")
        elif predicted == declared_crop:
            reasons.append(f"ML-модель «{crop}» дақылын {prob*100:.0f}% сенімділікпен растайды.")
        elif predicted:
            reasons.append(
                f"ML-модель қатарды «{crop_label(predicted, lang)}» деп {prob*100:.0f}% сенімділікпен жіктейді."
            )
        if index < 70 and peak < (ref_peak * 0.85 if ref_peak else 0.6):
            reasons.append(f"Жауын-шашын норманың {index:.0f}% — құрғақшылық NDVI деңгейін төмендетуі мүмкін.")
        elif index > 120:
            reasons.append(f"Жауын-шашын норманың {index:.0f}% — ылғалдандыру нормадан жоғары.")
        return reasons

    if reason_code == "flat_canopy":
        reasons.append(f"Максимум сезона {peak:.2f}: кривая не показывает признаков сформированного посева.")
    if reason_code == "low_peak":
        reasons.append(
            f"Пик {peak:.2f} ниже эталонного уровня {ref_peak:.2f}; этого недостаточно для подтверждения."
        )
    if growth:
        harvest_text = f", уборка {_fmt_date(harvest)}" if harvest else ""
        reasons.append(
            f"Рост NDVI начался {_fmt_date(growth)}, вегетация около {duration} {_days_ru(duration)}{harvest_text}."
        )
    else:
        reasons.append("Кривая NDVI не пересекает порог 0.25 — устойчивого роста нет.")
    if peak < 0.30:
        reasons.append(f"Максимум сезона {peak:.2f} — ниже порога 0.30.")
    elif ref_peak:
        ratio = peak / ref_peak * 100
        if ratio < 70:
            reasons.append(f"Пик {peak:.2f} на {100-ratio:.0f}% ниже эталона ({ref_peak:.2f}).")
        elif ratio >= 90:
            reasons.append(f"Пик {peak:.2f} близок к эталону ({ref_peak:.2f}).")
        else:
            reasons.append(f"Пик {peak:.2f} немного ниже эталона ({ref_peak:.2f}).")
    if corr >= 0.75:
        reasons.append(f"Форма кривой хорошо совпадает с эталоном «{crop}» (r={corr:.2f}).")
    elif corr >= 0.40:
        reasons.append(f"Умеренное сходство с эталоном (r={corr:.2f}); часть фенологии отличается.")
    else:
        reasons.append(f"Слабое сходство формы (r={corr:.2f}); кривая не похожа на эталон.")
    if diff is not None:
        if abs(diff) <= 15:
            reasons.append(f"Начало роста в пределах ±15 дней от заявленного сева (Δ={abs(diff)} дн.).")
        elif diff > 0:
            reasons.append(f"Рост начался на {diff} дн. позже заявленного сева.")
        else:
            reasons.append(f"Рост начался на {abs(diff)} дн. раньше заявленного сева.")
    if predicted == "double_wave" or features.get("two_wave_detected"):
        reasons.append(f"В ряду два выраженных локальных максимума (уверенность ML {prob*100:.0f}%).")
    elif predicted == declared_crop:
        reasons.append(f"ML-модель подтверждает «{crop}» с вероятностью {prob*100:.0f}%.")
    elif predicted:
        reasons.append(
            f"ML-модель относит ряд к «{crop_label(predicted, lang)}» с вероятностью {prob*100:.0f}%."
        )
    if index < 70 and peak < (ref_peak * 0.85 if ref_peak else 0.6):
        reasons.append(f"Осадков {index:.0f}% нормы — засуха могла снизить NDVI, это не обязательно искажение.")
    elif index > 120:
        reasons.append(f"Осадков {index:.0f}% нормы — увлажнение выше обычного.")
    return reasons


def _recommendation(
    verdict: str, features: dict, weather: dict, lang: str
) -> tuple[str, str]:
    index = float(weather.get("precipitation_index") or 100)
    if verdict == "confirmed":
        code = "approve"
    elif verdict == "doubtful":
        code = "request_docs"
    elif verdict == "crop_mismatch":
        code = "request_docs"
    elif verdict == "needs_check":
        code = "field_visit"
    else:
        code = "field_visit"
    if lang == "kz":
        texts = {
            "approve": "Мақұлдау: қашықтықтан тексеру сәйкессіздік белгілерін таппады. Есепті кредиттік іске сақтаңыз.",
            "request_docs": (
                "Құжаттарды сұрау: тұқым шарты, ЖЖМ/тыңайтқыш түбіртектері және себу актісі. "
                "Айырмашылық сақталса — маманды жерге жіберу."
            ),
            "field_visit": "Маманды жерге жіберу: қашықтықтан алынған деректер мәлімделген егісті растамайды немесе қолмен түсіндіруді қажет етеді.",
        }
        if code == "request_docs" and index < 70:
            texts["request_docs"] += " Құрғақ маусымды ескеріңіз: төмен NDVI бұрмалау емес, ауа райының салдары болуы мүмкін."
        return texts[code], code
    texts = {
        "approve": "Одобрить: дистанционная проверка не выявила признаков несоответствия. Сохраните отчёт в кредитном деле.",
        "request_docs": (
            "Запросить документы: договор на семена, чеки на ГСМ/удобрения и акт сева. "
            "При сохранении расхождений — выезд специалиста."
        ),
        "field_visit": "Выезд специалиста: дистанционные данные не подтверждают заявленный посев либо требуют ручной интерпретации.",
    }
    if code == "request_docs" and index < 70:
        texts["request_docs"] += " Учтите засушливый сезон: низкий NDVI может быть следствием погоды, а не искажением."
    return texts[code], code


def template_explanation(
    verdict: str,
    score: int,
    features: dict,
    weather: dict,
    declared_crop: str,
    ml: dict,
    lang: str = "ru",
    reason_code: str | None = None,
) -> dict:
    recommendation, code = _recommendation(verdict, features, weather, lang)
    return {
        "verdict": verdict,
        "confidence": score,
        "summary": _summary(verdict, score, features, weather, declared_crop, ml, lang, reason_code),
        "reasons": _reasons(verdict, features, weather, declared_crop, ml, lang, reason_code),
        "weather_context": _weather_context(weather, lang),
        "recommendation": recommendation,
        "recommendation_code": code,
    }


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ResponseQualityError("No valid JSON object in LLM response")


def _llm_explanation(
    verdict: str,
    score: int,
    features: dict,
    weather: dict,
    declared_crop: str,
    ml: dict,
    lang: str,
    reason_code: str | None = None,
    retry: bool = False,
) -> dict:
    from .chat import base_url_for_provider, model_for_lang

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    base_url = base_url_for_provider()
    url = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
    model = model_for_lang(lang)
    schema = {
        "summary": "2-4 предложения простым языком",
        "reasons": ["3-5 коротких причин на основе переданных чисел"],
        "weather_context": "1-2 предложения о погоде",
        "recommendation": "одна конкретная рекомендация аналитику",
    }
    payload_data = {
        "lang": lang,
        "crop": {"id": declared_crop, "name": crop_label(declared_crop, lang)},
        "verdict": verdict,
        "score": score,
        "features": features,
        "weather": {
            "precipitation_mm": weather.get("precipitation_mm"),
            "precipitation_index_pct": weather.get("precipitation_index"),
            "norm_precipitation_mm": weather.get("norm_precipitation_mm"),
            "temperature_mean_c": weather.get("temperature_mean_c"),
            "source": weather.get("source"),
        },
        "ml_top3": ml.get("top3", []),
    }
    system_prompt = (
        "Ты — помощник кредитного аналитика в агрокредитовании. Отвечай ТОЛЬКО валидным JSON "
        "без markdown. Пиши короткими простыми предложениями только на языке интерфейса. "
        "Не используй латиницу, кроме терминов NDVI, Sentinel-2, L2A, SCL, ML, "
        "RandomForest, API, PDF и CSV. Опирайся только на переданные числа; "
        "не выдумывай факты и даты. "
        "Не меняй вердикт и балл. Если данных не хватает, напиши «недостаточно данных». "
        f"Язык интерфейса: {'казахский' if lang == 'kz' else 'русский'}. "
        + ("Предыдущий ответ не прошёл проверку качества. Исправь его." if retry else "")
        + " Схема ответа: "
        + json.dumps(schema, ensure_ascii=False)
    )
    body = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 1000,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Данные проверки поля (только они являются источником фактов):\n"
                + json.dumps(payload_data, ensure_ascii=False),
            },
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key and not base_url.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]")):
        headers["Authorization"] = f"Bearer {api_key}"
    response = httpx.post(
        url,
        headers=headers,
        json=body,
        timeout=8.0,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    summary = parsed.get("summary")
    reasons = parsed.get("reasons")
    weather_context = parsed.get("weather_context")
    recommendation = parsed.get("recommendation")
    if not isinstance(summary, str):
        raise ResponseQualityError("Invalid LLM summary")
    if not summary.strip():
        raise ResponseQualityError("empty response text")
    if len(summary.strip()) < 10:
        raise ResponseQualityError("Invalid LLM summary")
    if not isinstance(reasons, list) or not reasons or not all(isinstance(x, str) and x.strip() for x in reasons):
        raise ResponseQualityError("Invalid LLM reasons")
    if not isinstance(weather_context, str) or not weather_context.strip():
        raise ResponseQualityError("Invalid LLM weather_context")
    if not isinstance(recommendation, str) or not recommendation.strip():
        raise ResponseQualityError("Invalid LLM recommendation")
    if not all(isinstance(item, str) for item in reasons):
        raise ResponseQualityError("Invalid LLM reasons")
    displayed_text = "\n".join(
        [summary, *reasons, weather_context, recommendation]
    )
    validate_llm_text(displayed_text, max_chars=900)
    _, code = _recommendation(verdict, features, weather, lang)
    return {
        "verdict": verdict,
        "confidence": score,
        "summary": summary.strip(),
        "reasons": [x.strip() for x in reasons][:6],
        "weather_context": weather_context.strip(),
        "recommendation": recommendation.strip(),
        "recommendation_code": code,
    }


def build_explanation(
    verdict: str,
    score: int,
    features: dict,
    weather: dict,
    declared_crop: str,
    ml: dict,
    lang: str = "ru",
    reason_code: str | None = None,
) -> tuple[dict, str]:
    """Return (explanation, source).  Never fails because of the optional LLM."""
    base = template_explanation(verdict, score, features, weather, declared_crop, ml, lang, reason_code)
    from .chat import is_configured

    if is_configured(lang=lang):
        for attempt in range(2):
            try:
                return _llm_explanation(
                    verdict,
                    score,
                    features,
                    weather,
                    declared_crop,
                    ml,
                    lang,
                    reason_code,
                    retry=attempt > 0,
                ), "llm"
            except ResponseQualityError as exc:
                logger.warning(
                    "LLM explanation rejected by quality checks (attempt %s): %s",
                    attempt + 1,
                    exc,
                )
                if attempt == 0:
                    continue
                break
            except Exception as exc:
                logger.warning("LLM explanation failed; using template: %s", exc)
                break
    return base, "template"
