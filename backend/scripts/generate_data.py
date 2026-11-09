"""Build reference profiles and train the synthetic-data crop classifier.

Run from the project root:
    python -m backend.scripts.generate_data

Real demo fields are fetched separately by ``backend.scripts.fetch_real_demo``;
this script never invents or overwrites real observations.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import config
from backend.features import CROP_IDS, CROP_LABELS, season_dates
from backend.model import MODEL_PATH, train_and_save
from backend.synthetic import (
    CROP_PROFILES,
    generate_bare_sample,
    generate_crop_sample,
    generate_two_wave_sample,
    profile_values,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "backend" / "data"
SEED = 42


def build_profiles() -> dict:
    dates = season_dates(config.SEASON_YEAR)
    descriptions = {
        "wheat": (
            "Яровая пшеница: сев в середине мая, пик около 10–20 июля, "
            "уборка в конце августа. Широкое плато допускает сдвиг фазы на 7–10 дней.",
            "Жаздық бидай: мамыр ортасында себу, шілде ортасында пик, тамыз соңында жинау.",
        ),
        "barley": (
            "Ячмень: ранний сев в начале мая, пик в конце июня — начале июля, "
            "уборка в начале августа.",
            "Арпа: мамыр басында ерте себу, маусым соңы — шілде басында пик, тамыз басында жинау.",
        ),
        "sunflower": (
            "Подсолнечник: поздний сев в середине мая, пик в конце июля — начале августа, "
            "уборка в сентябре.",
            "Күнбағыс: мамыр ортасында кеш себу, шілде соңы — тамыз басында пик, қыркүйекте жинау.",
        ),
        "rapso": (
            "Рапс: ранний сев, пик в конце июня — начале июля, уборка в августе.",
            "Рапс: ерте себу, маусым соңы — шілде басында пик, тамызда жинау.",
        ),
    }
    crops = {}
    for crop in CROP_IDS:
        p = CROP_PROFILES[crop]
        values = profile_values(crop)
        crops[crop] = {
            "name_ru": CROP_LABELS[crop]["ru"],
            "name_kz": CROP_LABELS[crop]["kz"],
            "sowing_date": p["sowing"].isoformat(),
            "peak_date": p["peak"].isoformat(),
            "peak_ndvi": p["peak_ndvi"],
            "harvest_window_ru": p["harvest"],
            "description_ru": descriptions[crop][0],
            "description_kz": descriptions[crop][1],
            "series": [
                {"date": d.isoformat(), "ndvi": round(float(v), 3)}
                for d, v in zip(dates, values)
            ],
        }
    return {
        "meta": {
            "season_year": config.SEASON_YEAR,
            "season_start": dates[0].isoformat(),
            "season_end": dates[-1].isoformat(),
            "step_days": config.STEP_DAYS,
            "profile_type": "литературно-калиброванная асимметричная гауссиана",
            "calibration": (
                "Профили заданы по обобщённым литературным фенодатам Северного Казахстана: "
                "яровая пшеница — сев 10–25 мая, пик 5–20 июля, уборка конец августа; "
                "ячмень — сев 1–10 мая, пик 20 июня–5 июля; подсолнечник — сев 15–25 мая, "
                "пик конец июля–начало августа; рапс — сев 25 апреля–5 мая, пик конец июня. "
                "Это априорная модель, а не статистика конкретных хозяйств."
            ),
            "honesty_note": (
                "Истинная культура реальных демо-контуров неизвестна; заявка в демо условна. "
                "Профили не валидированы на размеченных производственных полях."
            ),
        },
        "crops": crops,
    }


def build_training_set(rng: random.Random) -> tuple[list[np.ndarray], list[str]]:
    series: list[np.ndarray] = []
    labels: list[str] = []
    for crop in CROP_IDS:
        for _ in range(210):
            series.append(np.asarray(generate_crop_sample(crop, rng)["values"], dtype=float))
            labels.append(crop)
        for _ in range(60):
            series.append(
                np.asarray(generate_crop_sample(crop, rng, stress="late_low")["values"], dtype=float)
            )
            labels.append(crop)
        for _ in range(30):
            series.append(
                np.asarray(generate_crop_sample(crop, rng, stress="early_stress")["values"], dtype=float)
            )
            labels.append(crop)
    for _ in range(300):
        series.append(np.asarray(generate_bare_sample(rng)["values"], dtype=float))
        labels.append("bare")
    for _ in range(300):
        series.append(np.asarray(generate_two_wave_sample(rng)["values"], dtype=float))
        labels.append("double_wave")
    combined = list(zip(series, labels))
    rng.shuffle(combined)
    return [item[0] for item in combined], [item[1] for item in combined]


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print("1/3  Building literature-calibrated reference profiles ...")
    profiles = build_profiles()
    with (DATA_DIR / "profiles.json").open("w", encoding="utf-8") as fh:
        json.dump(profiles, fh, ensure_ascii=False, indent=2)

    print("2/3  Generating synthetic training curves ...")
    series, labels = build_training_set(rng)
    print(f"     samples: {len(series)} (synthetic only)")

    print("3/3  Training RandomForest and writing metrics ...")
    bundle = train_and_save(series, labels)
    metrics = bundle["metrics"]
    label_order = metrics["labels"]
    print(f"     hold-out accuracy (synthetic): {metrics['accuracy']:.4f}")
    print(f"     train/test: {metrics['n_train']}/{metrics['n_test']}")
    print("     confusion matrix (rows=true, cols=pred):")
    print("       true \\ pred | " + " | ".join(f"{label:>12}" for label in label_order))
    for label, row in zip(label_order, metrics["confusion_matrix"]):
        print(f"       {label:>11} | " + " | ".join(f"{value:>12}" for value in row))

    with (DATA_DIR / "model_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "accuracy": metrics["accuracy"],
                "labels": label_order,
                "confusion_matrix": metrics["confusion_matrix"],
                "n_samples": metrics["n_samples"],
                "n_train": metrics["n_train"],
                "n_test": metrics["n_test"],
                "dataset": "synthetic",
                "validation": "hold-out from synthetic generator; not validated on real fields",
                "trained_at": bundle["trained_at"],
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Done. Model: {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
