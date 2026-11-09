"""Feature extraction, scoring and synthetic-generation unit tests."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from backend import features, scoring, synthetic


def test_season_dates_grid() -> None:
    dates = features.season_dates(2026)
    assert len(dates) == 21
    assert dates[0] == date(2026, 4, 1)
    assert dates[-1] == date(2026, 10, 18)
    assert all((b - a).days == 10 for a, b in zip(dates, dates[1:]))


def test_temporal_stage_uses_extracted_feature_boundaries_and_accepts_gaps() -> None:
    dates = [date(2026, 4, 1) + timedelta(days=10 * i) for i in range(9)]
    extracted = {
        "growth_start_date": "2026-04-15",
        "peak_date": "2026-05-21",
        "harvest_date": "2026-06-15",
    }
    values = [0.1, 0.2, 0.3, 0.5, 0.65, 0.7, 0.6, 0.2, 0.1]
    measured = [True] * len(values)

    assert features.temporal_stage(dates, values, 0, extracted, measured) == "before_growth"
    assert features.temporal_stage(dates, values, 2, extracted, measured) == "growth"
    assert features.temporal_stage(dates, values, 4, extracted, measured) == "peak"
    assert features.temporal_stage(dates, values, 5, extracted, measured) == "peak"
    assert features.temporal_stage(dates, values, 6, extracted, measured) == "decline"
    assert features.temporal_stage(dates, values, 7, extracted, measured) == "after_harvest"
    assert features.temporal_stage(dates, values, 8, extracted, measured) == "after_harvest"
    assert features.temporal_stage(dates, [0.2] * 9, 5, extracted, measured) is None


def test_temporal_stage_handles_missing_growth_and_short_series() -> None:
    dates = [date(2026, 4, 1) + timedelta(days=10 * i) for i in range(3)]
    values = [0.1, 0.2, 0.3]

    assert features.temporal_stage(
        dates,
        values,
        1,
        {"growth_start_date": None, "peak_date": None, "harvest_date": None},
        measured=[True, True, False],
    ) is None
    assert features.temporal_stage(
        dates[:2], values[:2], 0, {"growth_start_date": None}
    ) is None
    assert features.temporal_stage(
        dates, values, 0, {"growth_start_date": None}, measured=[True, True, True]
    ) == "before_growth"


def test_temporal_stage_marks_near_peak_and_threshold_boundaries() -> None:
    dates = [date(2026, 6, 1) + timedelta(days=10 * i) for i in range(7)]
    values = [0.1, 0.24, 0.25, 0.5, 0.65, 0.4, 0.2]
    extracted = {
        "growth_start_date": "2026-06-20",
        "peak_date": "2026-07-11",
        "harvest_date": "2026-07-21",
    }

    assert features.temporal_stage(dates, values, 1, extracted) == "before_growth"
    assert features.temporal_stage(dates, values, 2, extracted) == "growth"
    assert features.temporal_stage(dates, values, 3, extracted) == "growth"
    assert features.temporal_stage(dates, values, 4, extracted) == "peak"
    assert features.temporal_stage(dates, values, 5, extracted) == "decline"
    assert features.temporal_stage(dates, values, 6, extracted) == "after_harvest"


def test_temporal_color_boundaries_and_missing_values() -> None:
    colors = features.TEMPORAL_NDVI_COLORS
    assert colors == (
        "#E9E1BF",
        "#D8D79A",
        "#BCD07F",
        "#8FBE6C",
        "#5FA462",
        "#3A8660",
        "#1F6F5C",
    )
    assert features.temporal_color(0) == colors[0]
    assert features.temporal_color(1 / len(colors)) == colors[1]
    assert features.temporal_color(1) == colors[-1]
    assert features.temporal_color(-1) == colors[0]
    assert features.temporal_color(None) == "#D8DADD"
    assert features.temporal_color(float("nan")) == "#D8DADD"

    neutral = features.TEMPORAL_DEVIATION_COLORS
    assert features.temporal_color(-0.5, mode="deviation") == neutral[0]
    assert features.temporal_color(0, mode="deviation") == neutral[3]
    assert features.temporal_color(0.5, mode="deviation") == neutral[-1]
    assert features.temporal_color(None, mode="deviation") == "#D8DADD"


def test_interpolate_fills_gaps_and_clips() -> None:
    result = features.interpolate_series([0.1, None, 1.2, -0.2])
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0.0) & (result <= 1.0))
    assert result[0] == 0.1
    assert abs(float(result[1]) - 0.55) < 1e-9


def test_interpolate_all_missing_returns_zeros() -> None:
    result = features.interpolate_series([None, None, None])
    assert np.array_equal(result, np.zeros(3))


def test_pearson_edge_cases() -> None:
    assert features.pearson([1, 2, 3], [2, 4, 6]) == 1.0
    assert features.pearson([1, 2, 3], [3, 2, 1]) == -1.0
    assert features.pearson([1, 1, 1], [1, 2, 3]) == 0.0
    assert features.pearson([1, 2], [1, 2]) == 0.0


def test_build_features_on_reference_curve() -> None:
    dates = features.season_dates(2026)
    values = synthetic.profile_values("wheat")
    feat = features.build_features(
        dates,
        values,
        reference_values=values,
        sowing_date=date(2026, 5, 15),
        weather_summary={"precipitation_index": 95},
    )
    assert feat["correlation_with_reference"] == 1.0
    assert feat["peak_ndvi"] > 0.7
    assert feat["growth_start_date"] is not None
    assert feat["vegetation_days"] > 30
    assert feat["two_wave_detected"] is False
    assert feat["precipitation_index_pct"] == 95


def test_two_wave_detection() -> None:
    dates = features.season_dates(2026)
    first = synthetic.asymmetric_gaussian(dates, date(2026, 6, 10), 0.62, 0.1, 15, 15)
    second = synthetic.asymmetric_gaussian(dates, date(2026, 8, 5), 0.72, 0.1, 15, 15)
    assert features.two_wave_detected(np.maximum(first, second)) is True
    assert features.two_wave_detected(synthetic.profile_values("wheat")) is False


def test_model_feature_vector_shape() -> None:
    vector = features.model_feature_vector(synthetic.profile_values("barley"))
    assert vector.shape == (len(features.model_feature_names()),)
    assert np.all(np.isfinite(vector))
    assert len(vector) == features.N_POINTS + 17


def test_normalize_crop_aliases() -> None:
    assert features.normalize_crop("Пшеница яровая") == "wheat"
    assert features.normalize_crop("АРПА") == "barley"
    assert features.normalize_crop("rapeseed") == "rapso"
    assert features.normalize_crop("кукуруза") is None
    assert features.normalize_crop(None) is None


def test_synthetic_field_is_deterministic() -> None:
    kwargs = {
        "lat": 51.234,
        "lon": 70.765,
        "declared_crop": "wheat",
        "sowing_date": date(2026, 5, 15),
    }
    first = synthetic.synthetic_series_for_field(**kwargs)
    second = synthetic.synthetic_series_for_field(**kwargs)
    assert first["values"] == second["values"]
    assert first["seed"] == second["seed"]
    other = synthetic.synthetic_series_for_field(
        lat=52.234,
        lon=71.765,
        declared_crop="wheat",
        sowing_date=date(2026, 5, 15),
    )
    assert other["seed"] != first["seed"]


def test_score_formula_weights() -> None:
    score, breakdown = scoring.calculate_score(
        {
            "correlation_with_reference": 1.0,
            "growth_start_date": "2026-05-12",
            "peak_ndvi": 0.75,
        },
        reference_peak_value=0.75,
        sowing_date=date(2026, 5, 15),
    )
    assert score == 100
    assert breakdown["shape"] == 50.0
    assert breakdown["timing"] == 25.0
    assert breakdown["peak"] == 25.0
    assert breakdown["timing_diff_days"] == 3


def test_score_penalises_late_sowing() -> None:
    score, breakdown = scoring.calculate_score(
        {
            "correlation_with_reference": 1.0,
            "growth_start_date": "2026-06-14",
            "peak_ndvi": 0.75,
        },
        reference_peak_value=0.75,
        sowing_date=date(2026, 5, 15),
    )
    assert breakdown["timing_diff_days"] == 30
    assert breakdown["timing"] == 16.7
    assert score == 92


def test_verdict_rules() -> None:
    sowing = date(2026, 5, 15)
    bare_features = {
        "peak_ndvi": 0.12,
        "growth_start_date": None,
        "vegetation_days": 0,
        "two_wave_detected": False,
    }
    verdict, code = scoring.decide_verdict(
        bare_features, 0, {"predicted": "bare", "probability": 0.99}, "wheat"
    )
    assert (verdict, code) == ("not_found", "no_growth")

    wave_features = {
        "peak_ndvi": 0.7,
        "growth_start_date": (sowing + timedelta(days=30)).isoformat(),
        "vegetation_days": 60,
        "two_wave_detected": True,
    }
    assert scoring.decide_verdict(
        wave_features, 80, {"predicted": "double_wave", "probability": 0.9}, "wheat"
    )[0] == "needs_check"

    normal_features = {
        "peak_ndvi": 0.7,
        "growth_start_date": (sowing + timedelta(days=30)).isoformat(),
        "vegetation_days": 60,
        "two_wave_detected": False,
    }
    assert scoring.decide_verdict(
        normal_features, 82, {"predicted": "sunflower", "probability": 0.91}, "wheat"
    )[0] == "crop_mismatch"
    assert scoring.decide_verdict(
        normal_features, 70, {"predicted": "wheat", "probability": 0.91}, "wheat"
    )[0] == "confirmed"
    assert scoring.decide_verdict(
        normal_features, 40, {"predicted": "wheat", "probability": 0.61}, "wheat"
    )[0] == "doubtful"
    assert scoring.decide_verdict(
        normal_features, 39, {"predicted": "wheat", "probability": 0.61}, "wheat"
    )[0] == "not_found"


def test_profiles_exist_for_all_crops() -> None:
    for crop in features.CROP_IDS:
        values = scoring.profile_values(crop)
        assert values is not None
        assert len(values) == 21
        assert max(values) > 0.6
    assert scoring.profile_values("unknown") is None
