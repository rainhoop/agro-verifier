"""Central configuration and the only place for thresholds and limits.

Environment variables win over values from a local ``.env`` file.  The loader
accepts both the documented ``CDSE_CLIENT_ID`` / ``CDSE_CLIENT_SECRET`` names
and the ``$id`` / ``$secret`` names exported by the Copernicus dashboard.
Secrets are never logged or returned by the API.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
CDSE_CACHE_DIR = DATA_DIR / "cdse_cache"

_ENV_PATTERN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_$][\w$]*)\s*=\s*(.*?)\s*$")
_DOTENV: dict[str, str] = {}


def _clean_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    # A small inline comment after an unquoted value is common in .env files.
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ENV_PATTERN.match(line)
        if not match:
            continue
        values[match.group(1)] = _clean_value(match.group(2))
    return values


def load_environment() -> None:
    """Load the closest .env files without overriding process env vars."""
    global _DOTENV
    candidates = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env",
        Path.cwd() / ".env",
    ]
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        for key, value in _read_env_file(path).items():
            _DOTENV.setdefault(key, value)
            os.environ.setdefault(key, value)


load_environment()


def secret(name: str, *aliases: str) -> str | None:
    for key in (name, *aliases):
        value = os.environ.get(key) or _DOTENV.get(key)
        if value:
            return value.strip()
    return None


def cdse_credentials() -> tuple[str, str] | None:
    client_id = secret("CDSE_CLIENT_ID", "$id", "id")
    client_secret = secret("CDSE_CLIENT_SECRET", "$secret", "secret")
    if not client_id or not client_secret:
        return None
    return client_id, client_secret


def network_disabled() -> bool:
    return os.environ.get("AGRO_DISABLE_NETWORK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ---------------------------------------------------------------------------
# Season and data limits
# ---------------------------------------------------------------------------
SEASON_YEAR = 2026
SEASON_START = (4, 1)
SEASON_END = (10, 31)
STEP_DAYS = 10
N_POINTS = 21

AREA_MIN_HA = 10.0
AREA_MAX_HA = 5000.0
DEMO_AREA_MIN_HA = 50.0
DEMO_AREA_MAX_HA = 150.0

CDSE_STATS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_TIMEOUT = 30.0
CDSE_DRAWN_TIMEOUT = 25.0
CDSE_RETRIES = 1
CDSE_TOKEN_SAFETY_SECONDS = 60
CDSE_MIN_VALID_FRACTION = 0.30
CDSE_MIN_VALID_POINTS = 4
CDSE_MAX_PIXELS = 250_000
CDSE_BASE_RESX = 0.00015
CDSE_BASE_RESY = 0.00010

WEATHER_TIMEOUT = 5.0
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = (
    "AgroVerifier/0.2 (hackathon MVP; field-verification demo; "
    "no commercial use)"
)
NOMINATIM_TIMEOUT = 8.0

# ---------------------------------------------------------------------------
# Scoring and verdict thresholds.  Keep these in one place: the public
# /api/config endpoint and the "How it is calculated" panel read them.
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {"shape": 0.50, "timing": 0.25, "peak": 0.25}
TIMING_FULL_DAYS = 15
TIMING_ZERO_DAYS = 60
CONFIRMED_SCORE = 70
DOUBTFUL_SCORE = 40
NO_GROWTH_PEAK = 0.30
FLAT_PEAK_MAX = 0.40
LOW_PEAK_RATIO = 0.65
CROP_MISMATCH_PROBABILITY = 0.60
LOW_MODEL_CONFIDENCE = 0.60
DOUBLE_WAVE_GUARD = True

MODEL_DATASET_NOTE = (
    "RandomForest обучен на синтетических сезонных кривых; "
    "на реальных полях модель не валидировалась."
)

SOURCE_LABELS = {
    "sentinel-2-l2a": {
        "ru": "Sentinel-2 L2A · 10-дневные композиты · 2026",
        "kz": "Sentinel-2 L2A · 10 күндік композиттер · 2026",
    },
    "synthetic": {
        "ru": "Синтетика · демонстрационный ряд",
        "kz": "Синтетика · демонстрациялық қатар",
    },
}


def public_config() -> dict:
    return {
        "season": {
            "year": SEASON_YEAR,
            "start": f"{SEASON_YEAR}-04-01",
            "end": f"{SEASON_YEAR}-10-31",
            "step_days": STEP_DAYS,
            "points": N_POINTS,
            "aggregation": "P10D",
        },
        "score": {
            "weights": SCORE_WEIGHTS,
            "timing_full_days": TIMING_FULL_DAYS,
            "timing_zero_days": TIMING_ZERO_DAYS,
            "confirmed_min": CONFIRMED_SCORE,
            "doubtful_min": DOUBTFUL_SCORE,
            "peak_reference": "отношение пика NDVI поля к пику эталона, не больше 1",
        },
        "verdict": {
            "no_growth_peak": NO_GROWTH_PEAK,
            "flat_peak_max": FLAT_PEAK_MAX,
            "low_peak_ratio": LOW_PEAK_RATIO,
            "crop_mismatch_probability": CROP_MISMATCH_PROBABILITY,
            "double_wave_guard": DOUBLE_WAVE_GUARD,
        },
        "area": {"min_ha": AREA_MIN_HA, "max_ha": AREA_MAX_HA},
        "data": {
            "source": "sentinel-2-l2a",
            "min_valid_fraction": CDSE_MIN_VALID_FRACTION,
            "min_valid_points": CDSE_MIN_VALID_POINTS,
            "network_enabled": not network_disabled(),
            "credentials_configured": cdse_credentials() is not None,
        },
        "model": {
            "dataset": "synthetic",
            "note": MODEL_DATASET_NOTE,
            "low_confidence_threshold": LOW_MODEL_CONFIDENCE,
        },
    }
