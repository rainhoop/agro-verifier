"""Static checks for the RU/KZ dictionary and frontend wiring."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def _load() -> dict:
    return json.loads((FRONTEND / "i18n.json").read_text("utf-8"))


def test_dictionaries_have_same_keys() -> None:
    data = _load()
    assert set(data) == {"ru", "kz"}
    assert set(data["ru"]) == set(data["kz"])
    assert len(data["ru"]) >= 160
    for lang in ("ru", "kz"):
        for key, value in data[lang].items():
            if isinstance(value, list):
                assert key == "months"
                assert len(value) == 12
                assert all(isinstance(item, str) and item.strip() for item in value)
            else:
                assert isinstance(value, str) and value.strip(), key


def test_placeholders_match() -> None:
    data = _load()
    placeholder = re.compile(r"\{(\w+)\}")
    for key, ru_value in data["ru"].items():
        if not isinstance(ru_value, str):
            continue
        ru_params = set(placeholder.findall(ru_value))
        kz_params = set(placeholder.findall(data["kz"][key]))
        assert ru_params == kz_params, key


def test_all_html_keys_are_translated() -> None:
    data = _load()
    html = (FRONTEND / "index.html").read_text("utf-8")
    keys = set(re.findall(r'data-i18n="([^"]+)"', html))
    assert keys
    missing_ru = sorted(key for key in keys if key not in data["ru"])
    missing_kz = sorted(key for key in keys if key not in data["kz"])
    assert not missing_ru, missing_ru
    assert not missing_kz, missing_kz


def test_every_verdict_is_translated() -> None:
    data = _load()
    for verdict in ("confirmed", "doubtful", "not_found", "crop_mismatch", "needs_check"):
        key = f"verdict.{verdict}"
        assert data["ru"][key]
        assert data["kz"][key]


def test_kazakh_glyphs_and_local_fonts() -> None:
    data = _load()
    kazakh_text = "".join(
        value for value in data["kz"].values() if isinstance(value, str)
    )
    for char in "әғқңөұү":
        assert char in kazakh_text, char

    font_dir = FRONTEND / "vendor" / "fonts"
    for weight in (400, 500, 600, 700):
        for subset in ("cyrillic-ext", "cyrillic", "latin"):
            path = font_dir / f"inter-{weight}-{subset}.woff2"
            assert path.is_file(), path.name
            assert path.stat().st_size > 1000

    # The Google Fonts unicode-range splits Kazakh letters between the
    # cyrillic and cyrillic-ext files.  The union must cover all of them.
    font_tools = pytest.importorskip("fontTools.ttLib")
    try:
        union: set[int] = set()
        for path in font_dir.glob("inter-400-*.woff2"):
            font = font_tools.TTFont(path)
            union.update(font.getBestCmap())
    except ImportError:
        pytest.skip("brotli is required to decode woff2 in fontTools")
    for char in "ӘәҒғҚқҢңӨөҰұҮүҺһІі":
        assert ord(char) in union, char


def test_frontend_has_no_external_cdn_scripts() -> None:
    html = (FRONTEND / "index.html").read_text("utf-8")
    assert "unpkg.com" not in html
    assert "cdn.jsdelivr.net" not in html
    assert "cdnjs.cloudflare.com" not in html
    assert "fonts.googleapis.com" not in html
    assert "/vendor/leaflet.js" in html
    assert "/vendor/chart.umd.js" in html
    assert "/vendor/leaflet.css" in html


def test_frontend_uses_local_api_only() -> None:
    app_js = (FRONTEND / "app.js").read_text("utf-8")
    assert "nominatim.openstreetmap.org" not in app_js
    assert "/api/geocode" in app_js
    assert "/api/verify" in app_js
    assert "/api/demo-fields" in app_js
    assert "CDSE_CLIENT" not in app_js
    assert "client_secret" not in app_js.lower()


def test_demo_fields_are_loaded_only_when_requested() -> None:
    html = (FRONTEND / "index.html").read_text("utf-8")
    app_js = (FRONTEND / "app.js").read_text("utf-8")
    assert 'class="demo-link" id="demo-toggle"' in html
    assert 'data-i18n="demo.load"' in html
    assert 'id="clear-field"' in html and 'data-i18n="action.clear"' in html
    assert 'get("demo") === "1"' in app_js
    assert 'selectDemo(state.demoFields[0].id' not in app_js
    init = app_js.split("async function init()", 1)[1].split(
        "async function loadDemos()", 1
    )[0]
    assert "loadDemos()" not in init


def test_user_state_is_saved_and_restored_in_the_browser() -> None:
    app_js = (FRONTEND / "app.js").read_text("utf-8")
    assert 'const STORAGE_KEY = "agro-verifier-state-v1"' in app_js
    assert "restoreAppState();" in app_js
    assert "function saveAppState()" in app_js
    for saved_value in (
        "selectedDemoId",
        "geojson",
        "lastResult",
        "chatMessages",
        "assistantDraft",
        "searchDraft",
        "batchRows",
        "mapView",
        "effectVisit",
    ):
        assert saved_value in app_js
    assert 'window.addEventListener("pagehide", saveAppState)' in app_js
