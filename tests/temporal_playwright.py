"""Playwright UI regression checks for the world map and on-demand demo flow."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["AGRO_DISABLE_NETWORK"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from backend import chat  # noqa: E402
from backend.main import DEMO_FIELDS, app  # noqa: E402

chat.is_configured = lambda lang="ru": False


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _polygon(lon: float, lat: float, size: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon, lat],
            [lon + size, lat],
            [lon + size, lat + size],
            [lon, lat + size],
            [lon, lat],
        ]],
    }


def _wait_for_server(url: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Uvicorn exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.2)
    raise TimeoutError("Uvicorn did not become ready")


def _rect(page: Any, selector: str) -> dict[str, float]:
    locator = page.locator(selector).first
    assert locator.is_visible(), f"{selector} is not visible"
    box = locator.bounding_box()
    assert box is not None, f"{selector} has no bounding box"
    return box


def _overlap(left: dict[str, float], right: dict[str, float]) -> bool:
    return (
        left["x"] < right["x"] + right["width"]
        and left["x"] + left["width"] > right["x"]
        and left["y"] < right["y"] + right["height"]
        and left["y"] + left["height"] > right["y"]
    )


def _verify_layout(page: Any, width: int) -> None:
    layout = page.evaluate(
        """() => {
          const selectors = ['.leaflet-control-zoom', '.leaflet-control-attribution',
            '.basemap-toggle', '#temporal-panel', '#temporal-legend',
            '#map-stage', '.side-panel'];
          const result = {};
          for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (!element) { result[selector] = null; continue; }
            const rect = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            result[selector] = {
              x: rect.x, y: rect.y, width: rect.width, height: rect.height,
              visible: style.display !== 'none' && style.visibility !== 'hidden'
            };
          }
          return {...result, viewportHeight: window.innerHeight,
            drawing: document.body.classList.contains('is-drawing')};
        }"""
    )
    zoom = layout[".leaflet-control-zoom"]
    attribution = layout[".leaflet-control-attribution"]
    basemap = layout[".basemap-toggle"]
    assert all(layout[selector] and layout[selector]["visible"] for selector in (
        ".leaflet-control-zoom", ".leaflet-control-attribution", ".basemap-toggle"
    ))
    assert zoom["x"] > width / 2
    assert not _overlap(zoom, basemap)
    assert not _overlap(zoom, attribution)
    text = page.locator(".leaflet-control-attribution").inner_text()
    assert "Esri" in text and "Maxar" in text
    temporal = layout["#temporal-panel"]
    if temporal and temporal["visible"]:
        map_stage = layout["#map-stage"]
        if width < 700 and not layout["drawing"]:
            expected_height = min(420, max(300, layout["viewportHeight"] * 0.46))
            assert map_stage["height"] >= expected_height - 1
        assert map_stage["y"] + map_stage["height"] <= temporal["y"] + 1
        legend = layout["#temporal-legend"]
        assert not _overlap(zoom, legend)
        if width < 700:
            drawer = layout[".side-panel"]
            assert temporal["y"] + temporal["height"] <= drawer["y"] + 1


def _temporal_snapshot(page: Any) -> dict[str, str]:
    return page.evaluate(
        """() => {
          const slider = document.querySelector('#temporal-slider');
          const chart = document.querySelector('#ndvi-chart');
          const polygon = document.querySelector('path.demo-polygon, path.drawn-polygon');
          return {
            index: slider.value,
            date: document.querySelector('#temporal-date').textContent.trim(),
            value: document.querySelector('#temporal-value').textContent.trim(),
            stage: document.querySelector('#temporal-stage').textContent.trim(),
            color: polygon ? getComputedStyle(polygon).fill : '',
            opacity: polygon ? getComputedStyle(polygon).fillOpacity : '',
            cursorX: chart.dataset.temporalCursorX || '',
            cursorIndex: chart.dataset.temporalCursorIndex || '',
            aria: slider.getAttribute('aria-valuetext') || ''
          };
        }"""
    )


def _dispatch_range_input(page: Any, index: int) -> None:
    page.locator("#temporal-slider").evaluate(
        "(element, value) => { element.value = String(value); "
        "element.dispatchEvent(new Event('input', { bubbles: true })); }",
        index,
    )
    page.wait_for_function(
        "(value) => Number(document.querySelector('#temporal-slider').value) === value",
        arg=index,
    )


def _assert_temporal_changed(before: dict[str, str], after: dict[str, str]) -> None:
    assert before["index"] != after["index"]
    assert before["date"] != after["date"]
    assert before["value"] != after["value"]
    assert before["stage"] != after["stage"]
    assert before["color"] != after["color"]
    assert float(after["opacity"]) >= 0.7
    assert before["cursorX"] != after["cursorX"]
    assert before["cursorIndex"] != after["cursorIndex"]
    assert after["date"] in after["aria"]


def main() -> None:
    with TestClient(app) as client:
        demo = DEMO_FIELDS[0]
        demo_payload = client.post(
            "/api/verify",
            json={
                "demo_id": demo["id"],
                "crop": demo["crop"],
                "sowing_date": demo["sowing_date"],
                "lang": "ru",
            },
        ).json()
        drawn_payload = client.post(
            "/api/verify",
            json={
                "geojson": _polygon(0, 0, 0.01),
                "crop": "wheat",
                "sowing_date": "2026-05-15",
                "lang": "ru",
            },
        ).json()
    assert drawn_payload["field"].get("region_warning") is True
    peak_index = max(
        range(len(demo_payload["series"])),
        key=lambda index: demo_payload["series"][index]["ndvi"],
    )
    drawn_peak_index = max(
        range(len(drawn_payload["series"])),
        key=lambda index: drawn_payload["series"][index]["ndvi"],
    )
    end_index = len(demo_payload["series"]) - 1
    verify_requests: list[dict] = []
    geocode_requests: list[str] = []

    def fulfill_verify(route: Any) -> None:
        request = route.request.post_data_json
        verify_requests.append(request)
        payload = demo_payload if request.get("demo_id") else drawn_payload
        route.fulfill(status=200, content_type="application/json", body=json.dumps(payload))

    def fulfill_tokyo(route: Any) -> None:
        geocode_requests.append(route.request.url)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "query": "Tokyo",
                    "results": [{
                        "lat": 35.6762,
                        "lon": 139.6503,
                        "display_name": "Tokyo, Japan",
                        "boundingbox": ["35.5", "35.8", "139.5", "139.9"],
                    }],
                },
            ),
        )

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["AGRO_DISABLE_NETWORK"] = "1"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_server(f"{base_url}/api/health", server)
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            for width, height, label in ((1440, 900, "desktop"), (390, 844, "mobile")):
                page = browser.new_page(viewport={"width": width, "height": height})
                page.route("**/api/verify", fulfill_verify)
                page.route("**/api/geocode**", fulfill_tokyo)
                demo_field_requests: list[str] = []
                page.on(
                    "request",
                    lambda request: demo_field_requests.append(request.url)
                    if "/api/demo-fields" in request.url
                    else None,
                )
                page.route(
                    "**/api/chat/config**",
                    lambda route: route.fulfill(
                        status=200,
                        content_type="application/json",
                        body=json.dumps({
                            "session_id": "world-map-playwright",

                            "free_text_enabled": False,
                        }),
                    ),
                )
                page.goto(base_url, wait_until="domcontentloaded")
                page.locator("#field-card.is-empty").wait_for()
                page.wait_for_function("() => document.querySelector('#verify-button').disabled")
                assert page.locator("path.demo-polygon").count() == 0
                assert page.locator(".demo-label").count() == 0
                assert not page.locator("#demo-panel").is_visible()
                assert page.locator("#clear-field").is_hidden()
                assert demo_field_requests == []
                _verify_layout(page, width)
                page.screenshot(path=str(ROOT / "docs" / f"map-demo-{label}-empty.png"))

                page.locator("#demo-toggle").click()
                page.locator("#demo-panel").wait_for(state="visible")
                page.wait_for_function(
                    "() => document.querySelectorAll('#demo-list [data-demo-id]').length === 6"
                )
                assert len(demo_field_requests) == 1
                assert page.locator("path.demo-polygon").count() == 0
                page.screenshot(path=str(ROOT / "docs" / f"map-demo-{label}-list.png"))

                page.locator('[data-demo-id="field-01"]').click()
                assert page.locator("#demo-panel").is_hidden()
                assert page.locator("path.demo-polygon").count() == 1
                assert "conditional" not in page.locator("#field-honesty").get_attribute("class")
                assert page.locator("#crop-select").input_value() == demo["crop"]
                assert page.locator("#sowing-date").input_value() == demo["sowing_date"]
                assert not page.locator("#verify-button").is_disabled()
                page.locator("#verify-button").click()
                page.locator("#temporal-panel:not(.hidden)").wait_for()
                page.wait_for_function(
                    "() => document.querySelector('#ndvi-chart').dataset.temporalCursorIndex"
                )
                _verify_layout(page, width)
                page.screenshot(path=str(ROOT / "docs" / f"map-demo-{label}-result.png"))

                slider = page.locator("#temporal-slider")
                start = _temporal_snapshot(page)
                slider.focus()
                slider.press("End")
                end = _temporal_snapshot(page)
                assert start["index"] != end["index"]
                assert start["cursorX"] != end["cursorX"]
                assert start["date"] != end["date"]
                _dispatch_range_input(page, peak_index)
                peak = _temporal_snapshot(page)
                _assert_temporal_changed(start, peak)

                # The URL entry point must open the list without selecting a field.
                query_page = browser.new_page(viewport={"width": width, "height": height})
                query_page.route("**/api/verify", fulfill_verify)
                query_demo_requests: list[str] = []
                query_page.on(
                    "request",
                    lambda request: query_demo_requests.append(request.url)
                    if "/api/demo-fields" in request.url
                    else None,
                )
                query_page.route("**/api/chat/config**", lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"session_id": "demo-query"}),
                ))
                query_page.goto(base_url + "/?demo=1", wait_until="domcontentloaded")
                query_page.locator("#demo-panel").wait_for(state="visible")
                query_page.wait_for_function(
                    "() => document.querySelectorAll('#demo-list [data-demo-id]').length === 6"
                )
                assert query_page.locator("path.demo-polygon").count() == 0
                assert len(query_demo_requests) == 1
                query_page.close()

                page.locator("#clear-field").click()
                assert page.locator("path.demo-polygon").count() == 0
                assert page.locator("#field-card.is-empty").is_visible()
                assert page.locator("#temporal-panel").is_hidden()
                assert page.locator("#verify-button").is_disabled()

                # Global address and coordinates must no longer be restricted to Kazakhstan.
                search = page.locator("#place-search")
                search.fill("Tokyo")
                search.press("Enter")
                page.locator("#search-result-0").wait_for(state="visible")
                page.locator("#search-result-0").click()
                assert geocode_requests
                search.fill("0, 0")
                search.press("Enter")

                page.locator("#draw-polygon").click()
                open_map = _rect(page, "#map")
                open_points = (
                    (0.55, 0.35),
                    (0.78, 0.35),
                    (0.78, 0.56),
                    (0.88, 0.46),
                )
                for x_ratio, y_ratio in open_points:
                    page.mouse.click(
                        open_map["x"] + open_map["width"] * x_ratio,
                        open_map["y"] + open_map["height"] * y_ratio,
                    )
                assert "4" in page.locator("#draw-sheet-count").inner_text()
                assert page.locator("#map-notice").is_hidden()
                assert page.locator("#map-hint").is_hidden()
                page.mouse.click(
                    open_map["x"] + open_map["width"] * 0.55,
                    open_map["y"] + open_map["height"] * 0.82,
                )
                assert "5" in page.locator("#draw-sheet-count").inner_text()
                assert page.locator("#sheet-finish").is_enabled()
                page.keyboard.press("Escape")

                page.locator("#draw-rectangle").click()
                assert page.locator("body").evaluate(
                    "(element) => element.classList.contains('is-drawing')"
                )
                if width < 700:
                    sheet = page.locator("#draw-sheet")
                    handle = page.locator("#draw-sheet-handle")
                    sheet_before = _rect(page, "#draw-sheet")
                    handle_box = _rect(page, "#draw-sheet-handle")
                    page.mouse.move(
                        handle_box["x"] + handle_box["width"] / 2,
                        handle_box["y"] + handle_box["height"] / 2,
                    )
                    page.mouse.down()
                    page.mouse.move(
                        handle_box["x"] + handle_box["width"] / 2,
                        handle_box["y"] + handle_box["height"] / 2 + 72,
                        steps=6,
                    )
                    page.mouse.up()
                    sheet_moved = _rect(page, "#draw-sheet")
                    assert sheet_moved["y"] >= sheet_before["y"] + 40
                    assert handle.get_attribute("aria-valuenow") != "0"
                    handle.focus()
                    page.keyboard.press("Home")
                    sheet_reset = _rect(page, "#draw-sheet")
                    assert abs(sheet_reset["y"] - sheet_before["y"]) <= 1
                    assert page.locator(".basemap-toggle").is_visible()
                    assert page.locator(".leaflet-control-attribution").is_visible()
                    sheet_top = sheet_reset["y"]
                    for selector in (".basemap-toggle", ".leaflet-control-attribution"):
                        box = _rect(page, selector)
                        assert box["y"] + box["height"] <= sheet_top + 2
                    page.locator("#map-hint").wait_for(state="hidden", timeout=5000)
                map_box = _rect(page, "#map")
                center_x = map_box["x"] + map_box["width"] / 2
                center_y = map_box["y"] + map_box["height"] / 2
                page.mouse.click(center_x - 55, center_y - 55)
                page.mouse.click(center_x + 55, center_y + 55)
                page.wait_for_timeout(100)
                still_drawing = page.locator("body").evaluate(
                    "(element) => element.classList.contains('is-drawing')"
                )
                if still_drawing:
                    finish = page.locator("#sheet-finish")
                    if finish.is_visible() and finish.is_enabled():
                        finish.click()
                    else:
                        page.locator("#finish-draw").click()
                assert page.locator("#field-card.is-empty").is_hidden()
                assert not page.locator("#verify-button").is_disabled()
                page.screenshot(path=str(ROOT / "docs" / f"map-demo-{label}-drawn.png"))
                page.locator("#verify-button").click()
                page.locator("#temporal-panel:not(.hidden)").wait_for()
                assert page.locator("#field-honesty.region-warning").is_visible()
                assert "ориентировочный" in page.locator("#field-honesty").inner_text()
                drawn_start = _temporal_snapshot(page)
                _dispatch_range_input(page, drawn_peak_index)
                drawn_peak = _temporal_snapshot(page)
                assert drawn_start["date"] != drawn_peak["date"]
                assert drawn_start["value"] != drawn_peak["value"]
                assert drawn_start["cursorX"] != drawn_peak["cursorX"]
                assert drawn_start["color"] != drawn_peak["color"]
                _verify_layout(page, width)
                page.screenshot(path=str(ROOT / "docs" / f"map-demo-{label}-world-warning.png"))
                assert page.locator("path.demo-polygon").count() == 0
                assert any(item.get("geojson") for item in verify_requests)
                page.close()

            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)

    print(
        "Playwright passed: empty start, list link and ?demo=1, selected offline demo, "
        "temporal slider, clear, worldwide search, drawn contour, region warning, "
        "desktop/mobile layout and screenshots."
    )


if __name__ == "__main__":
    main()
