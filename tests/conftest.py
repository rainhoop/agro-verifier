"""Shared fixtures. Tests must never touch the network or rewrite demo data."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# This must be set before backend.main is imported by any test module.
os.environ["AGRO_DISABLE_NETWORK"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def disable_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend import chat

    monkeypatch.setattr(chat, "is_configured", lambda lang="ru": False)


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def demo_fields() -> list[dict]:
    path = ROOT / "backend" / "data" / "demo_fields.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["fields"]
