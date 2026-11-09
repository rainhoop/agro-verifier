"""Secret-hygiene checks.  Never print a secret value, even on failure."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _env_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    pattern = re.compile(r"^\s*(?:export\s+)?([A-Za-z_$][\w$]*)\s*=\s*(.*?)\s*$")
    for line in _read(path).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[match.group(1)] = value
    return values


def _secret_values() -> list[str]:
    values: list[str] = []
    for path in (ROOT / ".env", WORKSPACE / ".env"):
        env = _env_values(path)
        for key in ("CDSE_CLIENT_SECRET", "$secret", "secret", "CDSE_CLIENT_ID", "$id"):
            value = env.get(key)
            if value and len(value) >= 8:
                values.append(value)
    return sorted(set(values))


def test_env_is_ignored_by_git_and_docker() -> None:
    gitignore = _read(ROOT / ".gitignore")
    dockerignore = _read(ROOT / ".dockerignore")
    assert re.search(r"(?m)^\s*\.env\s*$", gitignore)
    assert re.search(r"(?m)^\s*\.env\s*$", dockerignore)
    assert ".env" in dockerignore


def test_llm_examples_document_groq_and_local_lm_studio() -> None:
    env_example = _read(ROOT / ".env.example")
    readme = _read(ROOT / "README.md")
    assert "LLM_BASE_URL=http://127.0.0.1:1234/v1" in env_example
    assert "LLM_MODEL=qwen/qwen3.5-9b" in env_example
    assert "LLM_BASE_URL=https://api.groq.com/openai/v1" in env_example
    assert "LLM_MODEL=llama-3.1-8b-instant" in env_example
    assert "api.groq.com/openai/v1" in readme
    assert "llama-3.1-8b-instant" in readme
    assert "api.x.ai" not in env_example.lower() + readme.lower()


def test_mockups_and_candidate_pool_are_excluded_from_build_artifacts() -> None:
    gitignore = _read(ROOT / ".gitignore")
    dockerignore = _read(ROOT / ".dockerignore")
    assert "mockups/" in gitignore
    assert (ROOT / "mockups").is_dir()
    assert not (ROOT / "mocaps").exists()
    assert "backend/data/candidate_pool.json" in dockerignore


def test_explanation_and_chat_show_response_source() -> None:
    html = _read(ROOT / "frontend" / "index.html")
    app_js = _read(ROOT / "frontend" / "app.js")
    translations = json.loads(_read(ROOT / "frontend" / "i18n.json"))
    assert 'id="explanation-source"' in html
    assert "explanation.source.llm" in app_js
    assert "explanation.source.template" in app_js
    assert '"chat.template"' in app_js
    assert '"chat.ai"' in app_js
    assert translations["ru"]["explanation.source.llm"] == "ИИ"
    assert translations["ru"]["explanation.source.template"] == "Шаблон"


def test_dockerfile_does_not_copy_env_or_whole_context() -> None:
    dockerfile = _read(ROOT / "Dockerfile")
    copy_lines = [line.strip() for line in dockerfile.splitlines() if line.strip().upper().startswith(("COPY ", "ADD "))]
    assert copy_lines
    for line in copy_lines:
        assert ".env" not in line
        assert line.split() != ["COPY", ".", "."]
        assert not line.startswith("ADD .")
    assert "COPY backend ./backend" in dockerfile
    assert "COPY frontend ./frontend" in dockerfile
    assert "COPY . ." not in dockerfile


def test_known_secrets_are_absent_from_source_tree() -> None:
    secrets = _secret_values()
    assert secrets, "No .env secret was found; set the documented environment variables"
    skip_dirs = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "node_modules",
        "cdse_cache",
        "vendor",
    }
    secret_bytes = [value.encode("utf-8") for value in secrets]
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.name.startswith(".env"):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".woff2", ".pdf", ".zip"}:
            continue
        if path == Path(__file__).resolve():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if any(value in data for value in secret_bytes):
            hits.append(str(path.relative_to(ROOT)))
    assert not hits, f"Secret value found in: {hits}"


def test_frontend_has_no_secret_names() -> None:
    for path in (ROOT / "frontend").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".js", ".html", ".css", ".json"}:
            continue
        text = _read(path)
        assert "CDSE_CLIENT_SECRET" not in text, path.name
        assert "CDSE_CLIENT_ID" not in text, path.name
        assert "client_secret" not in text.lower(), path.name
