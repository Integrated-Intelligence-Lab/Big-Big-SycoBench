"""Minimal OpenAI REST client (stdlib only): files, batches, models."""

import json
import os
import sys
import urllib.error
import urllib.request
import uuid

from config import API_BASE, KEY_FILE


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        sys.exit(f"No API key. Set OPENAI_API_KEY or put the key in {KEY_FILE}")
    return key


def _request(method: str, path: str, body: bytes | None = None,
             content_type: str | None = None) -> dict | bytes:
    req = urllib.request.Request(f"{API_BASE}{path}", data=body, method=method)
    req.add_header("Authorization", f"Bearer {api_key()}")
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {path}: {e.read().decode(errors='replace')[:2000]}")
    if path.endswith("/content"):
        return data
    return json.loads(data)


def get(path: str) -> dict:
    return _request("GET", path)


def post_json(path: str, payload: dict) -> dict:
    return _request("POST", path, json.dumps(payload).encode(), "application/json")


def upload_file(filepath: str, purpose: str = "batch") -> dict:
    boundary = uuid.uuid4().hex
    with open(filepath, "rb") as f:
        content = f.read()
    name = os.path.basename(filepath)
    parts = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\n{purpose}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
        f"Content-Type: application/jsonl\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return _request("POST", "/files", parts,
                    f"multipart/form-data; boundary={boundary}")


def create_batch(input_file_id: str, endpoint: str = "/v1/chat/completions") -> dict:
    return post_json("/batches", {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": "24h",
    })


def batch_status(batch_id: str) -> dict:
    return get(f"/batches/{batch_id}")


def download_file(file_id: str) -> bytes:
    return get(f"/files/{file_id}/content")


def preflight_model(model: str) -> None:
    """Abort with a helpful message if the configured model id is absent."""
    ids = [m["id"] for m in get("/models").get("data", [])]
    if model in ids:
        print(f"preflight ok: {model} is available")
        return
    close = sorted(i for i in ids if "gpt-5" in i)
    sys.exit(
        f"Model id '{model}' not found for this key.\n"
        f"Available gpt-5* ids:\n  " + "\n  ".join(close) +
        "\nEdit MODEL in config.py and re-run."
    )
