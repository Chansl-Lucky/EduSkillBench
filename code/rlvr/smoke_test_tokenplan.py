#!/usr/bin/env python3
"""Non-destructive TokenPlan model and chat-completions smoke test.

The API key is read only from TOKENPLAN_API_KEY. The script never prints it.
Listing models is free according to the provider documentation; chat probes may
consume quota and therefore require the explicit --probe flag.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://discovery-api.intern-ai.org.cn/v1"


def request_json(
    url: str, api_key: str, payload: dict | None, timeout: int
) -> tuple[int, float, dict]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
    elapsed = time.monotonic() - started
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"non_json_body_prefix": raw.decode("utf-8", errors="replace")[:200]}
    return status, elapsed, parsed


def summarize_chat(model: str, status: int, elapsed: float, data: dict) -> dict:
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    error = data.get("error") or {}
    return {
        "model": model,
        "http_status": status,
        "elapsed_seconds": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
        "content_present": bool(content),
        "content_prefix": str(content or "").strip()[:80],
        "reasoning_chars": len(reasoning or ""),
        "usage": data.get("usage"),
        "error_code": error.get("code") or error.get("type"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default=os.environ.get("TOKENPLAN_BASE_URL", DEFAULT_BASE_URL)
    )
    parser.add_argument("--probe", action="store_true", help="Send quota-consuming chat probes")
    parser.add_argument(
        "--models", nargs="*", help="Probe only these model IDs; defaults to all visible models"
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("TOKENPLAN_API_KEY")
    if not api_key:
        raise SystemExit("TOKENPLAN_API_KEY is required")

    base_url = args.base_url.rstrip("/")
    status, elapsed, model_data = request_json(f"{base_url}/models", api_key, None, args.timeout)
    model_ids = [item.get("id") for item in model_data.get("data", []) if item.get("id")]
    print(
        json.dumps(
            {
                "operation": "list_models",
                "http_status": status,
                "elapsed_seconds": round(elapsed, 3),
                "models": model_ids,
            },
            ensure_ascii=False,
        )
    )

    if status != 200:
        return 1
    if not args.probe:
        return 0

    selected = args.models or model_ids
    unknown = sorted(set(selected) - set(model_ids))
    if unknown:
        print(json.dumps({"warning": "models_not_in_visible_list", "models": unknown}))

    for model in selected:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": False,
        }
        status, elapsed, data = request_json(
            f"{base_url}/chat/completions", api_key, payload, args.timeout
        )
        print(json.dumps(summarize_chat(model, status, elapsed, data), ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
