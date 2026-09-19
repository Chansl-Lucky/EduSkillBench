#!/usr/bin/env python3
"""Fail-closed smoke tests for normal chat and OpenAI tool-call parsing."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request


PORT = os.environ.get("QWEN4B_PORT", "8000")


def default_base_url() -> str:
    gateway = subprocess.check_output(
        [
            "docker",
            "network",
            "inspect",
            "bridge",
            "--format",
            "{{range .IPAM.Config}}{{.Gateway}}{{end}}",
        ],
        text=True,
    ).strip()
    if not gateway:
        raise SystemExit("Could not determine Docker bridge gateway")
    return f"http://{gateway}:{PORT}/v1"


BASE_URL = os.environ.get("QWEN4B_BASE_URL") or default_base_url()
API_KEY = os.environ.get("LOCAL_QWEN_API_KEY", "local-eduskillbench")
MODEL = os.environ.get("QWEN4B_SERVED_NAME", "qwen3-4b-instruct-2507-cdbee75")


def post(payload: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def main() -> None:
    normal = post(
        {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Reply with exactly OK"}],
        }
    )
    content = normal["choices"][0]["message"].get("content") or ""
    if "OK" not in content:
        raise SystemExit(f"Normal chat smoke failed: {content!r}")

    tool = post(
        {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": 128,
            "tool_choice": "required",
            "messages": [
                {"role": "user", "content": "Use the lookup_skill tool for lesson-builder."}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_skill",
                        "description": "Look up an installed skill",
                        "parameters": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                    },
                }
            ],
        }
    )
    calls = tool["choices"][0]["message"].get("tool_calls") or []
    if not calls or calls[0].get("function", {}).get("name") != "lookup_skill":
        raise SystemExit(f"Tool-call smoke failed: {json.dumps(tool)[:1000]}")
    print(f"qwen_api_smoke_ok model={MODEL} normal_chat=1 tool_call=1")


if __name__ == "__main__":
    main()
