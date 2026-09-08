#!/usr/bin/env python3
"""Standalone script to check API key status for all configured providers."""

import os
import sys

# Load .env manually
from pathlib import Path
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

import json

def check_gemini_key(key_name: str):
    """Test a single Gemini API key with a minimal request."""
    api_key = os.getenv(key_name, "")
    if not api_key or api_key == "your_gemini_api_key_here":
        print(f"  {key_name}: NOT SET / EMPTY")
        return

    try:
        import httpx
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            json={"contents": [{"parts": [{"text": "Say hello in one word"}]}]},
            timeout=15,
        )
        if response.status_code == 200:
            print(f"  {key_name}: ✅ VALID & WORKING (200 OK)")
        elif response.status_code == 429:
            print(f"  {key_name}: ⚠️  VALID but RATE-LIMITED (429)")
        elif response.status_code == 400:
            body = response.json()
            err_msg = body.get("error", {}).get("message", "unknown")
            print(f"  {key_name}: ❌ 400 Bad Request — {err_msg}")
        elif response.status_code == 403:
            print(f"  {key_name}: ❌ INVALID / FORBIDDEN (403)")
        else:
            print(f"  {key_name}: ❓ Unexpected status {response.status_code} — {response.text[:200]}")
    except Exception as e:
        print(f"  {key_name}: ❌ ERROR — {type(e).__name__}: {e}")


def check_openrouter_key():
    """Test the OpenRouter API key."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("  OPENROUTER_API_KEY: NOT SET / EMPTY")
        return

    try:
        import httpx
        # Check credits/limits
        response = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if response.status_code == 200:
            data = response.json().get("data", {})
            limit = data.get("limit")
            usage = data.get("usage")
            label = data.get("label", "unknown")
            if limit is not None and usage is not None:
                remaining = limit - usage
                print(f"  OPENROUTER_API_KEY: ✅ VALID — label='{label}', limit=${limit}, used=${usage}, remaining=${remaining:.4f}")
            else:
                print(f"  OPENROUTER_API_KEY: ✅ VALID — label='{label}', limit={limit}, usage={usage}")
        elif response.status_code == 401:
            print(f"  OPENROUTER_API_KEY: ❌ INVALID (401 Unauthorized)")
        else:
            print(f"  OPENROUTER_API_KEY: ❓ Unexpected status {response.status_code} — {response.text[:200]}")
    except Exception as e:
        print(f"  OPENROUTER_API_KEY: ❌ ERROR — {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("API KEY STATUS CHECK")
    print("=" * 60)
    print()
    print("--- Gemini API Keys ---")
    check_gemini_key("GEMINI_API_KEY_1")
    check_gemini_key("GEMINI_API_KEY_2")
    check_gemini_key("GEMINI_API_KEY_3")
    print()
    print("--- OpenRouter ---")
    check_openrouter_key()
    print()
    print("=" * 60)
    print("Done.")
