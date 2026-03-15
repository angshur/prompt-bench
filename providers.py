import os
import time
from typing import Dict, Any


class ProviderError(Exception):
    pass


def _now_ms() -> int:
    return int(time.time() * 1000)


def call_model(provider: str, model_version: str, prompt: str) -> Dict[str, Any]:
    provider = provider.strip()
    if provider == "OpenAI":
        return _call_openai(model_version, prompt)
    if provider in ("Claude/Anthropic", "Anthropic"):
        return _call_anthropic(model_version, prompt)
    if provider == "Gemini":
        return _call_gemini(model_version, prompt)
    if provider.startswith("xAI"):
        return _call_xai(model_version, prompt)
    if provider == "Llama":
        return _call_llama_gateway(model_version, prompt)
    raise ProviderError(f"Unknown provider: {provider}")


def _call_openai(model_version: str, prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ProviderError("OPENAI_API_KEY not set")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    t0 = _now_ms()
    resp = client.chat.completions.create(
        model=model_version,
        messages=[
            {"role": "system", "content": "You are a precise analytics assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    t1 = _now_ms()

    out = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    tokens_in = getattr(usage, "prompt_tokens", None) if usage else None
    tokens_out = getattr(usage, "completion_tokens", None) if usage else None

    return {"output_text": out, "latency_ms": t1 - t0, "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_est_usd": None}


def _call_anthropic(model_version: str, prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ProviderError("ANTHROPIC_API_KEY not set")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    t0 = _now_ms()
    msg = client.messages.create(
        model=model_version,
        max_tokens=1200,
        temperature=0.2,
        system="You are a precise analytics assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    t1 = _now_ms()

    out = "".join([b.text for b in msg.content if getattr(b, "type", "") == "text"])
    return {"output_text": out, "latency_ms": t1 - t0, "tokens_in": None, "tokens_out": None, "cost_est_usd": None}


def _call_gemini(model_version: str, prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ProviderError("GOOGLE_API_KEY not set")

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_version)

    t0 = _now_ms()
    resp = model.generate_content(prompt)
    t1 = _now_ms()

    out = getattr(resp, "text", "") or ""
    return {"output_text": out, "latency_ms": t1 - t0, "tokens_in": None, "tokens_out": None, "cost_est_usd": None}


def _call_xai(model_version: str, prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("XAI_API_KEY")
    base_url = os.getenv("XAI_BASE_URL")
    if not api_key or not base_url:
        raise ProviderError("XAI_API_KEY or XAI_BASE_URL not set")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    t0 = _now_ms()
    resp = client.chat.completions.create(
        model=model_version,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    t1 = _now_ms()

    out = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return {"output_text": out, "latency_ms": t1 - t0, "tokens_in": getattr(usage, "prompt_tokens", None) if usage else None, "tokens_out": getattr(usage, "completion_tokens", None) if usage else None, "cost_est_usd": None}


def _call_llama_gateway(model_version: str, prompt: str) -> Dict[str, Any]:
    import requests
    url = os.getenv("LLAMA_GATEWAY_URL")
    if not url:
        raise ProviderError("LLAMA_GATEWAY_URL not set")
    headers = {}
    if os.getenv("LLAMA_GATEWAY_KEY"):
        headers["Authorization"] = f"Bearer {os.getenv('LLAMA_GATEWAY_KEY')}"
    payload = {"model": model_version, "prompt": prompt}

    t0 = _now_ms()
    r = requests.post(url, json=payload, headers=headers, timeout=90)
    t1 = _now_ms()

    if r.status_code >= 400:
        raise ProviderError(f"Llama gateway error: {r.status_code} {r.text[:200]}")
    data = r.json()
    return {
        "output_text": data.get("output_text", ""),
        "latency_ms": t1 - t0,
        "tokens_in": data.get("tokens_in"),
        "tokens_out": data.get("tokens_out"),
        "cost_est_usd": data.get("cost_est_usd"),
    }

