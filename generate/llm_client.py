"""Dual-LLM client — Gemini Flash (default) + NVIDIA NIM hosted (alternative).

Providers:
  - gemini : Google Gemini 2.5 Flash via the free-tier API key
             (google-genai SDK, `generate_content`)
  - nvidia : NVIDIA NIM hosted on build.nvidia.com
             (OpenAI-compatible endpoint, `integrate.api.nvidia.com/v1`)
  - auto   : try primary (default gemini), fall back to nvidia on
             rate-limit / auth / upstream errors

No GPU binding, no self-hosted NIMs. All three NVIDIA models are hosted
externally, so the API key is the only thing required.

Every result is post-processed through the citation validator below;
invalid citations make the result a Refusal.
"""

import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from generate.prompt import build_messages, is_refusal

# Load .env at module import so adapters work even when this module is
# used directly (e.g. `python -c "from generate.llm_client import generate"`)
# without going through api/main.py's load_dotenv().
load_dotenv()

# Regex for citation brackets: [p.<page>, <chunk_id>]
_CITATION_RE = re.compile(r"\[\s*p\.\s*([^\],]+)\s*,\s*([^\]]+)\s*\]")


@dataclass
class Result:
    """Successful (non-refusal) generation result."""
    answer: str
    provider: str
    model: str
    tokens_used: int


@dataclass
class Refusal:
    """Structured refusal: the model (or validator) declined to answer."""
    reason: str
    provider: str
    model: str


# ---------------------------------------------------------------------------
# Adapters — each returns (raw_answer, tokens_used) or raises on failure
# ---------------------------------------------------------------------------

class GeminiAdapter:
    """Gemini Flash via google-genai SDK (model id from GEMINI_MODEL env).

    Retries 3x with linear backoff on transient errors (503 high demand,
    429 rate limits) before giving up — mirrors NvidiaAdapter's behaviour
    so `auto` mode doesn't fall back to NVIDIA on a transient spike.
    """

    def __init__(self, model: str | None = None) -> None:
        from google import genai  # lazy import keeps boot fast when unused
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def complete(self, system: str, user: str, temperature: float, max_tokens: int) -> tuple[str, int]:
        last_exc: Exception | None = None
        for attempt in range(3):  # 3 attempts: immediate, +2s, +4s
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config={
                        "system_instruction": system,
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )
                tokens = 0
                try:
                    tokens = int(resp.usage_metadata.candidates_token_count or 0)
                except Exception:
                    pass
                return (resp.text or ""), tokens
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        raise last_exc  # type: ignore[misc]


class NvidiaAdapter:
    """NVIDIA NIM hosted on build.nvidia.com (OpenAI-compatible).

    The hosted endpoint can return transient 504 gateway timeouts under
    load on the free tier. We retry up to 3 times with linear backoff
    (2s, 4s) before giving up — this turns the most common upstream
    failure into a transparent retry rather than a 500.
    """

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI  # lazy import
        self.model = model or os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4.1-flash")
        self._client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
        )

    def complete(self, system: str, user: str, temperature: float, max_tokens: int) -> tuple[str, int]:
        last_exc: Exception | None = None
        for attempt in range(3):  # 3 attempts: immediate, +2s, +4s
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # Reasoning models (nemotron) burn the whole token budget on
                    # hidden reasoning_content and return empty content — disable
                    # their thinking mode so the cited answer lands in `content`.
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                tokens = 0
                try:
                    tokens = int(resp.usage.total_tokens or 0)
                except Exception:
                    pass
                content = ""
                try:
                    content = resp.choices[0].message.content or ""
                except Exception:
                    content = ""
                return content, tokens
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Citation validator — every bracket must point at a retrieved chunk
# ---------------------------------------------------------------------------

def _validate_citations(answer: str, chunk_ids: set[str]) -> tuple[bool, str]:
    """Validate that every [p.x, chunk_id] in the answer is real."""
    found = _CITATION_RE.findall(answer)
    if not found:
        return False, "no citations found in answer"
    for _page, chunk_id in found:
        if chunk_id.strip() not in chunk_ids:
            return False, f"citation [{chunk_id.strip()}] does not match a retrieved chunk"
    return True, "ok"


# ---------------------------------------------------------------------------
# Facade — the single function api/main.py calls
# ---------------------------------------------------------------------------

def _attempt(
    adapter,
    system: str,
    user: str,
    chunk_ids: set[str],
    temperature: float,
    max_tokens: int,
    provider_name: str,
) -> Result | Refusal:
    """Call one adapter, then validate the output against retrieved chunks."""
    raw, tokens = adapter.complete(system, user, temperature, max_tokens)
    if is_refusal(raw):
        return Refusal(reason="model_refused", provider=provider_name, model=adapter.model)
    valid, why = _validate_citations(raw, chunk_ids)
    if not valid:
        return Refusal(reason=f"citation_validation_failed: {why}", provider=provider_name, model=adapter.model)
    return Result(answer=raw, provider=provider_name, model=adapter.model, tokens_used=tokens)


def _nvidia_models(primary: str | None = None) -> list[str]:
    """Ordered list of NVIDIA models to try: primary first, then the rest.

    Each model has its own rate-limit/timeout bucket on the free tier, so
    a DeepSeek 504 doesn't kill the request — GLM or Nemotron picks it up.
    """
    env_list = [
        m.strip()
        for m in os.getenv(
            "NVIDIA_MODELS",
            "deepseek-ai/deepseek-v4.1-flash,z-ai/glm-5.3-flash,nvidia/nemotron-3.5-lightning-30b-a3b",
        ).split(",")
        if m.strip()
    ]
    ordered = [primary] if primary else []
    for m in env_list:
        if m not in ordered:
            ordered.append(m)
    return ordered


def generate(
    question: str,
    chunks: list[dict],
    provider: str = "auto",
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> Result | Refusal:
    """Generate a citation-backed answer, with auto-fallback across providers.

    `model` overrides the env-driven default model id, allowing the UI to
    switch between any model in NVIDIA_MODELS without a redeploy.

    Provider behaviour:
      - "gemini"  -> single attempt; failure -> Refusal(reason="gemini_failed")
      - "nvidia"  -> try each model in NVIDIA_MODELS (primary first) with
                     in-adapter retries; all fail -> Refusal(reason="nvidia_all_failed")
      - "auto"    -> gemini first, then nvidia; both fail -> Refusal
    Never raises — upstream failures surface as a clean Refusal so the
    API returns 200 "Not in notes" instead of HTTP 500.
    """
    system, user = build_messages(question, chunks)
    chunk_ids = {c.get("chunk_id", "") for c in chunks}

    provider = (provider or "auto").lower()

    # Explicit single-provider requests
    if provider == "gemini":
        try:
            return _attempt(GeminiAdapter(model), system, user, chunk_ids, temperature, max_tokens, "gemini")
        except Exception as e:
            return Refusal(reason=f"gemini_failed: {e}", provider="gemini", model=model or "gemini")

    if provider == "nvidia":
        last_exc: Exception | None = None
        for m in _nvidia_models(model):
            try:
                return _attempt(NvidiaAdapter(m), system, user, chunk_ids, temperature, max_tokens, "nvidia")
            except Exception as e:
                last_exc = e
                continue
        return Refusal(reason=f"nvidia_all_failed: {last_exc}", provider="nvidia", model=model or "all")

    # auto: try gemini first, fall back to nvidia on any exception
    try:
        return _attempt(GeminiAdapter(), system, user, chunk_ids, temperature, max_tokens, "gemini")
    except Exception:
        try:
            return _attempt(NvidiaAdapter(model), system, user, chunk_ids, temperature, max_tokens, "nvidia")
        except Exception as e:
            return Refusal(reason=f"all_providers_failed: {e}", provider="auto", model="all")