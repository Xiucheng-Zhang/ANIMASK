"""LLM client shared by the pipeline scripts, the probes, and the simulation
engine.

    query(prompt, model, stdin=None)         single turn, temperature 0   -> str
    chat(messages, model, temperature=...)   multi turn                   -> str
    query_many(jobs) / chat_many(jobs)       thread-pooled batches        -> list[str]

Each returns the model's text, or the sentinel "<<ERROR: reason>>" when every
attempt failed; callers test for that prefix. A missing key or endpoint
raises LLMConfigError at once instead of producing a thousand sentinels.

Configuration lives in `.env` at the repository root (copy .env.example). A
variable already present in the process environment wins over the file.

Providers. The provider is inferred from the model name, or forced with a
"provider:" prefix ("anthropic:claude-sonnet-5", "compat:my-local-model"):

    provider    model names         key                 base URL (default)
    openai      gpt-*, o*           OPENAI_API_KEY      https://api.openai.com/v1
    anthropic   claude-*            ANTHROPIC_API_KEY   https://api.anthropic.com
    gemini      gemini-*            GEMINI_API_KEY      https://generativelanguage.googleapis.com/v1beta/openai
    deepseek    deepseek-*          DEEPSEEK_API_KEY    https://api.deepseek.com/v1
    moonshot    kimi-*, moonshot-*  MOONSHOT_API_KEY    https://api.moonshot.ai/v1
    dashscope   qwen*, qwq*         DASHSCOPE_API_KEY   https://dashscope.aliyuncs.com/compatible-mode/v1
    openrouter  (prefix only)       OPENROUTER_API_KEY  https://openrouter.ai/api/v1
    compat      anything else       COMPAT_API_KEY      COMPAT_BASE_URL (no default)

<PROVIDER>_BASE_URL overrides any default. Every provider except Anthropic
speaks the OpenAI chat-completions format; Anthropic uses its Messages API.
The "compat" provider covers any OpenAI-compatible server (vLLM, Ollama,
LM Studio, a gateway).

Retry policy: rate limits and server errors (429, 5xx, 529) back off and
retry up to `retries` times, honouring Retry-After; a timeout, connection
error or gateway timeout (408, 524) is retried once, because such a request
has usually been served, and billed, upstream; authentication, billing and
malformed-request errors are never retried.

Accounting: each call appends one JSONL line to LLM_USAGE_LOG (default
results/llm_usage.jsonl; "0" disables) with sizes, latency, attempt count and
the provider's usage object, never the text. LLM_USAGE_TAG, when set, is
copied into each record so batch stages can be told apart.
"""
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# models used in the paper (actor and judge defaults of the pipeline)
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"
GPT = "gpt-5.6-sol"
DEEPSEEK = "deepseek-v4-pro"

# batch orchestration lowers this through the environment so that several
# stages running at once do not flood one provider
MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "10"))

ROOT = Path(__file__).resolve().parent.parent
ANTHROPIC_VERSION = "2023-06-01"
BACKOFF_CAP = 90          # seconds between attempts, at most
RETRY_AFTER_CAP = 120     # longest Retry-After we honour

PROVIDERS = {
    # provider: (env-var stem, default base URL)
    "openai": ("OPENAI", "https://api.openai.com/v1"),
    "anthropic": ("ANTHROPIC", "https://api.anthropic.com"),
    "gemini": ("GEMINI",
               "https://generativelanguage.googleapis.com/v1beta/openai"),
    "deepseek": ("DEEPSEEK", "https://api.deepseek.com/v1"),
    "moonshot": ("MOONSHOT", "https://api.moonshot.ai/v1"),
    "dashscope": ("DASHSCOPE",
                  "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "openrouter": ("OPENROUTER", "https://openrouter.ai/api/v1"),
    "compat": ("COMPAT", ""),
}
PREFIXES = (
    ("gpt-", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
    ("chatgpt", "openai"), ("claude", "anthropic"), ("gemini", "gemini"),
    ("deepseek", "deepseek"), ("kimi", "moonshot"), ("moonshot", "moonshot"),
    ("qwen", "dashscope"), ("qwq", "dashscope"),
)
# The paper ran these reasoning-by-default models with reasoning switched
# off (their APIs accept the `thinking` field); LLM_THINKING=default leaves
# the provider's own default in place.
NO_THINKING_PROVIDERS = {"deepseek", "moonshot"}


class LLMConfigError(RuntimeError):
    """A key or endpoint needed for this model is not configured."""


def load_env(root: Path = ROOT) -> dict:
    """`.env` at the repo root, with the process environment on top."""
    env: dict[str, str] = {}
    path = root / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in os.environ.items():
        if k.endswith(("_API_KEY", "_BASE_URL")) or k.startswith("LLM_"):
            env[k] = v
    return env


ENV = load_env()


@dataclass(frozen=True)
class Route:
    provider: str
    model: str        # the name sent to the provider (prefix stripped)
    base_url: str
    api_key: str


def provider_of(model: str) -> tuple[str, str]:
    """(provider, model name as the provider expects it)."""
    if ":" in model:
        p, m = model.split(":", 1)
        if p in PROVIDERS:
            return p, m
    low = model.lower()
    for prefix, p in PREFIXES:
        if low.startswith(prefix):
            return p, model
    return "compat", model


def resolve(model: str, env: dict | None = None) -> Route:
    env = ENV if env is None else env
    provider, name = provider_of(model)
    stem, default = PROVIDERS[provider]
    base = env.get(f"{stem}_BASE_URL") or default
    key = env.get(f"{stem}_API_KEY", "")
    if not base:
        raise LLMConfigError(f"{model!r} routes to provider {provider!r}: "
                             f"set {stem}_BASE_URL in .env")
    if not key:
        raise LLMConfigError(f"{model!r} routes to provider {provider!r}: "
                             f"set {stem}_API_KEY in .env")
    return Route(provider, name, base.rstrip("/"), key)


# --------------------------------------------------------------------------
# one HTTP attempt
# --------------------------------------------------------------------------
@dataclass
class Attempt:
    ok: bool
    kind: str                 # ok | empty | transport | retry | fatal | credit
    text: str = ""
    usage: dict | None = None
    error: str = ""
    status: int = 0
    retry_after: float | None = None


def _post(url: str, headers: dict, payload: dict,
          timeout: int) -> tuple[int, dict, bytes]:
    """POST JSON; returns (status, headers, body) for any HTTP status.
    Network failures propagate. Tests replace this function."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except Exception:
            body = b""
        return e.code, dict(e.headers or {}), body


def build_request(route: Route, messages: list, temperature: float,
                  max_tokens: int) -> tuple[str, dict, dict]:
    """(url, headers, payload) in the provider's wire format."""
    if route.provider == "anthropic":
        system = "\n\n".join(str(m["content"]) for m in messages
                             if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]}
                 for m in messages if m["role"] != "system"]
        payload = {"model": route.model, "max_tokens": max_tokens,
                   "messages": turns, "temperature": temperature}
        if system:
            payload["system"] = system
        headers = {"x-api-key": route.api_key,
                   "anthropic-version": ANTHROPIC_VERSION}
        return route.base_url + "/v1/messages", headers, payload
    payload = {"model": route.model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if route.provider in NO_THINKING_PROVIDERS \
            and ENV.get("LLM_THINKING", "off") != "default":
        payload["thinking"] = {"type": "disabled"}
    headers = {"Authorization": f"Bearer {route.api_key}"}
    return route.base_url + "/chat/completions", headers, payload


def parse_response(route: Route, body: bytes) -> tuple[str, dict | None]:
    d = json.loads(body)
    if route.provider == "anthropic":
        text = "".join(b.get("text", "") for b in d.get("content", [])
                       if b.get("type") == "text")
        return text, d.get("usage")
    msg = d["choices"][0]["message"]
    return msg.get("content") or "", d.get("usage")


_CREDIT_MARKERS = ("insufficient_quota", "insufficient balance",
                   "credit balance", "billing")
_RETRY_STATUS = (409, 429, 500, 502, 503, 504, 529)
_TRANSPORT_STATUS = (408, 524)


def classify(status: int, body: bytes) -> tuple[str, str]:
    """(kind, error message) for a non-2xx response."""
    snippet = body[:600].decode("utf-8", "replace")
    low = snippet.lower()
    if status == 402 or (status in (400, 403, 429)
                         and any(m in low for m in _CREDIT_MARKERS)):
        return "credit", f"CREDIT_EXHAUSTED: HTTP {status}: {snippet[:200]}"
    if status in _RETRY_STATUS:
        return "retry", f"HTTP {status}: {snippet[:200]}"
    if status in _TRANSPORT_STATUS:
        return "transport", f"HTTP {status}: {snippet[:200]}"
    return "fatal", f"HTTP {status}: {snippet[:200]}"


def attempt(route: Route, messages: list, temperature: float,
            max_tokens: int, timeout: int) -> Attempt:
    url, headers, payload = build_request(route, messages, temperature,
                                          max_tokens)
    try:
        status, hdrs, body = _post(url, headers, payload, timeout)
    except Exception as e:  # timeout, connection reset, DNS
        return Attempt(False, "transport", error=f"{type(e).__name__}: {e}")
    if 200 <= status < 300:
        try:
            text, usage = parse_response(route, body)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            return Attempt(False, "transport", status=status,
                           error=f"unreadable response: {e}")
        text = (text or "").strip()
        if not text:
            return Attempt(False, "empty", status=status, usage=usage,
                           error="empty response")
        return Attempt(True, "ok", text=text, usage=usage, status=status)
    kind, err = classify(status, body)
    ra = hdrs.get("Retry-After") or hdrs.get("retry-after")
    try:
        retry_after = float(ra) if ra else None
    except ValueError:
        retry_after = None
    return Attempt(False, kind, status=status, error=err,
                   retry_after=retry_after)


# --------------------------------------------------------------------------
# retry loop
# --------------------------------------------------------------------------
def complete(messages: list, model: str, temperature: float = 0.0,
             max_tokens: int = 4096, timeout: int = 300,
             retries: int = 4) -> tuple[str | None, dict]:
    """Run the retry policy. Returns (text or None, info); info carries
    provider, attempts, usage, latency_s, and on failure error and kind."""
    route = resolve(model)
    t0 = time.time()
    info: dict = {"provider": route.provider, "attempts": 0}
    transport_fails = 0
    last: Attempt | None = None
    for n in range(retries + 1):
        last = attempt(route, messages, temperature, max_tokens, timeout)
        info["attempts"] = n + 1
        if last.ok:
            info.update(usage=last.usage,
                        latency_s=round(time.time() - t0, 2))
            return last.text, info
        if last.kind in ("credit", "fatal"):
            break
        if last.kind == "transport":
            transport_fails += 1
            if transport_fails > 1:
                break
        if n < retries:
            wait = min(5 * 3 ** n, BACKOFF_CAP)
            if last.retry_after:
                wait = max(wait, min(last.retry_after, RETRY_AFTER_CAP))
            time.sleep(wait)
    info.update(error=last.error, kind=last.kind, status=last.status,
                latency_s=round(time.time() - t0, 2))
    return None, info


def usage_log(record: dict) -> None:
    """Append one accounting record (no prompt text) to LLM_USAGE_LOG."""
    path = os.getenv("LLM_USAGE_LOG")
    if path is None:
        path = str(ROOT / "results" / "llm_usage.jsonl")
    if not path or path == "0":
        return
    try:
        tag = os.getenv("LLM_USAGE_TAG")
        if tag:
            record = {"tag": tag, **record}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # accounting must never break a stage


def _finish(kind: str, model: str, text: str | None, info: dict,
            chars: int, stdin_chars: int = 0) -> str:
    rec = {"ts": round(time.time() - info.get("latency_s", 0), 2),
           "model": model, "provider": info.get("provider"), "kind": kind,
           "ok": text is not None, "prompt_chars": chars,
           "stdin_chars": stdin_chars, "attempts": info.get("attempts"),
           "latency_s": info.get("latency_s")}
    if text is not None:
        rec.update(response_chars=len(text), usage=info.get("usage"))
        usage_log(rec)
        return text
    rec["error"] = info.get("error", "")[:200]
    usage_log(rec)
    return f"<<ERROR: {info.get('error', '')}>>"


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------
def query(prompt: str, model: str = GPT, stdin: str | None = None,
          timeout: int = 300, retries: int = 4,
          max_tokens: int = 4096) -> str:
    """Single-turn call at temperature 0; `stdin` is prepended as context."""
    content = f"<context>\n{stdin}\n</context>\n\n{prompt}" if stdin else prompt
    text, info = complete([{"role": "user", "content": content}], model,
                          0.0, max_tokens, timeout, retries)
    return _finish("query", model, text, info, len(prompt), len(stdin or ""))


def chat(messages: list, model: str = GPT, temperature: float = 0.0,
         max_tokens: int = 2048, timeout: int = 300, retries: int = 4) -> str:
    """Messages-array call (system / user / assistant turns)."""
    text, info = complete(messages, model, temperature, max_tokens, timeout,
                          retries)
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return _finish("chat", model, text, info, chars)


def query_many(jobs: list[dict], max_workers: int = MAX_WORKERS) -> list[str]:
    """jobs: [{prompt, model?, stdin?, max_tokens?}] -> outputs in order."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(query, j["prompt"], j.get("model", GPT),
                          j.get("stdin"), max_tokens=j.get("max_tokens", 4096))
                for j in jobs]
        return [f.result() for f in futs]


def chat_many(jobs: list[dict], max_workers: int = MAX_WORKERS) -> list[str]:
    """jobs: [{messages, model?, temperature?, max_tokens?}] -> outputs in order."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(chat, j["messages"], j.get("model", GPT),
                          j.get("temperature", 0.0),
                          j.get("max_tokens", 2048)) for j in jobs]
        return [f.result() for f in futs]
