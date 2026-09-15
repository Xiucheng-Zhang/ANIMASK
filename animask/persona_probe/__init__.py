"""persona_probe — persona-consistency probe plugin for finished resim runs.

Zero-intrusion: consumes only saved run artifacts —
  results/runs/<pack>/transcript_<tag>.json
  results/runs/<pack>/llm_calls_<tag>.jsonl   (may be absent on old runs)
  engine/data/roles/<pack>/<role>/role_info.json
  engine/presets/<pack>.json
— and writes only NEW files:
  results/runs/<pack>/persona_probe_<tag>[.l1|.l2|.l3].json
  data/cache/persona_probe/<pack>/<role>.atoms.json
  results/persona_probe/aggregate_<batchtag>.{json,md}

Layers:
  L1 l1_stream      per-utterance LLM-NLI vs atomized persona card (cheap, full)
  L2 l2_checkpoint  snapshot-then-probe on the forked live context
  L3 l3_ablation    persona-ablation attribution arm (2AFC + probe gap)

All LLM traffic goes through the module-level shims llm_query / llm_query_many /
llm_chat / llm_chat_many so tests can monkeypatch them; the real backend is
animask/llm.py (query/query_many) plus a messages-array variant that reuses the
same credentials/route (llm.query only takes a single prompt string, but the
L2/L3 context fork must replay a full messages list — see README "deviations").
"""
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent          # animask/persona_probe
DEFAULT_ROOT = PKG_DIR.parent.parent               # repository root

GPT = "gpt-5.6-sol"          # actor-grade model (the default simulation actor)
SONNET = "claude-sonnet-5"   # calibration judge
DEEPSEEK = "deepseek-v4-pro" # main judge (a different family from the actors)
MINI = "gpt-5.6-luna"        # cheap-tier judge (L1 primary / L2 second rater)
# Family structure: L1 primary and L2 second rater are gpt-family; the main
# judge family is DeepSeek, distinct from every actor arm. The paper's actor
# arms span six model families; GPT above is the extraction/world default.

# respect the batch orchestrator's cap (LLM_MAX_WORKERS): 3 concurrent
# probes x 10 workers used to burst 30 parallel judge calls and provoke
# intermittent empty responses upstream
MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "10"))


# --------------------------------------------------------------------------
# paths (root overridable for tests via ANIMASK_ROOT)
# --------------------------------------------------------------------------
def get_root() -> Path:
    return Path(os.environ.get("ANIMASK_ROOT", str(DEFAULT_ROOT)))


def run_dir(pack: str) -> Path:
    return get_root() / "results" / "runs" / pack


def roles_dir(pack: str) -> Path:
    return get_root() / "engine" / "data" / "roles" / pack


def preset_path(pack: str) -> Path:
    return (get_root() / "engine" / "presets"
            / f"{pack}.json")


def cache_dir(pack: str) -> Path:
    return get_root() / "data" / "cache" / "persona_probe" / pack


def transcript_path(pack: str, tag: str) -> Path:
    return run_dir(pack) / f"transcript_{tag}.json"


def call_log_path(pack: str, tag: str) -> Path:
    return run_dir(pack) / f"llm_calls_{tag}.jsonl"


def layer_path(pack: str, tag: str, layer: str) -> Path:
    return run_dir(pack) / f"persona_probe_{tag}.{layer}.json"


def probe_out_path(pack: str, tag: str) -> Path:
    return run_dir(pack) / f"persona_probe_{tag}.json"


def aggregate_dir() -> Path:
    return get_root() / "results" / "persona_probe"


# --------------------------------------------------------------------------
# small IO helpers
# --------------------------------------------------------------------------
def load_json(path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json_atomic(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_json_reply(text: str):
    """Best-effort JSON extraction from an LLM reply."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty LLM reply")
    t = text.strip()
    if t.startswith("<<ERROR"):
        raise ValueError(f"backend error reply: {t[:200]}")
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except (ValueError, TypeError):
        pass
    for op, cl in (("{", "}"), ("[", "]")):
        i = t.find(op)
        j = t.rfind(cl)
        if 0 <= i < j:
            try:
                return json.loads(t[i:j + 1])
            except (ValueError, TypeError):
                continue
    raise ValueError("unparseable JSON reply: " + t[:200])


def resolve_lang(pack: str, cli_lang: str = "auto") -> str:
    """zh/en from CLI override, else the experiment preset's language field."""
    if cli_lang and cli_lang != "auto":
        return cli_lang
    p = preset_path(pack)
    if p.exists():
        try:
            return load_json(p).get("language", "en") or "en"
        except (ValueError, OSError):
            pass
    return "en"


def L(lang: str, zh: str, en: str) -> str:
    """Bilingual template picker."""
    return zh if lang == "zh" else en


def role_display_name(pack: str, role: str) -> str:
    try:
        info = load_json(roles_dir(pack) / role / "role_info.json")
        return info.get("role_name") or role.split("-")[0]
    except (OSError, ValueError):
        return role.split("-")[0]


def seeded_sample(items: list, frac: float, seed: int, min_n: int = 1) -> list:
    """Deterministic sample of round(frac*n) (at least min_n) items,
    in original order."""
    if not items:
        return []
    n = min(len(items), max(min_n, round(frac * len(items))))
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(items)), n))
    return [items[i] for i in idx]


def quantile_indices(n_items: int, n_points: int) -> list:
    """Sorted unique indices at n_points evenly spaced quantiles of 0..n-1."""
    if n_items <= 0 or n_points <= 0:
        return []
    if n_points == 1:
        return [n_items - 1]
    fr = [i / (n_points - 1) for i in range(n_points)]
    return sorted({round(f * (n_items - 1)) for f in fr})


# --------------------------------------------------------------------------
# call-log utilities (shared by L2/L3)
# --------------------------------------------------------------------------
def load_call_log(pack: str, tag: str):
    """List of jsonl records, or None if the log file does not exist."""
    path = call_log_path(pack, tag)
    if not path.exists():
        return None
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue  # torn line from a crashed run
    return records


def is_acting(record: dict) -> bool:
    """Acting calls carry the actual in-scene behavior in response["detail"]."""
    resp = record.get("response", "")
    try:
        parsed = json.loads(resp)
        return isinstance(parsed, dict) and "detail" in parsed
    except (ValueError, TypeError):
        return '"detail"' in str(resp)


def acting_calls(records: list, role_code: str) -> list:
    tag = f"role:{role_code}"
    return [r for r in records if r.get("tag") == tag and is_acting(r)]


def extract_detail(response: str) -> str:
    """The behavioral payload of an acting response (fallback: raw text)."""
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and parsed.get("detail"):
            return str(parsed["detail"])
    except (ValueError, TypeError):
        pass
    return str(response)


# --------------------------------------------------------------------------
# resumable chunked store (one per layer file)
# --------------------------------------------------------------------------
class ResumableStore:
    """items keyed by string; flushed to disk every `flush_every` puts.

    One put ~ one LLM call in every layer, so flush_every=50 matches the
    "save every ~50 calls" requirement. Re-entry: keys already present are
    skipped by the layer loops.
    """

    def __init__(self, path, flush_every: int = 50):
        self.path = Path(path)
        self.flush_every = flush_every
        self._dirty = 0
        self.data = {"items": {}, "meta": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = loaded
            except ValueError:
                pass  # corrupt partial file -> start fresh
        self.data.setdefault("items", {})
        self.data.setdefault("meta", {})

    def has(self, key: str) -> bool:
        return key in self.data["items"]

    def get(self, key: str, default=None):
        return self.data["items"].get(key, default)

    def put(self, key: str, value) -> None:
        self.data["items"][key] = value
        self._dirty += 1
        if self._dirty >= self.flush_every:
            self.flush()

    def set_meta(self, key: str, value) -> None:
        self.data["meta"][key] = value

    def flush(self) -> None:
        save_json_atomic(self.path, self.data)
        self._dirty = 0


# --------------------------------------------------------------------------
# LLM shims — the only place the package touches a model backend.
# Tests monkeypatch these four attributes.
# --------------------------------------------------------------------------
_llm_mod = None

# call accounting: an earlier batch wrote empty probe layers as "done" because
# <<ERROR results were skipped one by one with no global tally. Every real
# backend result passes through _tally(); the runner reads CALL_STATS to
# decide whether the final json may be written. A CREDIT_EXHAUSTED result
# latches: later calls short-circuit instead of burning a request + backoff
# each (that batch lost two whole probe passes to exactly this).
CALL_STATS = {"ok": 0, "error": 0, "parse_skip": 0}
_STATS_LOCK = threading.Lock()
_CREDIT_LATCH = False


def _tally(out):
    global _CREDIT_LATCH
    is_err = isinstance(out, str) and out.startswith("<<ERROR")
    with _STATS_LOCK:
        CALL_STATS["error" if is_err else "ok"] += 1
        if is_err and "CREDIT_EXHAUSTED" in out:
            _CREDIT_LATCH = True
    return out


def tally_parse_skip():
    """A judge reply arrived (transport ok) but was unparseable and NOT
    stored — the item stays missing from the layer store and a re-run would
    retry it. Without this counter such runs were complete=true and the
    missing items (e.g. one of only 8 possible 2AFC votes) were never
    retried by the batch runner."""
    with _STATS_LOCK:
        CALL_STATS["parse_skip"] += 1


def _credit_latched():
    if _CREDIT_LATCH:
        return _tally("<<ERROR: CREDIT_EXHAUSTED (latched — call skipped)>>")
    return None


def _load_llm():
    global _llm_mod
    if _llm_mod is None:
        from animask import llm
        _llm_mod = llm
    return _llm_mod


def llm_query(prompt: str, model: str = GPT, stdin: str = None) -> str:
    short = _credit_latched()
    if short is not None:
        return short
    return _tally(_load_llm().query(prompt, model=model, stdin=stdin))


def llm_query_many(jobs: list, max_workers: int = MAX_WORKERS) -> list:
    """jobs: [{prompt, model?, stdin?}] -> outputs in order (llm.query_many)."""
    if _CREDIT_LATCH:
        return [_tally("<<ERROR: CREDIT_EXHAUSTED (latched — call skipped)>>")
                for _ in jobs]
    return [_tally(o)
            for o in _load_llm().query_many(jobs, max_workers=max_workers)]


def llm_chat(messages: list, model: str = GPT, temperature: float = 0.0,
             timeout: int = 300, retries: int = 3,
             max_tokens: int = 2048) -> str:
    """Messages-array completion through animask/llm.py (same providers and
    credentials as llm_query)."""
    short = _credit_latched()
    if short is not None:
        return short
    return _tally(_load_llm().chat(messages, model=model,
                                   temperature=temperature,
                                   max_tokens=max_tokens, timeout=timeout,
                                   retries=retries))


def llm_chat_many(jobs: list, max_workers: int = MAX_WORKERS) -> list:
    """jobs: [{messages, model?, temperature?}] -> outputs in order.
    Resolves llm_chat at call time so monkeypatching llm_chat also takes
    effect here."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(lambda j=j: llm_chat(**j)) for j in jobs]
        return [f.result() for f in futs]
