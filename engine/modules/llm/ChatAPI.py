"""Chat adapter for the simulation engine: the BaseLLM interface over the
project-wide client in animask/llm.py.

One adapter serves every provider; the provider is inferred from the model
name or forced with a "provider:" prefix (see animask/llm.py). Sampling is
per instance (`temperature`, `max_tokens`), and the engine tags instances
with `_resim_tag` (who is calling) and `_resim_kind` (what for) so the call
log can attribute each call.

If ANIMASK_CALL_LOG names a file, every call is appended there as one JSONL
record: messages, response, latency, the provider's usage object, and the
round stamped in BaseLLM.CALL_CTX. The evaluation chain reads this log back,
so finished runs can be analysed without re-querying. Unset or "0": no log.

Failure semantics: rate limits and server errors are retried with backoff;
after the retry budget the call returns "" (never None, every call site
assumes a string). A bad key, an unknown model name or an exhausted balance
is identical on every retry, so those raise RuntimeError and stop the run
instead of degrading silently.
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path

from .BaseLLM import BaseLLM, CALL_CTX

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from animask import llm as _client  # noqa: E402  (animask/llm.py)

_FATAL_KINDS = ("fatal", "credit")


def log_call(record: dict) -> None:
    path = os.getenv("ANIMASK_CALL_LOG", "")
    if not path or path == "0":
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging must never break a run


class ChatAPI(BaseLLM):
    def __init__(self, model="gpt-5.6-sol", temperature=0.7, max_tokens=2048):
        super().__init__()
        self.model_name = model
        self.provider = _client.provider_of(model)[0]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = []
        # id of the last SUCCESSFUL call (None after a failed one); records
        # store it to link history entries to their producing log line
        self._last_call_id = None

    def initialize_message(self):
        self.messages = []

    def ai_message(self, payload):
        self.messages.append({"role": "assistant", "content": payload})

    def system_message(self, payload):
        self.messages.append({"role": "system", "content": payload})

    def user_message(self, payload):
        self.messages.append({"role": "user", "content": payload})

    def get_response(self, retries=4, timeout=300):
        call_id = uuid.uuid4().hex[:16]
        self._last_call_id = None
        t0 = time.time()
        text, info = _client.complete(self.messages, self.model_name,
                                      self.temperature, self.max_tokens,
                                      timeout, retries)
        record = {"ts": round(t0, 2), "model": self.model_name,
                  "tag": getattr(self, "_resim_tag", None),
                  "call_id": call_id, "round": CALL_CTX.get("round"),
                  "kind": getattr(self, "_resim_kind", None),
                  "temperature": self.temperature,
                  "attempt": max(info.get("attempts", 1) - 1, 0),
                  "latency_s": round(time.time() - t0, 2),
                  "usage": info.get("usage"), "messages": self.messages}
        if text is not None:
            log_call({**record, "response": text})
            self._last_call_id = call_id
            return text
        # failed calls stay in the log (response="") so data gaps are visible
        log_call({**record, "response": "",
                  "error": str(info.get("error", ""))[:300]})
        if info.get("kind") in _FATAL_KINDS:
            raise RuntimeError(f"LLM call failed for {self.model_name!r}: "
                               f"{info.get('error')}")
        print(f"ChatAPI error after retries: {info.get('error')}")
        return ""

    def chat(self, text):
        self.initialize_message()
        if isinstance(text, str):
            self.user_message(text)
            return self.get_response()
        return ""

    def print_prompt(self):
        for message in self.messages:
            print(message)
