"""animask/llm.py and the engine's ChatAPI adapter: provider routing, wire
formats, retry policy, call log. No network: the HTTP function is replaced."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "engine"))

from animask import llm  # noqa: E402
from utils import get_models  # noqa: E402
from modules.llm.BaseLLM import set_call_ctx  # noqa: E402
from modules.llm.ChatAPI import ChatAPI  # noqa: E402

KEYS = {"OPENAI_API_KEY": "k-openai", "ANTHROPIC_API_KEY": "k-anthropic",
        "GEMINI_API_KEY": "k-gemini", "DEEPSEEK_API_KEY": "k-deepseek",
        "MOONSHOT_API_KEY": "k-moonshot", "DASHSCOPE_API_KEY": "k-dashscope",
        "OPENROUTER_API_KEY": "k-openrouter",
        "COMPAT_BASE_URL": "http://localhost:8000/v1", "COMPAT_API_KEY": "x"}


def openai_reply(text, usage=None):
    body = {"choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": usage or {"prompt_tokens": 3, "completion_tokens": 2}}
    return 200, {}, json.dumps(body).encode()


def anthropic_reply(text):
    body = {"content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 3, "output_tokens": 2}}
    return 200, {}, json.dumps(body).encode()


class FakeHTTP(unittest.TestCase):
    def setUp(self):
        self._env, self._post, self._sleep = llm.ENV, llm._post, llm.time.sleep
        llm.ENV = dict(KEYS)
        llm.time.sleep = lambda s: None
        self.calls = []
        os.environ["LLM_USAGE_LOG"] = "0"

    def tearDown(self):
        llm.ENV, llm._post, llm.time.sleep = self._env, self._post, self._sleep
        os.environ.pop("LLM_USAGE_LOG", None)

    def serve(self, *responses):
        it = iter(responses)

        def _post(url, headers, payload, timeout):
            self.calls.append({"url": url, "headers": headers,
                               "payload": payload})
            r = next(it)
            if isinstance(r, Exception):
                raise r
            return r
        llm._post = _post


class TestRouting(FakeHTTP):
    def test_prefix_inference(self):
        cases = {"gpt-5.5": "openai", "o3-mini": "openai",
                 "claude-sonnet-5": "anthropic", "gemini-3.7-flash": "gemini",
                 "deepseek-v4-pro": "deepseek", "kimi-k2.6": "moonshot",
                 "qwen3.7-plus": "dashscope", "my-local-model": "compat"}
        for name, provider in cases.items():
            self.assertEqual(llm.provider_of(name), (provider, name), name)

    def test_explicit_prefix_wins_and_is_stripped(self):
        self.assertEqual(llm.provider_of("compat:gpt-5.5"), ("compat", "gpt-5.5"))
        self.assertEqual(llm.provider_of("openrouter:openai/gpt-5.5"),
                         ("openrouter", "openai/gpt-5.5"))
        route = llm.resolve("compat:gpt-5.5")
        self.assertEqual(route.base_url, "http://localhost:8000/v1")
        self.assertEqual(route.model, "gpt-5.5")

    def test_defaults_and_overrides(self):
        self.assertEqual(llm.resolve("gpt-5.5").base_url,
                         "https://api.openai.com/v1")
        llm.ENV["DEEPSEEK_BASE_URL"] = "https://proxy.example/v1/"
        self.assertEqual(llm.resolve("deepseek-v4-pro").base_url,
                         "https://proxy.example/v1")

    def test_missing_key_is_a_config_error(self):
        llm.ENV.pop("ANTHROPIC_API_KEY")
        with self.assertRaises(llm.LLMConfigError):
            llm.resolve("claude-sonnet-5")
        with self.assertRaises(llm.LLMConfigError):
            llm.query("hi", model="claude-sonnet-5")
        llm.ENV.pop("COMPAT_BASE_URL")
        with self.assertRaises(llm.LLMConfigError):
            llm.resolve("unknown-model")


class TestWireFormats(FakeHTTP):
    def test_openai_style_request(self):
        self.serve(openai_reply("pong"))
        out = llm.query("ping", model="gpt-5.5", stdin="CTX")
        self.assertEqual(out, "pong")
        c = self.calls[0]
        self.assertEqual(c["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(c["headers"]["Authorization"], "Bearer k-openai")
        p = c["payload"]
        self.assertEqual(p["model"], "gpt-5.5")
        self.assertEqual(p["temperature"], 0)
        self.assertEqual(p["messages"][0]["role"], "user")
        self.assertIn("<context>\nCTX\n</context>", p["messages"][0]["content"])
        self.assertNotIn("thinking", p)

    def test_anthropic_messages_request(self):
        self.serve(anthropic_reply("hello"))
        msgs = [{"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
                {"role": "user", "content": "again"}]
        out = llm.chat(msgs, model="claude-sonnet-5", temperature=0.3,
                       max_tokens=99)
        self.assertEqual(out, "hello")
        c = self.calls[0]
        self.assertEqual(c["url"], "https://api.anthropic.com/v1/messages")
        self.assertEqual(c["headers"]["x-api-key"], "k-anthropic")
        self.assertEqual(c["headers"]["anthropic-version"], llm.ANTHROPIC_VERSION)
        p = c["payload"]
        self.assertEqual(p["system"], "be brief")
        self.assertEqual([m["role"] for m in p["messages"]],
                         ["user", "assistant", "user"])
        self.assertEqual(p["max_tokens"], 99)
        self.assertEqual(p["temperature"], 0.3)

    def test_reasoning_switch_for_deepseek_and_moonshot(self):
        self.serve(openai_reply("a"), openai_reply("b"), openai_reply("c"))
        llm.query("x", model="deepseek-v4-pro")
        llm.query("x", model="kimi-k2.6")
        self.assertEqual(self.calls[0]["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(self.calls[1]["payload"]["thinking"], {"type": "disabled"})
        self.assertTrue(self.calls[1]["url"].startswith("https://api.moonshot.ai/v1"))
        llm.ENV["LLM_THINKING"] = "default"
        llm.query("x", model="deepseek-v4-pro")
        self.assertNotIn("thinking", self.calls[2]["payload"])


class TestRetryPolicy(FakeHTTP):
    def test_rate_limit_then_success(self):
        self.serve((429, {"Retry-After": "1"}, b"slow down"), openai_reply("ok"))
        self.assertEqual(llm.query("x", model="gpt-5.5"), "ok")
        self.assertEqual(len(self.calls), 2)

    def test_server_error_exhausts_budget(self):
        self.serve(*[(503, {}, b"down")] * 3)
        out = llm.query("x", model="gpt-5.5", retries=2)
        self.assertTrue(out.startswith("<<ERROR: HTTP 503"))
        self.assertEqual(len(self.calls), 3)

    def test_billing_error_never_retries(self):
        self.serve((402, {}, b'{"error":"Insufficient Balance"}'))
        out = llm.query("x", model="deepseek-v4-pro")
        self.assertTrue(out.startswith("<<ERROR: CREDIT_EXHAUSTED"))
        self.assertEqual(len(self.calls), 1)
        self.serve((429, {}, b'{"error":{"type":"insufficient_quota"}}'))
        out = llm.query("x", model="gpt-5.5")
        self.assertTrue(out.startswith("<<ERROR: CREDIT_EXHAUSTED"))

    def test_client_error_never_retries(self):
        self.serve((400, {}, b"bad request"))
        out = llm.query("x", model="gpt-5.5")
        self.assertTrue(out.startswith("<<ERROR: HTTP 400"))
        self.assertEqual(len(self.calls), 1)
        self.serve((401, {}, b"bad key"))
        self.assertTrue(llm.query("x", model="gpt-5.5").startswith("<<ERROR: HTTP 401"))

    def test_timeouts_get_one_retry(self):
        self.serve(TimeoutError("t"), TimeoutError("t"), openai_reply("late"))
        out = llm.query("x", model="gpt-5.5")
        self.assertTrue(out.startswith("<<ERROR: TimeoutError"))
        self.assertEqual(len(self.calls), 2)

    def test_empty_content_is_retried(self):
        self.serve(openai_reply(""), openai_reply("filled"))
        self.assertEqual(llm.query("x", model="gpt-5.5"), "filled")

    def test_many_keeps_order(self):
        self.serve(openai_reply("1"), openai_reply("2"), openai_reply("3"))
        outs = llm.query_many([{"prompt": "a", "model": "gpt-5.5"},
                               {"prompt": "b", "model": "gpt-5.5"},
                               {"prompt": "c", "model": "gpt-5.5"}],
                              max_workers=1)
        self.assertEqual(outs, ["1", "2", "3"])
        self.serve(openai_reply("m"))
        outs = llm.chat_many([{"messages": [{"role": "user", "content": "q"}],
                               "model": "gpt-5.5", "temperature": 0.5}])
        self.assertEqual(outs, ["m"])
        self.assertEqual(self.calls[-1]["payload"]["temperature"], 0.5)


class TestEngineAdapter(FakeHTTP):
    def test_factory_and_contract(self):
        a = get_models("deepseek-v4-pro")
        self.assertIsInstance(a, ChatAPI)
        self.assertEqual(a.provider, "deepseek")
        self.assertEqual(a.model_name, "deepseek-v4-pro")
        self.assertEqual(get_models("compat:local-llama").provider, "compat")

    def test_call_log_record(self):
        self.serve(openai_reply("said"))
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "calls.jsonl"
            os.environ["ANIMASK_CALL_LOG"] = str(log)
            try:
                a = ChatAPI(model="gpt-5.5", temperature=0.2)
                a._resim_tag, a._resim_kind = "role:Hero-en", "role_single"
                set_call_ctx(round=4)
                self.assertEqual(a.chat("act"), "said")
                self.assertIsNotNone(a._last_call_id)
                rec = json.loads(log.read_text().splitlines()[0])
            finally:
                os.environ.pop("ANIMASK_CALL_LOG", None)
                set_call_ctx(round=None)
        for k in ("ts", "model", "tag", "call_id", "round", "kind",
                  "temperature", "attempt", "latency_s", "usage", "messages",
                  "response"):
            self.assertIn(k, rec)
        self.assertEqual(rec["tag"], "role:Hero-en")
        self.assertEqual(rec["round"], 4)
        self.assertEqual(rec["response"], "said")
        self.assertEqual(rec["messages"][0]["content"], "act")

    def test_fatal_errors_stop_the_run(self):
        self.serve((401, {}, b"bad key"))
        with self.assertRaises(RuntimeError):
            ChatAPI(model="gpt-5.5").chat("x")
        self.serve((402, {}, b"no balance"))
        with self.assertRaises(RuntimeError):
            ChatAPI(model="gpt-5.5").chat("x")

    def test_exhausted_retries_return_empty_string(self):
        self.serve(*[(503, {}, b"down")] * 5)
        a = ChatAPI(model="gpt-5.5")
        self.assertEqual(a.get_response(retries=1) if not a.messages else "", "")
        a.user_message("x")
        self.assertEqual(a.get_response(retries=1), "")
        self.assertIsNone(a._last_call_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
