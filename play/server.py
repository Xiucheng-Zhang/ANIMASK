#!/usr/bin/env python3
"""ANIMASK, talk to a character: local server for the second page.

Zero-dependency (stdlib only). Serves:
  - docs/play/                 the built frontend (cd play/app && npm run build)
  - /api/books                 shelf: built-in public-domain books + uploads
  - /api/book/<bid>            one book (meta + normalized full text)
  - POST /api/book/upload      save a visitor-uploaded .txt (play/books/uploads/)
  - POST /api/book/delete      remove an uploaded book
  - POST /api/book/characters  LLM cast extraction, cached per book
  - POST /api/bookchat         talk to a character at a reading position
  - /api/sandbox/*             saved conversations (play/sandbox/)

LLM calls go through animask/llm.py, so the provider keys in .env at the
repository root apply. Without this server the same frontend runs in static
mode: uploads stay in the browser and the visitor supplies their own key.

Run from the repository root:  python3 play/server.py   (port 8123, SITE_PORT)
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
DATA = SITE / "app" / "public" / "data"       # built-in books and their casts
BOOKS_DIR = SITE / "books"
UPLOADS = BOOKS_DIR / "uploads"
CHARS_DIR = BOOKS_DIR / "chars"
SANDBOX = SITE / "sandbox"
DIST = ROOT / "docs" / "play"
PORT = int(os.environ.get("SITE_PORT", "8123"))

sys.path.insert(0, str(ROOT))
from animask import llm  # noqa: E402

MAX_BOOK_MB = 8
CHAR_SCAN_CHARS = 50000       # cast extraction reads the opening portion only
BOOKCHAT_CTX_CHARS = 12000    # chat context: tail of text before the reader

# ---------------------------------------------------------------- caching --
_cache: dict[str, tuple[float, object]] = {}
_cache_lock = threading.Lock()


def read_json(path: Path, default=None):
    """mtime-cached JSON read; returns default when missing/corrupt."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return default
    key = str(path)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return default
    with _cache_lock:
        _cache[key] = (mtime, obj)
    return obj


# ------------------------------------------------------------------ books --
_ILLUS_RE = re.compile(r"\[Illustration[^\]]*\]", re.S)


def _normalize_book(text: str) -> str:
    """Canonical text form. The frontend mirrors this exactly (bookstore.ts)
    so character offsets agree between browser and server."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ILLUS_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _count_words(text: str, language: str) -> int:
    if language == "zh":
        return sum(1 for ch in text if "一" <= ch <= "鿿")
    return len(text.split())


def builtin_books() -> list[dict]:
    books = (read_json(DATA / "books.json", {}) or {}).get("books", [])
    return [{**b, "source": "builtin"} for b in books
            if (DATA / f"book__{b.get('bid', '')}.json").is_file()]


def upload_books() -> list[dict]:
    out = []
    if UPLOADS.is_dir():
        for f in sorted(UPLOADS.glob("*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            o = read_json(f)
            if o:
                meta = {k: o.get(k) for k in
                        ("bid", "title", "author", "published",
                         "language", "words", "added")}
                meta["source"] = "upload"
                out.append(meta)
    return out


def api_books() -> dict:
    return {"books": upload_books() + builtin_books()}


def _book_record(bid: str) -> dict:
    bid = re.sub(r"[^\w.\-]", "", bid)
    up = read_json(UPLOADS / f"{bid}.json")
    if up:
        return {**up, "source": "upload"}
    doc = read_json(DATA / f"book__{bid}.json")
    if not doc:
        raise FileNotFoundError(bid)
    return {**doc, "source": "builtin",
            "text": _normalize_book(doc.get("text", ""))}


def api_book(bid: str) -> dict:
    return _book_record(bid)


def api_book_upload(body: dict) -> dict:
    text = _normalize_book(str(body.get("text") or ""))
    if not text.strip():
        raise ValueError("empty text")
    if len(text.encode()) > MAX_BOOK_MB * 1_000_000:
        raise ValueError(f"book too large (> {MAX_BOOK_MB} MB)")
    language = "zh" if body.get("language") == "zh" else "en"
    bid = "u" + hashlib.sha1(text.encode()).hexdigest()[:12]
    UPLOADS.mkdir(parents=True, exist_ok=True)
    (UPLOADS / f"{bid}.json").write_text(json.dumps({
        "bid": bid,
        "title": str(body.get("title") or "untitled")[:120],
        "author": str(body.get("author") or "")[:80],
        "published": "",
        "language": language,
        "words": _count_words(text, language),
        "added": time.strftime("%Y-%m-%d %H:%M"),
        "text": text,
    }, ensure_ascii=False))
    return {"bid": bid}


def api_book_delete(body: dict) -> dict:
    bid = re.sub(r"[^\w.\-]", "", str(body.get("bid", "")))
    f = UPLOADS / f"{bid}.json"
    if not bid.startswith("u") or not f.is_file():
        raise FileNotFoundError(bid)      # built-ins can't be removed
    f.unlink()
    (CHARS_DIR / f"{bid}.json").unlink(missing_ok=True)
    return {"ok": True}


# ------------------------------------------------------------- characters --
def _parse_chars(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for it in (arr if isinstance(arr, list) else []):
        if isinstance(it, dict) and it.get("name"):
            out.append({"name": str(it["name"])[:60],
                        "brief": str(it.get("brief", ""))[:300]})
    return out[:10]


def api_book_characters(body: dict) -> dict:
    """Cast list, extracted once per book from its opening portion and
    cached. cached_only=True never triggers an LLM call."""
    book = _book_record(str(body.get("bid", "")))
    cache = (read_json(CHARS_DIR / f"{book['bid']}.json")
             or read_json(DATA / f"chars__{book['bid']}.json"))
    if cache and not body.get("refresh"):
        return {"characters": cache}
    if body.get("cached_only"):
        return {"characters": None}
    model = body.get("model") or llm.GPT
    excerpt = book["text"][:CHAR_SCAN_CHARS]
    if book.get("language") == "zh":
        prompt = (f"你正在读《{book['title']}》（{book.get('author') or '佚名'}）"
                  "的开头部分。请找出在这部分里真正出场过的主要角色（至多 10 个）。"
                  "每个角色给出：\n"
                  '- "name"：文中最常用的称呼\n'
                  '- "brief"：一句话说明 TA 是谁（用原书的语言）\n\n'
                  '严格输出 JSON 数组，形如 [{"name": "…", "brief": "…"}]，'
                  "不要输出其他任何内容。")
    else:
        prompt = (f'You are reading the opening portion of "{book["title"]}" '
                  f'by {book.get("author") or "an unknown author"}. '
                  "List the main characters who actually appear in this "
                  "portion (at most 10). For each give:\n"
                  '- "name": the name the text uses most often\n'
                  "- \"brief\": one sentence on who they are, in the book's "
                  "own language\n\n"
                  "Output STRICTLY a JSON array like "
                  '[{"name": "...", "brief": "..."}]. Nothing else.')
    # interactive endpoint: fail fast, the visitor is watching a spinner
    raw = llm.query(prompt, model=model, stdin=excerpt,
                    timeout=120, retries=1)
    if raw.startswith("<<ERROR"):
        raise RuntimeError(raw[:300])
    chars = _parse_chars(raw)
    if chars:
        CHARS_DIR.mkdir(parents=True, exist_ok=True)
        (CHARS_DIR / f"{book['bid']}.json").write_text(
            json.dumps(chars, ensure_ascii=False))
    return {"characters": chars}


# ------------------------------------------------------------------- chat --
def api_bookchat(body: dict) -> dict:
    """Talk to a character; they only know the text before the reader's
    position. Prompt builders are mirrored in bookstore.ts; keep in sync."""
    book = _book_record(str(body.get("bid", "")))
    name = str(body.get("char_name") or "").strip()
    if not name:
        raise ValueError("char_name required")
    brief = str(body.get("char_brief") or "").strip()
    upto = max(600, int(body.get("upto", 600)))
    history = body.get("history") or []   # [{speaker:'user'|'char', text}]
    user_name = (body.get("user_name") or "").strip()
    model = body.get("model") or llm.GPT
    lang = "zh" if book.get("language") == "zh" else "en"
    ctx = book["text"][:upto][-BOOKCHAT_CTX_CHARS:]

    if lang == "zh":
        visitor = user_name or "一位不速之客"
        convo = "\n".join(
            (f"{visitor}: {h['text']}" if h["speaker"] == "user"
             else f"{name}: {h['text']}") for h in history)
        prompt = f"""这是一次故事内的对话。这本书在下面摘录的结尾处暂停了，一位书里从未出现过的访客「{visitor}」走进此刻的场景，来到 {name} 面前，与其交谈。

你扮演 {name}。严格依据原文体现的性格、说话方式与当下处境作答；人物只知道故事到此为止发生的事，不知道之后的任何情节，也不知道任何"书外"的信息。若访客问到人物不可能知道的事，人物应表现出困惑，或按自己的理解回应。

输出格式：内心独白用[…]，动作神态用（…），说出的话用「…」。只输出 {name} 的下一个回合，不要代访客说话，不要任何解说。

对话至今：
{convo if convo else '（访客刚刚到来，还未开口。）'}

{visitor} 刚说：{history[-1]['text'] if history and history[-1]['speaker'] == 'user' else '（沉默地看着你。）'}"""
    else:
        visitor = user_name or "a stranger"
        convo = "\n".join(
            (f"{visitor}: {h['text']}" if h["speaker"] == "user"
             else f"{name}: {h['text']}") for h in history)
        prompt = f"""This is an in-story conversation. The book is paused at the end of the excerpt below, and a visitor who never appears in it — "{visitor}" — steps into the scene and approaches {name}.

You play {name}. Stay strictly true to the personality, voice and situation the text shows. The character only knows what has happened in the story up to this point — nothing later, nothing outside the book. If the visitor asks about things the character cannot know, react with confusion or interpret it their own way.

Format: inner thoughts in [brackets], actions in (parentheses), spoken words in "quotes". Output only {name}'s next turn. Never speak for the visitor. No commentary.

Conversation so far:
{convo if convo else '(The visitor has just arrived and not yet spoken.)'}

{visitor} just said: {history[-1]['text'] if history and history[-1]['speaker'] == 'user' else '(watches you silently.)'}"""

    card = [f"BOOK: {book['title']} — {book.get('author') or ''}",
            f"CHARACTER: {name}"]
    if brief:
        card.append(f"ABOUT: {brief}")
    stdin = ("\n".join(card) +
             "\n\n===== THE STORY SO FAR (the reader has read exactly this "
             "far, and no further) =====\n\n" + ctx)
    reply = llm.query(prompt, model=model, stdin=stdin,
                      timeout=120, retries=1)
    return {"reply": reply, "role_name": name}


# ------------------------------------------------------------ saved chats --
def sandbox_list() -> dict:
    items = []
    if SANDBOX.is_dir():
        for f in sorted(SANDBOX.glob("*.json"), reverse=True):
            o = read_json(f)
            if o:
                items.append({k: o.get(k) for k in
                              ("id", "kind", "sid", "tag", "upto", "title",
                               "saved_at", "n_messages")})
    return {"sessions": items}


def sandbox_save(body: dict) -> dict:
    SANDBOX.mkdir(exist_ok=True)
    sess_id = body.get("id") or uuid.uuid4().hex[:12]
    body["id"] = sess_id
    body["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    body["n_messages"] = len(body.get("messages") or [])
    (SANDBOX / f"{sess_id}.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=1))
    return {"id": sess_id}


def sandbox_get(sess_id: str) -> dict:
    o = read_json(SANDBOX / f"{re.sub(r'[^a-f0-9]', '', sess_id)}.json")
    if not o:
        raise FileNotFoundError(sess_id)
    return o


# ------------------------------------------------------------- http glue --
MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css",
        ".svg": "image/svg+xml", ".png": "image/png", ".woff2": "font/woff2",
        ".woff": "font/woff", ".json": "application/json", ".ico": "image/x-icon"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in (str(args[0]) if args else ""):
            sys.stderr.write("[api] %s\n" % (args[0],))

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self):
        try:
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/ping":
                return self._json({"ok": True})
            if path == "/api/books":
                return self._json(api_books())
            m = re.match(r"^/api/book/([\w.\-]+)$", path)
            if m:
                return self._json(api_book(m.group(1)))
            if path == "/api/sandbox/list":
                return self._json(sandbox_list())
            m = re.match(r"^/api/sandbox/([\w]+)$", path)
            if m:
                return self._json(sandbox_get(m.group(1)))
            return self._static(path)
        except FileNotFoundError as e:
            return self._json({"error": f"not found: {e}"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/book/upload":
                return self._json(api_book_upload(body))
            if path == "/api/book/delete":
                return self._json(api_book_delete(body))
            if path == "/api/book/characters":
                return self._json(api_book_characters(body))
            if path == "/api/bookchat":
                return self._json(api_bookchat(body))
            if path == "/api/sandbox/save":
                return self._json(sandbox_save(body))
            return self._json({"error": "unknown endpoint"}, 404)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def _static(self, path: str):
        rel = path.lstrip("/") or "index.html"
        target = (DIST / rel).resolve()
        if not str(target).startswith(str(DIST)) or not target.is_file():
            target = DIST / "index.html"   # SPA fallback
        if not target.is_file():
            return self._send(200, ("frontend not built: run  cd play/app && "
                                    "npm ci && npm run build").encode(),
                              "text/plain; charset=utf-8")
        body = target.read_bytes()
        self._send(200, body, MIME.get(target.suffix, "application/octet-stream"))


def main():
    SANDBOX.mkdir(exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"ANIMASK play page on http://127.0.0.1:{PORT}  (built: {DIST.exists()})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
